"""Train on the fixed spatial training region; validation and test are excluded from losses.

Usage: python train_supervised_multitemporal.py --config config.json
See README.md for input formats, outputs and execution order.
"""

from __future__ import annotations
from pipeline_config import parse_arguments, configure
from pipeline_checks import preflight, record_outputs, record_training_inputs

ARGS = parse_arguments()
import datetime
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE = 32
PATCH = 256
BATCH = 12
EPOCHS = 50
LABELED_SAMPLES_PER_EPOCH = 8000
LR = 0.0003
WEIGHT_DECAY = 0.0001
SEED = 42
IN_CH = 4
NUM_WORKERS = 0
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2
PIN_MEMORY = True
EPS = 1e-06
globals().update(
    configure(
        ARGS,
        "supervised_multitemporal",
        "train",
        {k: v for k, v in globals().copy().items() if k.isupper()},
    )
)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")


def require_dir(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_2d(data: dict, key: str, path: Path) -> np.ndarray:
    if key not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{key}'. Keys: {keys}")
    array = np.squeeze(data[key])
    if array.ndim != 2:
        raise ValueError(f"{path.name}: '{key}' must be 2-D, got {array.shape}")
    return array


def load_img_mat(path: Path) -> np.ndarray:
    data = loadmat(path)
    if IMAGE_KEY not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Keys: {keys}")
    image = np.squeeze(data[IMAGE_KEY])
    if image.ndim != 3 or image.shape[2] != 2:
        raise ValueError(f"{path.name}: expected (H,W,2), got {image.shape}")
    image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)):
        raise RuntimeError(f"{path.name} contains NaN/Inf values.")
    return image


def get_single_after():
    require_file(AFTER_PATH, "AFTER acquisition")
    return AFTER_PATH


def get_all_before_files() -> list[Path]:
    files = sorted(BEFORE_DIR.glob("*.mat"))
    if len(files) != EXPECTED_BEFORE_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_BEFORE_COUNT} BEFORE MAT files in {BEFORE_DIR}, found {len(files)}."
        )
    return files


def load_data() -> dict[str, np.ndarray]:
    require_file(GT_MAT, "Verified SAR GT")
    require_file(SPLIT_MAT, "Frozen final SAR split")
    gt_data = loadmat(GT_MAT)
    split_data = loadmat(SPLIT_MAT)
    gt = load_2d(gt_data, "ground_truth", GT_MAT).astype(np.uint8)
    gt_valid = load_2d(gt_data, "valid_mask", GT_MAT).astype(bool)
    names = [
        "valid_mask",
        "train_pool",
        "labeled_train",
        "unlabeled_train",
        "validation",
        "test",
        "buffer_mask",
    ]
    masks = {name: load_2d(split_data, name, SPLIT_MAT).astype(bool) for name in names}
    shape = gt.shape
    if gt_valid.shape != shape:
        raise ValueError("GT ground_truth and valid_mask shape mismatch.")
    for name, mask in masks.items():
        if mask.shape != shape:
            raise ValueError(f"{name} shape {mask.shape} != GT {shape}")
    if not np.all(np.isin(np.unique(gt), [0, 1])):
        raise ValueError("Verified GT must contain only 0/1.")
    if not np.array_equal(gt_valid, masks["valid_mask"]):
        raise RuntimeError("Verified GT valid_mask differs from split valid_mask.")
    if "ground_truth" in split_data:
        split_gt = np.squeeze(split_data["ground_truth"]).astype(np.uint8)
        if not np.array_equal(split_gt, gt):
            raise RuntimeError("GT stored in split MAT differs from verified GT.")
    train_pool = masks["train_pool"]
    labeled = masks["labeled_train"]
    unlabeled = masks["unlabeled_train"]
    validation = masks["validation"]
    test = masks["test"]
    buffer_mask = masks["buffer_mask"]
    if not np.array_equal(labeled | unlabeled, train_pool):
        raise RuntimeError("labeled_train + unlabeled_train do not reconstruct train_pool.")
    if np.any(labeled & unlabeled):
        raise RuntimeError("labeled_train overlaps unlabeled_train.")
    for a_name, a in [("train_pool", train_pool), ("validation", validation), ("test", test)]:
        for b_name, b in [("train_pool", train_pool), ("validation", validation), ("test", test)]:
            if a_name >= b_name:
                continue
            overlap = int((a & b).sum())
            if overlap:
                raise RuntimeError(f"{a_name} overlaps {b_name} by {overlap:,} pixels.")
    if np.any(buffer_mask & (train_pool | validation | test)):
        raise RuntimeError("buffer_mask overlaps usable regions.")
    if not np.array_equal(train_pool | validation | test | buffer_mask, masks["valid_mask"]):
        raise RuntimeError("train/validation/test/buffer do not reconstruct valid_mask.")
    return {"ground_truth": gt, **masks}


def robust_norm_from_train_pool(
    image: np.ndarray, train_pool: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if image.shape[1:] != train_pool.shape:
        raise ValueError("Image and train_pool shape mismatch.")
    output = np.empty_like(image, dtype=np.float32)
    stats = []
    for channel in range(image.shape[0]):
        train_values = image[channel][train_pool]
        if train_values.size == 0:
            raise RuntimeError("train_pool is empty.")
        low = float(np.percentile(train_values, 1))
        high = float(np.percentile(train_values, 99))
        if high <= low:
            raise RuntimeError(f"Channel {channel}: invalid percentile range {low} -> {high}.")
        clipped_train = np.clip(train_values, low, high)
        mean = float(clipped_train.mean())
        std = float(clipped_train.std())
        if std <= EPS:
            raise RuntimeError(f"Channel {channel}: near-zero std={std}")
        clipped_full = np.clip(image[channel], low, high)
        output[channel] = (clipped_full - mean) / (std + EPS)
        stats.append({"low": low, "high": high, "mean": mean, "std": std})
    return (output, stats)


def aug_flip_rot_pair(x: np.ndarray, y: np.ndarray, valid: np.ndarray):
    k = random.randint(0, 3)
    if k:
        x = np.rot90(x, k, axes=(1, 2)).copy()
        y = np.rot90(y, k, axes=(0, 1)).copy()
        valid = np.rot90(valid, k, axes=(0, 1)).copy()
    if random.random() < 0.5:
        x = x[:, :, ::-1].copy()
        y = y[:, ::-1].copy()
        valid = valid[:, ::-1].copy()
    if random.random() < 0.5:
        x = x[:, ::-1, :].copy()
        y = y[::-1, :].copy()
        valid = valid[::-1, :].copy()
    return (x, y, valid)


def window_sums(mask: np.ndarray, patch: int) -> np.ndarray:
    h, w = mask.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int32)
    integral[1:, 1:] = np.cumsum(
        np.cumsum(mask.astype(np.int32, copy=False), axis=0, dtype=np.int32), axis=1, dtype=np.int32
    )
    sums = (
        integral[patch:, patch:]
        - integral[:-patch, patch:]
        - integral[patch:, :-patch]
        + integral[:-patch, :-patch]
    )
    return sums


def build_labeled_candidates(
    labeled_mask: np.ndarray, train_pool: np.ndarray, patch: int
) -> tuple[np.ndarray, int]:
    h, w = train_pool.shape
    if h < patch or w < patch:
        raise ValueError(f"PATCH={patch} larger than mask shape {(h, w)}.")
    print("Computing fully-in-train_pool patch mask...")
    train_sums = window_sums(train_pool, patch)
    fully_train = train_sums == patch * patch
    del train_sums
    print("Computing labeled-overlap patch mask...")
    labeled_sums = window_sums(labeled_mask, patch)
    candidate_mask = fully_train & (labeled_sums > 0)
    del labeled_sums
    del fully_train
    candidate_flat = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)
    del candidate_mask
    if candidate_flat.size == 0:
        raise RuntimeError(
            "No training patches satisfy both:\n  - 100% inside train_pool\n  - at least one labeled pixel."
        )
    n_left_positions = w - patch + 1
    return (candidate_flat, n_left_positions)


class MultiTemporalLabeledPatchDataset(Dataset):

    def __init__(
        self,
        before_stack: np.ndarray,
        after: np.ndarray,
        ground_truth: np.ndarray,
        labeled_region_mask: np.ndarray,
        train_pool: np.ndarray,
        patch: int = PATCH,
        samples: int = LABELED_SAMPLES_PER_EPOCH,
        train: bool = True,
    ):
        super().__init__()
        self.before_stack = before_stack
        self.after = after
        self.ground_truth = ground_truth.astype(np.float32, copy=False)
        self.labeled_mask = labeled_region_mask.astype(np.float32, copy=False)
        self.train_pool = train_pool.astype(bool, copy=False)
        self.patch = int(patch)
        self.samples = int(samples)
        self.train = bool(train)
        if self.before_stack.ndim != 4 or self.before_stack.shape[1] != 2:
            raise ValueError("before_stack must have shape (T,2,H,W).")
        if self.before_stack.shape[0] != EXPECTED_BEFORE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_BEFORE_COUNT} BEFORE acquisitions, got {self.before_stack.shape[0]}."
            )
        if self.before_stack.shape[2:] != self.after.shape[1:]:
            raise ValueError("BEFORE stack and AFTER shapes differ.")
        if self.ground_truth.shape != self.before_stack.shape[2:]:
            raise ValueError("GT shape differs from image shape.")
        print("\nBuilding final supervised patch candidates...")
        self.candidate_flat, self.n_left_positions = build_labeled_candidates(
            labeled_mask=labeled_region_mask, train_pool=train_pool, patch=self.patch
        )
        print("Accepted training patch top-lefts:", f"{self.candidate_flat.size:,}")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index):
        del index
        candidate_index = random.randrange(self.candidate_flat.size)
        flat = int(self.candidate_flat[candidate_index])
        top, left = divmod(flat, self.n_left_positions)
        before_index = random.randrange(self.before_stack.shape[0])
        xb = self.before_stack[before_index, :, top : top + self.patch, left : left + self.patch]
        xa = self.after[:, top : top + self.patch, left : left + self.patch]
        y = self.ground_truth[top : top + self.patch, left : left + self.patch]
        valid = self.labeled_mask[top : top + self.patch, left : left + self.patch]
        train_patch = self.train_pool[top : top + self.patch, left : left + self.patch]
        if not np.all(train_patch):
            raise RuntimeError("Candidate patch is not fully inside train_pool.")
        if not np.any(valid):
            raise RuntimeError("Candidate patch contains no labeled pixels.")
        x = np.concatenate([xb, xa], axis=0).astype(np.float32, copy=False)
        if self.train:
            x, y, valid = aug_flip_rot_pair(x, y, valid)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y[None, ...].astype(np.float32, copy=False)),
            torch.from_numpy(valid[None, ...].astype(np.float32, copy=False)),
        )


def conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):

    def __init__(self, in_ch: int = IN_CH, base: int = BASE):
        super().__init__()
        self.e1 = conv_block(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = conv_block(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = conv_block(base * 2, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)
        x2 = self.e2(self.p1(x1))
        x3 = self.e3(self.p2(x2))
        y = self.u2(x3)
        y = self.d2(torch.cat([y, x2], dim=1))
        y = self.u1(y)
        y = self.d1(torch.cat([y, x1], dim=1))
        return self.out(y)


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    loss_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denominator = valid.sum() + EPS
    return (loss_map * valid).sum() / denominator


def masked_dice_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits) * valid
    targets = targets * valid
    numerator = 2.0 * (probabilities * targets).sum(dim=(2, 3))
    denominator = (probabilities + targets).sum(dim=(2, 3)) + eps
    dice = numerator / denominator
    return 1.0 - dice.mean()


def make_dataloader(dataset: Dataset) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**kwargs)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_directory: Path,
    before_names: list[str],
    after_name: str,
    split_hash: str,
    gt_hash: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "run_dir": str(run_directory),
            "before_names": list(before_names),
            "temporal_training_policy": "uniform random BEFORE acquisition per sampled patch",
            "after_name": after_name,
            "split_sha256": split_hash,
            "gt_sha256": gt_hash,
            "config": {
                "BASE": BASE,
                "PATCH": PATCH,
                "BATCH": BATCH,
                "EPOCHS": EPOCHS,
                "LABELED_SAMPLES_PER_EPOCH": LABELED_SAMPLES_PER_EPOCH,
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "SEED": SEED,
                "IN_CH": IN_CH,
                "EXPECTED_BEFORE_COUNT": EXPECTED_BEFORE_COUNT,
                "TEMPORAL_POLICY": "uniform random BEFORE acquisition per sampled patch",
            },
        },
        path,
    )


def main() -> None:
    preflight(globals(), "supervised_multitemporal", "train")
    if RUN_DIR.exists() and not RESUME:
        raise FileExistsError(
            f"Training run already exists: {RUN_DIR}; choose a new run_name or resume."
        )
    seed_all(SEED)
    require_dir(BEFORE_DIR, "BEFORE directory")
    require_dir(AFTER_DIR, "AFTER directory")
    regions = load_data()
    gt = regions["ground_truth"]
    valid_mask = regions["valid_mask"]
    train_pool = regions["train_pool"]
    labeled = regions["labeled_train"]
    unlabeled = regions["unlabeled_train"]
    validation = regions["validation"]
    test = regions["test"]
    buffer_mask = regions["buffer_mask"]
    split_hash = sha256_file(SPLIT_MAT)
    gt_hash = sha256_file(GT_MAT)
    after_path = get_single_after()
    before_paths = get_all_before_files()
    before_names = [p.name for p in before_paths]
    print(f"Found {len(before_paths)} BEFORE acquisitions:")
    for i, p in enumerate(before_paths, 1):
        print(f" {i:02d}: {p.name}")
    print("\nLoading AFTER:")
    print(after_path.name)
    raw_after = load_img_mat(after_path)
    if raw_after.shape[1:] != gt.shape:
        raise ValueError("AFTER/GT shapes do not match.")
    before_stack = np.empty((len(before_paths), 2, gt.shape[0], gt.shape[1]), dtype=np.float32)
    before_stats_by_file = {}
    print("\nNormalizing all BEFORE acquisitions from train_pool only...")
    for before_index, before_path in enumerate(before_paths):
        print(f"[{before_index + 1:02d}/{len(before_paths):02d}] {before_path.name}")
        raw_before = load_img_mat(before_path)
        if raw_before.shape[1:] != gt.shape:
            raise ValueError(f"{before_path.name}: shape {raw_before.shape[1:]} != GT {gt.shape}")
        normalized_before, current_stats = robust_norm_from_train_pool(raw_before, train_pool)
        before_stack[before_index] = normalized_before
        before_stats_by_file[before_path.name] = current_stats
        del raw_before, normalized_before
    print("\nFitting AFTER normalization from train_pool only...")
    after, after_stats = robust_norm_from_train_pool(raw_after, train_pool)
    del raw_after
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if RESUME:
        if RESUME_CKPT_PATH is None:
            raise ValueError("RESUME=True but RESUME_CKPT_PATH is None.")
        resume_path = Path(RESUME_CKPT_PATH)
        require_file(resume_path, "Resume checkpoint")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("split_sha256") != split_hash:
            raise RuntimeError("Resume checkpoint uses a different split.")
        if checkpoint.get("gt_sha256") != gt_hash:
            raise RuntimeError("Resume checkpoint uses a different GT.")
        run_directory = Path(checkpoint["run_dir"])
        if (
            checkpoint.get("before_names") != before_names
            or checkpoint["after_name"] != after_path.name
        ):
            raise RuntimeError("Resume checkpoint acquisition set mismatch.")
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        timestamp = RUN_NAME
        run_directory = RUN_DIR
        run_directory.mkdir(parents=True, exist_ok=False)
        record_training_inputs(globals(), "supervised_multitemporal")
        start_epoch = 1
        np.savez_compressed(
            run_directory / "split_snapshot.npz",
            train_pool=train_pool.astype(np.uint8),
            labeled_train=labeled.astype(np.uint8),
            unlabeled_train=unlabeled.astype(np.uint8),
            validation=validation.astype(np.uint8),
            test=test.astype(np.uint8),
            buffer_mask=buffer_mask.astype(np.uint8),
            valid_mask=valid_mask.astype(np.uint8),
        )
        with (run_directory / "normalization_stats.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "policy": "per-acquisition robust normalization fitted on train_pool pixels only",
                    "before_files": before_names,
                    "after_file": after_path.name,
                    "before": before_stats_by_file,
                    "after": after_stats,
                    "temporal_policy": "uniform random BEFORE acquisition per sampled patch",
                },
                f,
                indent=2,
            )
        config = {
            "method": "final multi-temporal supervised SAR SmallUNet",
            "BASE": BASE,
            "PATCH": PATCH,
            "BATCH": BATCH,
            "EPOCHS": EPOCHS,
            "LABELED_SAMPLES_PER_EPOCH": LABELED_SAMPLES_PER_EPOCH,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "SEED": SEED,
            "IN_CH": IN_CH,
            "NUM_WORKERS": NUM_WORKERS,
            "before_files": before_names,
            "before_count": len(before_names),
            "after_file": after_path.name,
            "temporal_training_policy": "uniform random BEFORE acquisition per sampled patch",
            "input_channels_per_sample": "selected BEFORE VH/VV + fixed AFTER VH/VV = 4",
            "split_mat": str(SPLIT_MAT),
            "split_sha256": split_hash,
            "gt_mat": str(GT_MAT),
            "gt_sha256": gt_hash,
            "valid_pixels": int(valid_mask.sum()),
            "train_pool_pixels": int(train_pool.sum()),
            "labeled_pixels": int(labeled.sum()),
            "unlabeled_pixels": int(unlabeled.sum()),
            "validation_pixels": int(validation.sum()),
            "test_pixels": int(test.sum()),
            "buffer_pixels": int(buffer_mask.sum()),
            "labeled_fraction_of_train": float(labeled.sum() / train_pool.sum()),
            "labeled_positive_prevalence": float(gt[labeled].mean()),
            "normalization_fit_region": "train_pool only",
            "loss_supervision_region": "labeled_train only",
            "patch_constraint": "every training patch 100% inside train_pool",
            "validation_labels_used_in_training": False,
            "test_labels_used_in_training": False,
        }
        with (run_directory / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL SUPERVISED SAR TRAINING")
    print("=" * 80)
    print("Device:", device)
    print("Image shape:", gt.shape)
    print(f"Valid GT pixels:    {int(valid_mask.sum()):,}")
    print(f"Training pool:      {int(train_pool.sum()):,}")
    print(
        f"Labeled training:   {int(labeled.sum()):,} ({100 * labeled.sum() / train_pool.sum():.2f}% of train_pool)"
    )
    print(f"Unlabeled training: {int(unlabeled.sum()):,}")
    print(f"Validation pixels:  {int(validation.sum()):,}")
    print(f"Held-out test:      {int(test.sum()):,}")
    print(f"Spatial buffer:     {int(buffer_mask.sum()):,}")
    print(f"Labeled positive prevalence: {100 * gt[labeled].mean():.2f}%")
    print("\nValidation/test class labels are not inspected during training.")
    print(f"\nBEFORE acquisitions: {len(before_names)}")
    for i, name in enumerate(before_names, 1):
        print(f" {i:02d}: {name}")
    print("Temporal training policy:")
    print(" uniform random BEFORE acquisition per sampled patch")
    print("AFTER:", after_path.name)
    print("Run directory:")
    print(run_directory)
    print("\nBEFORE normalization (each acquisition fitted independently on train_pool):")
    for name in before_names:
        print(f" {name}")
        for i, s in enumerate(before_stats_by_file[name]):
            print(
                f"   channel {i}: low={s['low']:.6f}, high={s['high']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}"
            )
    print("\nAFTER normalization:")
    for i, s in enumerate(after_stats):
        print(
            f" channel {i}: low={s['low']:.6f}, high={s['high']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}"
        )
    train_dataset = MultiTemporalLabeledPatchDataset(
        before_stack=before_stack,
        after=after,
        ground_truth=gt,
        labeled_region_mask=labeled,
        train_pool=train_pool,
        patch=PATCH,
        samples=LABELED_SAMPLES_PER_EPOCH,
        train=True,
    )
    train_loader = make_dataloader(train_dataset)
    model = SmallUNet(in_ch=IN_CH, base=BASE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if RESUME:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Resuming at epoch:", start_epoch)
    log_file = run_directory / "training_log.csv"
    if not RESUME:
        with log_file.open("w", encoding="utf-8") as f:
            f.write("epoch,total_loss\n")
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        losses = []
        for x, y, v in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = masked_bce_with_logits(logits, y, v) + masked_dice_loss_with_logits(logits, y, v)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch:03d} | loss={mean_loss:.6f}")
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{mean_loss:.8f}\n")
        save_checkpoint(
            path=run_directory / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            run_directory=run_directory,
            before_names=before_names,
            after_name=after_path.name,
            split_hash=split_hash,
            gt_hash=gt_hash,
        )
        save_checkpoint(
            path=run_directory / "supervised_last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            run_directory=run_directory,
            before_names=before_names,
            after_name=after_path.name,
            split_hash=split_hash,
            gt_hash=gt_hash,
        )
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL SUPERVISED SAR TRAINING FINISHED")
    print("=" * 80)
    print("Saved to:")
    print(run_directory)
    print("\nNext step:")
    print(
        "Run multi-temporal full-scene inference for configured epochs (average predictions across all BEFORE acquisitions), then choose epoch / sigma / threshold using VALIDATION ONLY and evaluate TEST once."
    )


if __name__ == "__main__":
    main()
    record_outputs(globals(), "supervised_multitemporal", "train")
