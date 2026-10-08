"""Train on the fixed spatial training region; validation and test are excluded from losses.

Usage: python "semisupervised_multitemporal/train_semisupervised_multitemporal.py" --config config.json
See README.md for input formats, outputs and execution order.
"""

from __future__ import annotations
# Locate shared helpers when this entry point runs from a method subfolder.
import sys as _sys
from pathlib import Path as _RepoPath

_REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pipeline_config import parse_arguments, configure
from pipeline_checks import preflight, record_outputs, record_training_inputs

ARGS = parse_arguments()
import datetime
import hashlib
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE = 32
PATCH = 256
BATCH_L = 4
BATCH_U = 8
EPOCHS = 50
LABELED_SAMPLES_PER_EPOCH = 2000
UNLABELED_SAMPLES_PER_EPOCH = 4000
LR = 0.0003
WEIGHT_DECAY = 0.0001
SEED = 42
IN_CH = 4
EMA_DECAY = 0.99
LAMBDA_MAX = 1.0
RAMPUP_EPOCHS = 10
NOISE_STD = 0.03
CONF_LOW_START = 0.4
CONF_HIGH_START = 0.6
CONF_LOW_END = 0.15
CONF_HIGH_END = 0.85
MIN_LABELED_PIXELS_IN_PATCH = 1
MIN_UNLABELED_PIXELS_IN_PATCH = 1
NUM_WORKERS = 0
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2
EPS = 1e-06
globals().update(
    configure(
        ARGS,
        "semisupervised_multitemporal",
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_required_2d(data: dict[str, Any], key: str, source: Path) -> np.ndarray:
    if key not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"{source.name} missing '{key}'. Available keys: {keys}")
    arr = np.squeeze(data[key])
    if arr.ndim != 2:
        raise ValueError(f"{source.name}: '{key}' must be 2-D, got {arr.shape}.")
    return arr


def get_all_before_files() -> list[Path]:
    files = sorted(BEFORE_DIR.glob("*.mat"))
    if len(files) != EXPECTED_BEFORE_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_BEFORE_COUNT} BEFORE MAT files in:\n{BEFORE_DIR}\nFound {len(files)}."
        )
    return files


def get_single_after():
    require_file(AFTER_PATH, "AFTER acquisition")
    return AFTER_PATH


def load_training_regions() -> dict[str, np.ndarray]:
    require_file(GT_MAT, "Verified SAR GT")
    require_file(SPLIT_MAT, "Frozen final SAR split")
    gt_data = loadmat(GT_MAT)
    split_data = loadmat(SPLIT_MAT)
    ground_truth = load_required_2d(gt_data, "ground_truth", GT_MAT).astype(np.uint8, copy=False)
    gt_valid = load_required_2d(gt_data, "valid_mask", GT_MAT).astype(bool, copy=False)
    names = [
        "train_pool",
        "labeled_train",
        "unlabeled_train",
        "validation",
        "test",
        "buffer_mask",
        "valid_mask",
    ]
    regions = {
        name: load_required_2d(split_data, name, SPLIT_MAT).astype(bool, copy=False)
        for name in names
    }
    shapes = {ground_truth.shape, gt_valid.shape, *[mask.shape for mask in regions.values()]}
    if len(shapes) != 1:
        raise ValueError(f"GT/split shape mismatch: {shapes}")
    if not np.array_equal(gt_valid, regions["valid_mask"]):
        mismatch = int(np.count_nonzero(gt_valid != regions["valid_mask"]))
        raise RuntimeError(f"GT valid_mask and split valid_mask differ at {mismatch:,} pixels.")
    if not np.array_equal(
        regions["labeled_train"] | regions["unlabeled_train"], regions["train_pool"]
    ):
        raise RuntimeError("labeled_train + unlabeled_train do not reconstruct train_pool.")
    if np.any(regions["labeled_train"] & regions["unlabeled_train"]):
        raise RuntimeError("labeled_train and unlabeled_train overlap.")
    for left_name, right_name in [
        ("train_pool", "validation"),
        ("train_pool", "test"),
        ("validation", "test"),
        ("train_pool", "buffer_mask"),
        ("validation", "buffer_mask"),
        ("test", "buffer_mask"),
    ]:
        if np.any(regions[left_name] & regions[right_name]):
            raise RuntimeError(f"{left_name} overlaps {right_name}.")
    rebuilt_valid = (
        regions["train_pool"] | regions["validation"] | regions["test"] | regions["buffer_mask"]
    )
    if not np.array_equal(rebuilt_valid, regions["valid_mask"]):
        raise RuntimeError("train_pool / validation / test / buffer do not reconstruct valid_mask.")
    if not np.all(np.isin(np.unique(ground_truth[gt_valid]), [0, 1])):
        raise ValueError("Valid GT pixels must be binary 0/1.")
    training_target = np.zeros(ground_truth.shape, dtype=np.float32)
    training_target[regions["labeled_train"]] = ground_truth[regions["labeled_train"]].astype(
        np.float32
    )
    labeled_pixels = int(regions["labeled_train"].sum())
    labeled_positive = int(training_target[regions["labeled_train"]].sum())
    labeled_prevalence = labeled_positive / labeled_pixels
    del ground_truth
    regions["training_target"] = training_target
    regions["labeled_prevalence"] = np.array(labeled_prevalence, dtype=np.float64)
    return regions


def load_img_mat(path: Path) -> np.ndarray:
    require_file(path, "SAR MAT")
    data = loadmat(path)
    if IMAGE_KEY not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Available keys: {keys}")
    image = np.squeeze(data[IMAGE_KEY])
    if image.ndim != 3 or image.shape[2] != 2:
        raise ValueError(f"{path.name}: expected (H,W,2), got {image.shape}.")
    image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)):
        raise RuntimeError(f"{path.name} contains NaN/Inf.")
    return image


def robust_norm_from_train_pool(
    image: np.ndarray, train_pool: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if image.shape[1:] != train_pool.shape:
        raise ValueError("Image and train_pool shapes differ.")
    output = np.empty_like(image, dtype=np.float32)
    stats = []
    for channel in range(image.shape[0]):
        train_values = image[channel][train_pool]
        if train_values.size == 0:
            raise RuntimeError("train_pool has no pixels.")
        low = float(np.percentile(train_values, 1))
        high = float(np.percentile(train_values, 99))
        if high <= low:
            raise RuntimeError(f"Channel {channel}: invalid clipping range.")
        clipped_train = np.clip(train_values, low, high)
        mean = float(clipped_train.mean())
        std = float(clipped_train.std())
        if std <= EPS:
            raise RuntimeError(f"Channel {channel}: near-zero std.")
        clipped_full = np.clip(image[channel], low, high)
        output[channel] = (clipped_full - mean) / (std + EPS)
        stats.append({"low": low, "high": high, "mean": mean, "std": std})
    return (output, stats)


def patch_sum_map(mask: np.ndarray, patch: int) -> np.ndarray:
    if mask.shape[0] < patch or mask.shape[1] < patch:
        raise ValueError("PATCH is larger than image.")
    integral = np.pad(mask.astype(np.int64, copy=False), ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0, dtype=np.int64), axis=1, dtype=np.int64)
    return (
        integral[patch:, patch:]
        - integral[:-patch, patch:]
        - integral[patch:, :-patch]
        + integral[:-patch, :-patch]
    )


def build_training_candidates(
    train_pool: np.ndarray, target_region: np.ndarray, minimum_target_pixels: int, label: str
) -> np.ndarray:
    patch_area = PATCH * PATCH
    print(f"Computing fully-in-train_pool mask for {label} candidates...")
    train_counts = patch_sum_map(train_pool, PATCH)
    fully_inside_train = train_counts == patch_area
    del train_counts
    print(f"Computing {label}-overlap mask...")
    target_counts = patch_sum_map(target_region, PATCH)
    accepted = fully_inside_train & (target_counts >= minimum_target_pixels)
    del target_counts
    del fully_inside_train
    coordinates = np.argwhere(accepted).astype(np.int32, copy=False)
    del accepted
    if coordinates.size == 0:
        raise RuntimeError(f"No valid {label} patch candidates.")
    print(f"Accepted {label} patch top-lefts:", f"{coordinates.shape[0]:,}")
    return coordinates


def aug_flip_rot_labeled(image: np.ndarray, target: np.ndarray, valid: np.ndarray):
    rotations = random.randint(0, 3)
    if rotations:
        image = np.rot90(image, rotations, axes=(1, 2)).copy()
        target = np.rot90(target, rotations, axes=(0, 1)).copy()
        valid = np.rot90(valid, rotations, axes=(0, 1)).copy()
    if random.random() < 0.5:
        image = image[:, :, ::-1].copy()
        target = target[:, ::-1].copy()
        valid = valid[:, ::-1].copy()
    if random.random() < 0.5:
        image = image[:, ::-1, :].copy()
        target = target[::-1, :].copy()
        valid = valid[::-1, :].copy()
    return (image, target, valid)


def aug_flip_rot_unlabeled(image: np.ndarray, valid: np.ndarray):
    rotations = random.randint(0, 3)
    if rotations:
        image = np.rot90(image, rotations, axes=(1, 2)).copy()
        valid = np.rot90(valid, rotations, axes=(0, 1)).copy()
    if random.random() < 0.5:
        image = image[:, :, ::-1].copy()
        valid = valid[:, ::-1].copy()
    if random.random() < 0.5:
        image = image[:, ::-1, :].copy()
        valid = valid[::-1, :].copy()
    return (image, valid)


def add_small_noise(tensor: torch.Tensor, std: float = NOISE_STD) -> torch.Tensor:
    if std <= 0:
        return tensor
    return tensor + std * torch.randn_like(tensor)


class MultiTemporalLabeledPatchDataset(Dataset):

    def __init__(
        self,
        before_stack: np.ndarray,
        after: np.ndarray,
        training_target: np.ndarray,
        labeled_region: np.ndarray,
        candidates: np.ndarray,
        samples: int = LABELED_SAMPLES_PER_EPOCH,
    ):
        super().__init__()
        self.before_stack = before_stack
        self.after = after
        self.training_target = training_target
        self.labeled_region = labeled_region
        self.candidates = candidates
        self.samples = int(samples)
        if self.before_stack.ndim != 4 or self.before_stack.shape[1] != 2:
            raise ValueError("before_stack must have shape (T,2,H,W).")
        if self.before_stack.shape[0] != EXPECTED_BEFORE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_BEFORE_COUNT} BEFOREs, got {self.before_stack.shape[0]}."
            )
        if self.before_stack.shape[2:] != self.after.shape[1:]:
            raise ValueError("BEFORE stack and AFTER shapes differ.")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        del index
        candidate_index = random.randrange(self.candidates.shape[0])
        top = int(self.candidates[candidate_index, 0])
        left = int(self.candidates[candidate_index, 1])
        before_index = random.randrange(self.before_stack.shape[0])
        before_patch = self.before_stack[before_index, :, top : top + PATCH, left : left + PATCH]
        after_patch = self.after[:, top : top + PATCH, left : left + PATCH]
        target = self.training_target[top : top + PATCH, left : left + PATCH]
        valid = self.labeled_region[top : top + PATCH, left : left + PATCH]
        model_input = np.concatenate([before_patch, after_patch], axis=0).astype(
            np.float32, copy=False
        )
        model_input, target, valid = aug_flip_rot_labeled(model_input, target, valid)
        return (
            torch.from_numpy(model_input),
            torch.from_numpy(target[None, ...].astype(np.float32, copy=False)),
            torch.from_numpy(valid[None, ...].astype(np.float32, copy=False)),
        )


class MultiTemporalUnlabeledPatchDataset(Dataset):

    def __init__(
        self,
        before_stack: np.ndarray,
        after: np.ndarray,
        unlabeled_region: np.ndarray,
        candidates: np.ndarray,
        samples: int = UNLABELED_SAMPLES_PER_EPOCH,
    ):
        super().__init__()
        self.before_stack = before_stack
        self.after = after
        self.unlabeled_region = unlabeled_region
        self.candidates = candidates
        self.samples = int(samples)
        if self.before_stack.shape[0] != EXPECTED_BEFORE_COUNT:
            raise ValueError(f"Expected {EXPECTED_BEFORE_COUNT} BEFOREs.")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        del index
        candidate_index = random.randrange(self.candidates.shape[0])
        top = int(self.candidates[candidate_index, 0])
        left = int(self.candidates[candidate_index, 1])
        before_index = random.randrange(self.before_stack.shape[0])
        before_patch = self.before_stack[before_index, :, top : top + PATCH, left : left + PATCH]
        after_patch = self.after[:, top : top + PATCH, left : left + PATCH]
        valid = self.unlabeled_region[top : top + PATCH, left : left + PATCH]
        model_input = np.concatenate([before_patch, after_patch], axis=0).astype(
            np.float32, copy=False
        )
        model_input, valid = aug_flip_rot_unlabeled(model_input, valid)
        return (
            torch.from_numpy(model_input),
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
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities * valid
    targets = targets * valid
    numerator = 2.0 * (probabilities * targets).sum(dim=(2, 3))
    denominator = (probabilities + targets).sum(dim=(2, 3)) + EPS
    dice = numerator / denominator
    return 1.0 - dice.mean()


def confidence_filtered_mse(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    unlabeled_valid: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
):
    confidence_mask = (
        (teacher_probabilities < low_threshold) | (teacher_probabilities > high_threshold)
    ).float()
    effective_mask = confidence_mask * unlabeled_valid
    denominator = effective_mask.sum() + EPS
    squared_difference = (student_probabilities - teacher_probabilities) ** 2
    loss = (squared_difference * effective_mask).sum() / denominator
    confidence_ratio = (effective_mask.sum() / (unlabeled_valid.sum() + EPS)).detach()
    return (loss, confidence_ratio)


def rampup(epoch: int, ramp_epochs: int = RAMPUP_EPOCHS, maximum: float = LAMBDA_MAX) -> float:
    if epoch <= 0:
        return 0.0
    if epoch >= ramp_epochs:
        return maximum
    return maximum * epoch / ramp_epochs


def get_confidence_thresholds(epoch: int, maximum_epochs: int) -> tuple[float, float]:
    if maximum_epochs <= 1:
        interpolation = 1.0
    else:
        interpolation = (epoch - 1) / (maximum_epochs - 1)
    low = CONF_LOW_START + interpolation * (CONF_LOW_END - CONF_LOW_START)
    high = CONF_HIGH_START + interpolation * (CONF_HIGH_END - CONF_HIGH_START)
    return (float(low), float(high))


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)


def make_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
        "drop_last": True,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**kwargs)


def save_checkpoint(
    checkpoint_path: Path,
    student: nn.Module,
    teacher: nn.Module,
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
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "run_dir": str(run_directory),
            "before_names": list(before_names),
            "after_name": after_name,
            "split_sha256": split_hash,
            "gt_sha256": gt_hash,
            "temporal_training_policy": "uniform random BEFORE acquisition per labeled and unlabeled sample",
            "config": {
                "experiment": "final_multitemporal_mean_teacher",
                "BASE": BASE,
                "PATCH": PATCH,
                "BATCH_L": BATCH_L,
                "BATCH_U": BATCH_U,
                "EPOCHS": EPOCHS,
                "LABELED_SAMPLES_PER_EPOCH": LABELED_SAMPLES_PER_EPOCH,
                "UNLABELED_SAMPLES_PER_EPOCH": UNLABELED_SAMPLES_PER_EPOCH,
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "SEED": SEED,
                "IN_CH": IN_CH,
                "EXPECTED_BEFORE_COUNT": EXPECTED_BEFORE_COUNT,
                "EMA_DECAY": EMA_DECAY,
                "LAMBDA_MAX": LAMBDA_MAX,
                "RAMPUP_EPOCHS": RAMPUP_EPOCHS,
                "NOISE_STD": NOISE_STD,
                "CONF_LOW_START": CONF_LOW_START,
                "CONF_HIGH_START": CONF_HIGH_START,
                "CONF_LOW_END": CONF_LOW_END,
                "CONF_HIGH_END": CONF_HIGH_END,
                "consistency_region": "unlabeled_train only",
                "patch_context": "100% inside train_pool",
                "normalization": "per acquisition; train_pool only",
                "validation_used_during_training": False,
                "test_used_during_training": False,
            },
        },
        checkpoint_path,
    )


def main() -> None:
    preflight(globals(), "semisupervised_multitemporal", "train")
    if RUN_DIR.exists() and not RESUME:
        raise FileExistsError(
            f"Training run already exists: {RUN_DIR}; choose a new run_name or resume."
        )
    seed_all(SEED)
    regions = load_training_regions()
    train_pool = regions["train_pool"]
    labeled_train = regions["labeled_train"]
    unlabeled_train = regions["unlabeled_train"]
    validation = regions["validation"]
    test = regions["test"]
    buffer_mask = regions["buffer_mask"]
    training_target = regions["training_target"]
    labeled_prevalence = float(regions["labeled_prevalence"])
    split_hash = sha256_file(SPLIT_MAT)
    gt_hash = sha256_file(GT_MAT)
    before_paths = get_all_before_files()
    before_names = [path.name for path in before_paths]
    after_path = get_single_after()
    print(f"Found {len(before_paths)} BEFORE acquisitions:")
    for index, path in enumerate(before_paths, 1):
        print(f" {index:02d}: {path.name}")
    image_shape = train_pool.shape
    before_stack = np.empty(
        (EXPECTED_BEFORE_COUNT, 2, image_shape[0], image_shape[1]), dtype=np.float32
    )
    before_stats_by_file: dict[str, list[dict[str, float]]] = {}
    print("\nNormalizing all BEFORE acquisitions from train_pool only...")
    for before_index, before_path in enumerate(before_paths):
        print(f"[{before_index + 1:02d}/{EXPECTED_BEFORE_COUNT:02d}] {before_path.name}")
        raw_before = load_img_mat(before_path)
        if raw_before.shape[1:] != image_shape:
            raise ValueError(f"{before_path.name}: shape {raw_before.shape[1:]} != {image_shape}.")
        normalized_before, current_stats = robust_norm_from_train_pool(raw_before, train_pool)
        before_stack[before_index] = normalized_before
        before_stats_by_file[before_path.name] = current_stats
        del raw_before
        del normalized_before
    print("\nLoading AFTER:")
    print(after_path.name)
    raw_after = load_img_mat(after_path)
    if raw_after.shape[1:] != image_shape:
        raise ValueError("AFTER and final split shapes differ.")
    print("Fitting AFTER normalization from train_pool only...")
    after, after_stats = robust_norm_from_train_pool(raw_after, train_pool)
    del raw_after
    print("\nBuilding Mean Teacher patch candidates...")
    labeled_candidates = build_training_candidates(
        train_pool=train_pool,
        target_region=labeled_train,
        minimum_target_pixels=MIN_LABELED_PIXELS_IN_PATCH,
        label="labeled",
    )
    unlabeled_candidates = build_training_candidates(
        train_pool=train_pool,
        target_region=unlabeled_train,
        minimum_target_pixels=MIN_UNLABELED_PIXELS_IN_PATCH,
        label="unlabeled",
    )
    start_epoch = 1
    if RESUME:
        if RESUME_CKPT_PATH is None:
            raise ValueError("RESUME=True but RESUME_CKPT_PATH is None.")
        resume_path = Path(RESUME_CKPT_PATH)
        require_file(resume_path, "Resume checkpoint")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("split_sha256") != split_hash:
            raise RuntimeError("Resume checkpoint split mismatch.")
        if checkpoint.get("gt_sha256") != gt_hash:
            raise RuntimeError("Resume checkpoint GT mismatch.")
        if checkpoint.get("before_names") != before_names:
            raise RuntimeError("Resume checkpoint BEFORE set mismatch.")
        if checkpoint.get("after_name") != after_path.name:
            raise RuntimeError("Resume checkpoint AFTER mismatch.")
        run_directory = Path(checkpoint["run_dir"])
        if not run_directory.exists():
            raise FileNotFoundError(f"Saved run directory not found:\n{run_directory}")
        start_epoch = int(checkpoint["epoch"]) + 1
        print("Resuming from:")
        print(resume_path)
    else:
        timestamp = RUN_NAME
        run_directory = RUN_DIR
        run_directory.mkdir(parents=True, exist_ok=False)
        record_training_inputs(globals(), "semisupervised_multitemporal")
        with (run_directory / "normalization_stats.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "policy": "per-acquisition robust normalization fitted on train_pool pixels only",
                    "before_files": before_names,
                    "after_file": after_path.name,
                    "before": before_stats_by_file,
                    "after": after_stats,
                    "temporal_policy": "uniform random BEFORE acquisition per labeled and unlabeled sample",
                },
                f,
                indent=2,
            )
        with (run_directory / "config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": "final multi-temporal semi-supervised Mean Teacher SAR change detection",
                    "before_files": before_names,
                    "before_count": len(before_names),
                    "after_file": after_path.name,
                    "temporal_training_policy": "uniform random BEFORE acquisition per labeled and unlabeled sample",
                    "teacher_student_temporal_pairing": "same selected BEFORE for teacher and student within each unlabeled sample",
                    "student_perturbation": f"Gaussian noise std={NOISE_STD}",
                    "BASE": BASE,
                    "PATCH": PATCH,
                    "BATCH_L": BATCH_L,
                    "BATCH_U": BATCH_U,
                    "EPOCHS": EPOCHS,
                    "LABELED_SAMPLES_PER_EPOCH": LABELED_SAMPLES_PER_EPOCH,
                    "UNLABELED_SAMPLES_PER_EPOCH": UNLABELED_SAMPLES_PER_EPOCH,
                    "optimizer_steps_per_epoch": LABELED_SAMPLES_PER_EPOCH // BATCH_L,
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "SEED": SEED,
                    "IN_CH": IN_CH,
                    "EMA_DECAY": EMA_DECAY,
                    "LAMBDA_MAX": LAMBDA_MAX,
                    "RAMPUP_EPOCHS": RAMPUP_EPOCHS,
                    "CONF_LOW_START": CONF_LOW_START,
                    "CONF_HIGH_START": CONF_HIGH_START,
                    "CONF_LOW_END": CONF_LOW_END,
                    "CONF_HIGH_END": CONF_HIGH_END,
                    "split_mat": str(SPLIT_MAT),
                    "split_sha256": split_hash,
                    "gt_mat": str(GT_MAT),
                    "gt_sha256": gt_hash,
                    "labeled_patch_candidates": int(labeled_candidates.shape[0]),
                    "unlabeled_patch_candidates": int(unlabeled_candidates.shape[0]),
                    "supervised_loss_region": "labeled_train only",
                    "consistency_loss_region": "unlabeled_train only",
                    "patch_context": "100% inside train_pool",
                    "normalization_fit_region": "train_pool only",
                    "validation_used_in_training": False,
                    "test_used_in_training": False,
                },
                f,
                indent=2,
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL SEMI-SUPERVISED SAR TRAINING")
    print("=" * 80)
    print("Device:", device)
    print("Image shape:", image_shape)
    print(f"Valid GT pixels:    {int(regions['valid_mask'].sum()):,}")
    print(f"Training pool:      {int(train_pool.sum()):,}")
    print(
        f"Labeled training:   {int(labeled_train.sum()):,} ({100.0 * labeled_train.sum() / train_pool.sum():.2f}% of train_pool)"
    )
    print(f"Unlabeled training: {int(unlabeled_train.sum()):,}")
    print(f"Validation pixels:  {int(validation.sum()):,}")
    print(f"Held-out test:      {int(test.sum()):,}")
    print(f"Spatial buffer:     {int(buffer_mask.sum()):,}")
    print("Labeled positive prevalence:", f"{100.0 * labeled_prevalence:.2f}%")
    print("\nValidation/test class labels are not inspected during training.")
    print(f"\nBEFORE acquisitions: {len(before_names)}")
    for index, name in enumerate(before_names, 1):
        print(f" {index:02d}: {name}")
    print("\nTemporal training policy:")
    print(" uniform random BEFORE acquisition per labeled sample")
    print(" uniform random BEFORE acquisition per unlabeled sample")
    print(" teacher/student use the SAME selected BEFORE for each unlabeled sample")
    print("\nAFTER:")
    print(after_path.name)
    print("Run directory:")
    print(run_directory)
    print("\nBEFORE normalization (each independently fitted on train_pool):")
    for name in before_names:
        print(f" {name}")
        for channel, stats in enumerate(before_stats_by_file[name]):
            print(
                f"   channel {channel}: low={stats['low']:.6f}, high={stats['high']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}"
            )
    print("\nAFTER normalization:")
    for channel, stats in enumerate(after_stats):
        print(
            f" channel {channel}: low={stats['low']:.6f}, high={stats['high']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}"
        )
    print("\nMean Teacher schedule:")
    print(f" labeled samples/epoch:   {LABELED_SAMPLES_PER_EPOCH:,}")
    print(f" unlabeled samples/epoch: {UNLABELED_SAMPLES_PER_EPOCH:,}")
    print(f" optimizer steps/epoch:   {LABELED_SAMPLES_PER_EPOCH // BATCH_L:,}")
    print(" consistency region:      unlabeled_train only")
    labeled_dataset = MultiTemporalLabeledPatchDataset(
        before_stack=before_stack,
        after=after,
        training_target=training_target,
        labeled_region=labeled_train,
        candidates=labeled_candidates,
    )
    unlabeled_dataset = MultiTemporalUnlabeledPatchDataset(
        before_stack=before_stack,
        after=after,
        unlabeled_region=unlabeled_train,
        candidates=unlabeled_candidates,
    )
    labeled_loader = make_dataloader(labeled_dataset, BATCH_L)
    unlabeled_loader = make_dataloader(unlabeled_dataset, BATCH_U)
    if len(labeled_loader) != len(unlabeled_loader):
        raise RuntimeError(
            f"Expected equal labeled/unlabeled batch counts. Got {len(labeled_loader)} and {len(unlabeled_loader)}."
        )
    student = SmallUNet(in_ch=IN_CH, base=BASE).to(device)
    teacher = SmallUNet(in_ch=IN_CH, base=BASE).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if RESUME:
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Resuming at epoch:", start_epoch)
    log_file = run_directory / "training_log.csv"
    if not RESUME:
        with log_file.open("w", encoding="utf-8") as f:
            f.write(
                "epoch,total_loss,supervised_loss,consistency_loss,lambda,conf_low,conf_high,conf_mask_ratio\n"
            )
    for epoch in range(start_epoch, EPOCHS + 1):
        student.train()
        teacher.eval()
        consistency_weight = rampup(epoch)
        confidence_low, confidence_high = get_confidence_thresholds(epoch, EPOCHS)
        total_losses = []
        supervised_losses = []
        consistency_losses = []
        confidence_ratios = []
        unlabeled_iterator = iter(unlabeled_loader)
        progress = tqdm(
            labeled_loader,
            desc=f"Epoch {epoch}/{EPOCHS} (lambda={consistency_weight:.2f}, conf={confidence_low:.2f}/{confidence_high:.2f})",
        )
        for labeled_input, labeled_target, labeled_valid in progress:
            try:
                unlabeled_input, unlabeled_valid = next(unlabeled_iterator)
            except StopIteration:
                raise RuntimeError("Unlabeled loader ended before labeled loader.")
            labeled_input = labeled_input.to(device, non_blocking=True)
            labeled_target = labeled_target.to(device, non_blocking=True)
            labeled_valid = labeled_valid.to(device, non_blocking=True)
            unlabeled_input = unlabeled_input.to(device, non_blocking=True)
            unlabeled_valid = unlabeled_valid.to(device, non_blocking=True)
            teacher_input = unlabeled_input
            student_input = add_small_noise(unlabeled_input, std=NOISE_STD)
            optimizer.zero_grad(set_to_none=True)
            labeled_logits = student(labeled_input)
            supervised_loss = masked_bce_with_logits(
                labeled_logits, labeled_target, labeled_valid
            ) + masked_dice_loss_with_logits(labeled_logits, labeled_target, labeled_valid)
            with torch.no_grad():
                teacher_logits = teacher(teacher_input)
                teacher_probabilities = torch.sigmoid(teacher_logits)
            student_logits = student(student_input)
            student_probabilities = torch.sigmoid(student_logits)
            consistency_loss, confidence_ratio = confidence_filtered_mse(
                student_probabilities=student_probabilities,
                teacher_probabilities=teacher_probabilities,
                unlabeled_valid=unlabeled_valid,
                low_threshold=confidence_low,
                high_threshold=confidence_high,
            )
            loss = supervised_loss + consistency_weight * consistency_loss
            loss.backward()
            optimizer.step()
            update_ema(teacher, student, EMA_DECAY)
            total_losses.append(float(loss.detach().cpu()))
            supervised_losses.append(float(supervised_loss.detach().cpu()))
            consistency_losses.append(float(consistency_loss.detach().cpu()))
            confidence_ratios.append(float(confidence_ratio.detach().cpu()))
        mean_total = float(np.mean(total_losses))
        mean_supervised = float(np.mean(supervised_losses))
        mean_consistency = float(np.mean(consistency_losses))
        mean_confidence_ratio = float(np.mean(confidence_ratios))
        print(
            f"Epoch {epoch:03d} | loss={mean_total:.4f} | sup={mean_supervised:.4f} | cons={mean_consistency:.6f} | lambda={consistency_weight:.2f} | conf=({confidence_low:.3f},{confidence_high:.3f}) | mask_ratio={mean_confidence_ratio:.3f}"
        )
        with log_file.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{mean_total:.6f},{mean_supervised:.6f},{mean_consistency:.6f},{consistency_weight:.6f},{confidence_low:.6f},{confidence_high:.6f},{mean_confidence_ratio:.6f}\n"
            )
        save_checkpoint(
            checkpoint_path=run_directory / f"epoch_{epoch:03d}.pt",
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            epoch=epoch,
            run_directory=run_directory,
            before_names=before_names,
            after_name=after_path.name,
            split_hash=split_hash,
            gt_hash=gt_hash,
        )
        save_checkpoint(
            checkpoint_path=run_directory / "mean_teacher_last.pt",
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            epoch=epoch,
            run_directory=run_directory,
            before_names=before_names,
            after_name=after_path.name,
            split_hash=split_hash,
            gt_hash=gt_hash,
        )
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL MEAN TEACHER TRAINING FINISHED")
    print("=" * 80)
    print("Saved to:")
    print(run_directory)
    print("\nNext step:")
    print(
        "EMA TEACHER full-scene inference for each epoch using all BEFORE acquisitions, then equal-weight average of the per-date probability maps per epoch."
    )


if __name__ == "__main__":
    main()
    record_outputs(globals(), "semisupervised_multitemporal", "train")
