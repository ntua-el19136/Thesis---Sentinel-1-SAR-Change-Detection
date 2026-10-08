"""Train on the fixed spatial training region; validation and test are excluded from losses.

Usage: python train_semisupervised_bitemporal.py --config config.json
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
        "semisupervised_bitemporal",
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
    array = np.squeeze(data[key])
    if array.ndim != 2:
        raise ValueError(f"{source.name}: '{key}' must be 2-D, got {array.shape}.")
    return array


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
        raise RuntimeError("labeled_train + unlabeled_train do not exactly reconstruct train_pool.")
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
    unique_gt = np.unique(ground_truth[gt_valid])
    if not np.all(np.isin(unique_gt, [0, 1])):
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
    require_file(path, "SAR MAT image")
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
    image: np.ndarray, train_pool: np.ndarray, eps: float = EPS
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if image.shape[1:] != train_pool.shape:
        raise ValueError("Image and train_pool shapes differ.")
    output = np.empty_like(image, dtype=np.float32)
    stats: list[dict[str, float]] = []
    for channel in range(image.shape[0]):
        training_values = image[channel][train_pool]
        if training_values.size == 0:
            raise RuntimeError("train_pool has no pixels.")
        low = float(np.percentile(training_values, 1))
        high = float(np.percentile(training_values, 99))
        if high <= low:
            raise RuntimeError(f"Channel {channel}: invalid clipping range {low}..{high}.")
        clipped_full = np.clip(image[channel], low, high)
        clipped_training = np.clip(training_values, low, high)
        mean = float(clipped_training.mean())
        std = float(clipped_training.std())
        if std <= eps:
            raise RuntimeError(f"Channel {channel}: near-zero std={std}.")
        output[channel] = (clipped_full - mean) / (std + eps)
        stats.append({"low": low, "high": high, "mean": mean, "std": std})
    return (output, stats)


def patch_sum_map(mask: np.ndarray, patch: int) -> np.ndarray:
    if mask.shape[0] < patch or mask.shape[1] < patch:
        raise ValueError("PATCH is larger than the image.")
    integral = np.pad(mask.astype(np.int64, copy=False), ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0, dtype=np.int64), axis=1, dtype=np.int64)
    sums = (
        integral[patch:, patch:]
        - integral[:-patch, patch:]
        - integral[patch:, :-patch]
        + integral[:-patch, :-patch]
    )
    return sums


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


class LabeledPatchDataset(Dataset):

    def __init__(
        self,
        before: np.ndarray,
        after: np.ndarray,
        training_target: np.ndarray,
        labeled_region: np.ndarray,
        candidates: np.ndarray,
        samples: int = LABELED_SAMPLES_PER_EPOCH,
    ):
        super().__init__()
        self.before = before
        self.after = after
        self.training_target = training_target
        self.labeled_region = labeled_region
        self.candidates = candidates
        self.samples = int(samples)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        del index
        candidate_index = random.randrange(self.candidates.shape[0])
        top = int(self.candidates[candidate_index, 0])
        left = int(self.candidates[candidate_index, 1])
        before_patch = self.before[:, top : top + PATCH, left : left + PATCH]
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


class UnlabeledPatchDataset(Dataset):

    def __init__(
        self,
        before: np.ndarray,
        after: np.ndarray,
        unlabeled_region: np.ndarray,
        candidates: np.ndarray,
        samples: int = UNLABELED_SAMPLES_PER_EPOCH,
    ):
        super().__init__()
        self.before = before
        self.after = after
        self.unlabeled_region = unlabeled_region
        self.candidates = candidates
        self.samples = int(samples)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        del index
        candidate_index = random.randrange(self.candidates.shape[0])
        top = int(self.candidates[candidate_index, 0])
        left = int(self.candidates[candidate_index, 1])
        before_patch = self.before[:, top : top + PATCH, left : left + PATCH]
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
    unlabeled_pixels = unlabeled_valid.sum() + EPS
    confidence_ratio = (effective_mask.sum() / unlabeled_pixels).detach()
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
    before_name: str,
    after_name: str,
    gt_hash: str,
    split_hash: str,
    before_hash: str,
    after_hash: str,
) -> None:
    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "run_dir": str(run_directory),
            "labeled_before_name": before_name,
            "before_name": before_name,
            "after_name": after_name,
            "gt_mat": str(GT_MAT),
            "split_mat": str(SPLIT_MAT),
            "gt_sha256": gt_hash,
            "split_sha256": split_hash,
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "config": {
                "experiment": "final_bitemporal_mean_teacher",
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
                "temporal_setup": "single fixed BEFORE + single fixed AFTER",
                "normalization": "per acquisition; fit on train_pool only",
                "validation_used_during_training": False,
                "test_used_during_training": False,
            },
        },
        checkpoint_path,
    )


def main() -> None:
    preflight(globals(), "semisupervised_bitemporal", "train")
    if RUN_DIR.exists() and not RESUME:
        raise FileExistsError(
            f"Training run already exists: {RUN_DIR}; choose a new run_name or resume."
        )
    seed_all(SEED)
    require_file(BEFORE_PATH, "Hardcoded bi-temporal BEFORE")
    require_file(AFTER_PATH, "Hardcoded AFTER")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regions = load_training_regions()
    train_pool = regions["train_pool"]
    labeled_train = regions["labeled_train"]
    unlabeled_train = regions["unlabeled_train"]
    validation = regions["validation"]
    test = regions["test"]
    buffer_mask = regions["buffer_mask"]
    training_target = regions["training_target"]
    labeled_prevalence = float(regions["labeled_prevalence"])
    print("Loading bi-temporal BEFORE:")
    print(BEFORE_PATH.name)
    raw_before = load_img_mat(BEFORE_PATH)
    print("Loading AFTER:")
    print(AFTER_PATH.name)
    raw_after = load_img_mat(AFTER_PATH)
    if raw_before.shape[1:] != train_pool.shape or raw_after.shape[1:] != train_pool.shape:
        raise ValueError("BEFORE / AFTER / final split shapes differ.")
    print("\nFitting BEFORE normalization from train_pool only...")
    before, before_stats = robust_norm_from_train_pool(raw_before, train_pool)
    del raw_before
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
    print("\nComputing reproducibility hashes...")
    gt_hash = sha256_file(GT_MAT)
    split_hash = sha256_file(SPLIT_MAT)
    before_hash = sha256_file(BEFORE_PATH)
    after_hash = sha256_file(AFTER_PATH)
    start_epoch = 1
    if RESUME:
        if RESUME_CKPT_PATH is None:
            raise ValueError("RESUME=True but RESUME_CKPT_PATH is None.")
        resume_path = Path(RESUME_CKPT_PATH)
        require_file(resume_path, "Resume checkpoint")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        run_directory = Path(checkpoint["run_dir"])
        if not run_directory.exists():
            raise FileNotFoundError(f"Saved run directory not found:\n{run_directory}")
        if (
            checkpoint.get("before_name") != BEFORE_PATH.name
            or checkpoint.get("after_name") != AFTER_PATH.name
        ):
            raise RuntimeError("Resume checkpoint acquisition mismatch.")
        if checkpoint.get("gt_sha256") != gt_hash or checkpoint.get("split_sha256") != split_hash:
            raise RuntimeError("Resume checkpoint GT/split hash mismatch.")
        start_epoch = int(checkpoint["epoch"]) + 1
        print("Resuming from:")
        print(resume_path)
    else:
        timestamp = RUN_NAME
        run_directory = RUN_DIR
        run_directory.mkdir(parents=True, exist_ok=False)
        record_training_inputs(globals(), "semisupervised_bitemporal")
    normalization_path = run_directory / "normalization_stats.json"
    if not RESUME:
        with normalization_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "before_file": BEFORE_PATH.name,
                    "after_file": AFTER_PATH.name,
                    "fit_region": "train_pool only",
                    "before": before_stats,
                    "after": after_stats,
                },
                f,
                indent=2,
            )
        config_path = run_directory / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": "final bi-temporal semi-supervised Mean Teacher SAR change detection",
                    "before_file": BEFORE_PATH.name,
                    "after_file": AFTER_PATH.name,
                    "gt_mat": str(GT_MAT),
                    "split_mat": str(SPLIT_MAT),
                    "gt_sha256": gt_hash,
                    "split_sha256": split_hash,
                    "before_sha256": before_hash,
                    "after_sha256": after_hash,
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
                    "NOISE_STD": NOISE_STD,
                    "CONF_LOW_START": CONF_LOW_START,
                    "CONF_HIGH_START": CONF_HIGH_START,
                    "CONF_LOW_END": CONF_LOW_END,
                    "CONF_HIGH_END": CONF_HIGH_END,
                    "labeled_patch_candidates": int(labeled_candidates.shape[0]),
                    "unlabeled_patch_candidates": int(unlabeled_candidates.shape[0]),
                    "normalization": "per acquisition; train_pool only",
                    "supervised_loss_region": "labeled_train only",
                    "consistency_loss_region": "unlabeled_train only",
                    "patch_context": "100% train_pool",
                    "validation_used_in_training": False,
                    "test_used_in_training": False,
                },
                f,
                indent=2,
            )
    print("\n" + "=" * 80)
    print("BI-TEMPORAL SEMI-SUPERVISED SAR TRAINING")
    print("=" * 80)
    print("Device:", device)
    print("Image shape:", train_pool.shape)
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
    print("\nBEFORE:")
    print(BEFORE_PATH.name)
    print("AFTER:")
    print(AFTER_PATH.name)
    print("Run directory:")
    print(run_directory)
    print("\nBEFORE normalization:")
    for channel, stats in enumerate(before_stats):
        print(
            f" channel {channel}: low={stats['low']:.6f}, high={stats['high']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}"
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
    labeled_dataset = LabeledPatchDataset(
        before=before,
        after=after,
        training_target=training_target,
        labeled_region=labeled_train,
        candidates=labeled_candidates,
    )
    unlabeled_dataset = UnlabeledPatchDataset(
        before=before,
        after=after,
        unlabeled_region=unlabeled_train,
        candidates=unlabeled_candidates,
    )
    labeled_loader = make_dataloader(labeled_dataset, BATCH_L)
    unlabeled_loader = make_dataloader(unlabeled_dataset, BATCH_U)
    if len(labeled_loader) != len(unlabeled_loader):
        raise RuntimeError(
            f"Final schedule expects equal numbers of labeled/unlabeled batches per epoch. Got {len(labeled_loader)} vs {len(unlabeled_loader)}."
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
            before_name=BEFORE_PATH.name,
            after_name=AFTER_PATH.name,
            gt_hash=gt_hash,
            split_hash=split_hash,
            before_hash=before_hash,
            after_hash=after_hash,
        )
        save_checkpoint(
            checkpoint_path=run_directory / "mean_teacher_last.pt",
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            epoch=epoch,
            run_directory=run_directory,
            before_name=BEFORE_PATH.name,
            after_name=AFTER_PATH.name,
            gt_hash=gt_hash,
            split_hash=split_hash,
            before_hash=before_hash,
            after_hash=after_hash,
        )
    print("\n" + "=" * 80)
    print("BI-TEMPORAL MEAN TEACHER TRAINING FINISHED")
    print("=" * 80)
    print("Saved to:")
    print(run_directory)
    print("\nNext step:")
    print(
        "Full-scene inference of the EMA TEACHER for configured epochs, then validation-only epoch/sigma/threshold selection."
    )


if __name__ == "__main__":
    main()
    record_outputs(globals(), "semisupervised_bitemporal", "train")
