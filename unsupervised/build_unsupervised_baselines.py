"""Estimate global and per-pixel pre-event reconstruction-error baselines.

Usage: python "unsupervised/build_unsupervised_baselines.py" --config config.json
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
from pipeline_checks import preflight, record_outputs

ARGS = parse_arguments()
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import autocast

PATCH = 256
STRIDE = 128
IN_CH = 2
BASE = 32
EPOCH_START = 1
EPOCH_END = 50
EPS = 1e-06
globals().update(
    configure(
        ARGS, "unsupervised", "baseline", {k: v for k, v in globals().copy().items() if k.isupper()}
    )
)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")


def require_dir(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint_dir():
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
    return RUN_DIR


def load_train_pool(expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    require_file(SPLIT_MAT, "Frozen final SAR split")
    data = loadmat(SPLIT_MAT)
    if "train_pool" not in data:
        raise KeyError(f"{SPLIT_MAT.name} missing 'train_pool'.")
    train_pool = np.squeeze(data["train_pool"]).astype(bool)
    if train_pool.ndim != 2:
        raise ValueError(f"train_pool must be 2-D; got {train_pool.shape}.")
    if expected_shape is not None and train_pool.shape != expected_shape:
        raise ValueError(f"train_pool shape {train_pool.shape} != image shape {expected_shape}.")
    return train_pool


def load_img_mat(path: Path) -> np.ndarray:
    data = loadmat(path)
    if IMAGE_KEY not in data:
        keys = [key for key in data.keys() if not key.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Available keys: {keys}")
    image = np.squeeze(data[IMAGE_KEY])
    if image.ndim != 3 or image.shape[2] != IN_CH:
        raise ValueError(f"{path.name}: expected (H,W,{IN_CH}), found {image.shape}.")
    image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)):
        raise RuntimeError(f"{path.name} contains NaN/Inf values.")
    return image


def load_training_normalization_stats(checkpoint_dir: Path) -> dict:
    stats_path = checkpoint_dir / "normalization_stats.json"
    require_file(stats_path, "Training normalization statistics")
    with stats_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_using_training_stats(
    image: np.ndarray, image_name: str, all_stats: dict
) -> np.ndarray:
    if image_name not in all_stats:
        raise KeyError(f"No saved training normalization statistics for:\n{image_name}")
    image_stats = all_stats[image_name]
    if len(image_stats) != IN_CH:
        raise ValueError(
            f"{image_name}: expected {IN_CH} channel-stat entries, found {len(image_stats)}."
        )
    output = np.empty_like(image, dtype=np.float32)
    for channel in range(IN_CH):
        params = image_stats[channel]
        if "lo" in params and "hi" in params:
            low = float(params["lo"])
            high = float(params["hi"])
        elif "low" in params and "high" in params:
            low = float(params["low"])
            high = float(params["high"])
        else:
            raise KeyError(
                f"{image_name}, channel {channel}: normalization stats missing lo/hi or low/high."
            )
        mean = float(params["mean"])
        std = float(params["std"])
        if high <= low:
            raise RuntimeError(
                f"{image_name}, channel {channel}: invalid clipping range {low} -> {high}."
            )
        if std <= EPS:
            raise RuntimeError(f"{image_name}, channel {channel}: near-zero std {std}.")
        clipped = np.clip(image[channel], low, high)
        output[channel] = (clipped - mean) / (std + EPS)
    return output


class ConvAE(nn.Module):

    def __init__(self, in_ch: int = IN_CH, base: int = BASE):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(self.p1(x))
        x = self.e3(self.p2(x))
        x = self.d2(self.u2(x))
        x = self.d1(self.u1(x))
        return self.out(x)


@torch.no_grad()
def compute_scalar_error_map(
    model: nn.Module, image: np.ndarray, device: torch.device
) -> np.ndarray:
    channels, height, width = image.shape
    if channels != IN_CH:
        raise ValueError(f"Expected {IN_CH} channels, found {channels}.")
    if min(height, width) < PATCH:
        raise ValueError("Image dimensions must be at least PATCH.")
    error_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    tops = list(range(0, height - PATCH + 1, STRIDE))
    lefts = list(range(0, width - PATCH + 1, STRIDE))
    if tops[-1] != height - PATCH:
        tops.append(height - PATCH)
    if lefts[-1] != width - PATCH:
        lefts.append(width - PATCH)
    for top in tqdm(tops, desc="Sliding-window rows", leave=False):
        for left in lefts:
            patch = image[:, top : top + PATCH, left : left + PATCH]
            tensor = torch.from_numpy(patch[None, ...]).to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                reconstruction = model(tensor)
                scalar_error = (reconstruction - tensor).abs().mean(dim=1)[0]
            error_numpy = scalar_error.float().cpu().numpy()
            error_sum[top : top + PATCH, left : left + PATCH] += error_numpy
            weight_sum[top : top + PATCH, left : left + PATCH] += 1.0
    if np.any(weight_sum <= 0):
        raise RuntimeError("Some pixels were not covered during sliding-window inference.")
    error_map = error_sum / weight_sum
    if not np.all(np.isfinite(error_map)):
        raise RuntimeError("Error map contains NaN/Inf values.")
    return error_map.astype(np.float32)


def main() -> None:
    preflight(globals(), "unsupervised", "baseline")
    require_dir(BEFORE_DIR, "BEFORE directory")
    before_files = sorted(BEFORE_DIR.glob("*.mat"))
    if len(before_files) < 2:
        raise RuntimeError("At least two BEFORE MAT files are required.")
    checkpoint_dir = resolve_checkpoint_dir()
    normalization_stats = load_training_normalization_stats(checkpoint_dir)
    split_hash = sha256_file(SPLIT_MAT)
    first_image = load_img_mat(before_files[0])
    expected_shape = first_image.shape[1:]
    train_pool = load_train_pool(expected_shape)
    for path in before_files[1:]:
        shape = load_img_mat(path).shape[1:]
        if shape != expected_shape:
            raise ValueError(f"{path.name}: shape {shape} != {expected_shape}.")
    output_dir = BASELINE_ROOT / checkpoint_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_before = len(before_files)
    train_pool_pixels = int(train_pool.sum())
    print("\n" + "=" * 80)
    print("SAR BASELINE GENERATION")
    print("=" * 80)
    print("Device:", device)
    print("Checkpoint run:")
    print(checkpoint_dir)
    print("BEFORE images:", n_before)
    print("Image shape:", expected_shape)
    print(f"Training-pool pixels: {train_pool_pixels:,}")
    print("Output:")
    print(output_dir)
    print("\nMethods:")
    print("  GLOBAL    -> BEFORE error mean/std from train_pool only")
    print("  PIXELWISE -> temporal BEFORE error mean/std at each pixel")
    print("\nNo ground-truth labels are used.")
    checkpoint_paths = []
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        require_file(path, f"Epoch {epoch:03d} checkpoint")
        checkpoint_paths.append(path)
    manifest = {
        "checkpoint_dir": str(checkpoint_dir),
        "split_mat": str(SPLIT_MAT),
        "split_sha256": split_hash,
        "before_files": [p.name for p in before_files],
        "number_of_before_images": n_before,
        "patch": PATCH,
        "stride": STRIDE,
        "epochs": [EPOCH_START, EPOCH_END],
        "global_baseline": "scalar reconstruction-error mean/std from frozen train_pool pixels only",
        "pixelwise_baseline": "temporal scalar reconstruction-error mean/sample-std independently at each location",
        "normalization": "exact normalization_stats.json saved during final AE training",
    }
    with (output_dir / "baseline_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    for epoch, checkpoint_path in zip(range(EPOCH_START, EPOCH_END + 1), checkpoint_paths):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch:03d}")
        print("=" * 80)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" not in checkpoint:
            raise KeyError(f"{checkpoint_path.name} missing 'model'.")
        checkpoint_split_hash = checkpoint.get("split_sha256")
        if checkpoint_split_hash is not None and checkpoint_split_hash != split_hash:
            raise RuntimeError(f"{checkpoint_path.name} was trained with a different split.")
        model = ConvAE(in_ch=IN_CH, base=BASE).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        height, width = expected_shape
        pixelwise_mean = np.zeros((height, width), dtype=np.float64)
        pixelwise_m2 = np.zeros((height, width), dtype=np.float64)
        global_count = 0
        global_sum = 0.0
        global_sum_sq = 0.0
        for image_index, path in enumerate(tqdm(before_files, desc=f"Epoch {epoch:03d}"), start=1):
            raw_image = load_img_mat(path)
            image = normalize_using_training_stats(raw_image, path.name, normalization_stats)
            del raw_image
            error_map = compute_scalar_error_map(model, image, device).astype(
                np.float64, copy=False
            )
            delta = error_map - pixelwise_mean
            pixelwise_mean += delta / image_index
            delta_2 = error_map - pixelwise_mean
            pixelwise_m2 += delta * delta_2
            train_errors = error_map[train_pool]
            global_count += int(train_errors.size)
            global_sum += float(train_errors.sum(dtype=np.float64))
            global_sum_sq += float(np.square(train_errors).sum(dtype=np.float64))
            del image
            del error_map
            del train_errors
        global_mean = global_sum / global_count
        global_variance = global_sum_sq / global_count - global_mean**2
        global_variance = max(global_variance, 0.0)
        global_std = float(np.sqrt(global_variance))
        if global_std <= EPS:
            raise RuntimeError(f"Epoch {epoch}: global std too small.")
        pixelwise_variance = pixelwise_m2 / (n_before - 1)
        pixelwise_variance = np.maximum(pixelwise_variance, 0.0)
        pixelwise_std = np.sqrt(pixelwise_variance).astype(np.float32)
        pixelwise_mean_float = pixelwise_mean.astype(np.float32)
        tiny_std_fraction = float(np.mean(pixelwise_std <= 0.0001))
        print("\nGLOBAL:")
        print(f"  mean = {global_mean:.8f}")
        print(f"  std  = {global_std:.8f}")
        print(f"  observations = {global_count:,}")
        print("\nPIXELWISE:")
        print(
            f"  mean map min/max = {float(pixelwise_mean_float.min()):.8f} / {float(pixelwise_mean_float.max()):.8f}"
        )
        print(f"  std map median   = {float(np.median(pixelwise_std)):.8f}")
        print(f"  std map p1       = {float(np.percentile(pixelwise_std, 1)):.8f}")
        print(f"  std <= 1e-4      = {100.0 * tiny_std_fraction:.6f}%")
        global_path = output_dir / f"global_{epoch:03d}.mat"
        savemat(
            global_path,
            {
                "mean": np.array(global_mean, dtype=np.float32),
                "std": np.array(global_std, dtype=np.float32),
                "epoch": np.array(epoch, dtype=np.int32),
                "number_of_before_images": np.array(n_before, dtype=np.int32),
                "train_pool_pixels": np.array(train_pool_pixels, dtype=np.int64),
                "global_observation_count": np.array(global_count, dtype=np.int64),
                "patch": np.array(PATCH, dtype=np.int32),
                "stride": np.array(STRIDE, dtype=np.int32),
            },
            do_compression=True,
        )
        pixelwise_path = output_dir / f"pixelwise_{epoch:03d}.mat"
        savemat(
            pixelwise_path,
            {
                "mean_map": pixelwise_mean_float,
                "std_map": pixelwise_std,
                "epoch": np.array(epoch, dtype=np.int32),
                "number_of_before_images": np.array(n_before, dtype=np.int32),
                "patch": np.array(PATCH, dtype=np.int32),
                "stride": np.array(STRIDE, dtype=np.int32),
            },
            do_compression=True,
        )
        print("\nSaved:")
        print(global_path)
        print(pixelwise_path)
        del model
        del pixelwise_mean
        del pixelwise_m2
        del pixelwise_mean_float
        del pixelwise_std
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\n" + "=" * 80)
    print("SAR BASELINE GENERATION FINISHED")
    print("=" * 80)
    print("Saved in:")
    print(output_dir)
    print("\nNext: final AFTER-image inference for GLOBAL + PIXELWISE only.")


if __name__ == "__main__":
    main()
    record_outputs(globals(), "unsupervised", "baseline")
