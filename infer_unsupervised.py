"""Generate unsmoothed per-epoch maps using the matching checkpoint run.

Usage: python infer_unsupervised.py --config config.json
See README.md for input formats, outputs and execution order.
"""

from __future__ import annotations
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
ERROR_SMOOTH_SIGMA = 0.0
USE_HIGH_ERROR = True
SAVE_RAW_Z = False
SAVE_ERROR_MAP = False
globals().update(
    configure(
        ARGS, "unsupervised", "infer", {k: v for k, v in globals().copy().items() if k.isupper()}
    )
)


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


def resolve_checkpoint_dir():
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
    return RUN_DIR


def load_split_masks(expected_shape: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    require_file(SPLIT_MAT, "Frozen final SAR split")
    data = loadmat(SPLIT_MAT)
    keys = ["train_pool", "validation", "test", "buffer_mask", "valid_mask"]
    masks = {}
    for key in keys:
        if key not in data:
            raise KeyError(f"{SPLIT_MAT.name} missing '{key}'.")
        mask = np.squeeze(data[key]).astype(bool)
        if mask.ndim != 2:
            raise ValueError(f"{key} must be 2-D, got {mask.shape}.")
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError(f"{key} shape {mask.shape} != image shape {expected_shape}.")
        masks[key] = mask
    train_pool = masks["train_pool"]
    validation = masks["validation"]
    test = masks["test"]
    buffer_mask = masks["buffer_mask"]
    valid_mask = masks["valid_mask"]
    if np.any(train_pool & validation):
        raise RuntimeError("train_pool overlaps validation.")
    if np.any(train_pool & test):
        raise RuntimeError("train_pool overlaps test.")
    if np.any(validation & test):
        raise RuntimeError("validation overlaps test.")
    reconstructed = train_pool | validation | test | buffer_mask
    if not np.array_equal(reconstructed, valid_mask):
        raise RuntimeError("Split masks do not reconstruct valid_mask.")
    return masks


def load_img_mat(path: Path) -> np.ndarray:
    data = loadmat(path)
    if IMAGE_KEY not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Available keys: {keys}")
    image = np.squeeze(data[IMAGE_KEY])
    if image.ndim != 3 or image.shape[2] != IN_CH:
        raise ValueError(f"{path.name}: expected (H,W,{IN_CH}), found {image.shape}.")
    image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)):
        raise RuntimeError(f"{path.name} contains NaN/Inf.")
    return image


def normalize_after_using_train_pool(
    image: np.ndarray, train_pool: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float]]]:
    output = np.empty_like(image, dtype=np.float32)
    stats = []
    for channel in range(IN_CH):
        train_values = image[channel][train_pool]
        low = float(np.percentile(train_values, 1))
        high = float(np.percentile(train_values, 99))
        if high <= low:
            raise RuntimeError(f"Channel {channel}: invalid range {low} -> {high}.")
        clipped_train = np.clip(train_values, low, high)
        mean = float(clipped_train.mean())
        std = float(clipped_train.std())
        if std <= EPS:
            raise RuntimeError(f"Channel {channel}: near-zero std.")
        clipped_full = np.clip(image[channel], low, high)
        output[channel] = (clipped_full - mean) / (std + EPS)
        stats.append({"low": low, "high": high, "mean": mean, "std": std})
    return (output, stats)


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
        raise RuntimeError("Some pixels were not covered.")
    error_map = error_sum / weight_sum
    if not np.all(np.isfinite(error_map)):
        raise RuntimeError("AFTER error map contains NaN/Inf.")
    return error_map.astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def load_scalar(data: dict, key: str, source: Path) -> float:
    if key not in data:
        raise KeyError(f"{source.name} missing '{key}'.")
    return float(np.asarray(data[key]).squeeze())


def load_map(data: dict, key: str, source: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{source.name} missing '{key}'.")
    array = np.squeeze(data[key]).astype(np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{source.name}: {key} shape {array.shape} != {expected_shape}.")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{source.name}: {key} contains NaN/Inf.")
    return array


def save_probability(
    output_path: Path,
    probability: np.ndarray,
    epoch: int,
    mode: str,
    raw_z: np.ndarray | None = None,
    error_map: np.ndarray | None = None,
) -> None:
    payload = {
        "prob": probability.astype(np.float32),
        "epoch": np.array(epoch, dtype=np.int32),
        "mode": np.array([mode], dtype=object),
    }
    if SAVE_RAW_Z and raw_z is not None:
        payload["z"] = raw_z.astype(np.float32)
    if SAVE_ERROR_MAP and error_map is not None:
        payload["err_after"] = error_map.astype(np.float32)
    savemat(output_path, payload, do_compression=True)


def main() -> None:
    preflight(globals(), "unsupervised", "infer")
    if not USE_HIGH_ERROR:
        raise ValueError("ABS evaluation requires unsmoothed HIGH-direction maps.")
    if ERROR_SMOOTH_SIGMA != 0.0:
        raise ValueError("Keep ERROR_SMOOTH_SIGMA=0.0. Smoothing is selected later on validation.")
    require_dir(AFTER_DIR, "AFTER directory")
    require_file(AFTER_PATH, "AFTER acquisition")
    after_path = AFTER_PATH
    checkpoint_dir = resolve_checkpoint_dir()
    baseline_dir = BASELINE_ROOT / checkpoint_dir.name
    require_dir(baseline_dir, "Matching baseline directory")
    split_hash = sha256_file(SPLIT_MAT)
    raw_after = load_img_mat(after_path)
    expected_shape = raw_after.shape[1:]
    split = load_split_masks(expected_shape)
    train_pool = split["train_pool"]
    after_image, after_norm_stats = normalize_after_using_train_pool(raw_after, train_pool)
    del raw_after
    output_dir = OUTPUT_ROOT / checkpoint_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "after_normalization_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "after_file": after_path.name,
                "policy": "same robust per-acquisition normalization as final training; statistics fitted on AFTER train_pool pixels only",
                "channels": after_norm_stats,
            },
            f,
            indent=2,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("UNSUPERVISED SAR AFTER INFERENCE")
    print("=" * 80)
    print("Device:", device)
    print("AFTER:", after_path.name)
    print("Checkpoint run:")
    print(checkpoint_dir)
    print("Baseline directory:")
    print(baseline_dir)
    print("Output directory:")
    print(output_dir)
    print("\nImage shape:", expected_shape)
    print(f"Train-pool pixels used for AFTER normalization: {int(train_pool.sum()):,}")
    print("\nAFTER normalization:")
    for channel, stats in enumerate(after_norm_stats):
        print(
            f"  channel {channel}: low={stats['low']:.6f}, high={stats['high']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}"
        )
    print("\nModes:")
    print("  GLOBAL")
    print("  PIXELWISE")
    print("\nNo GT labels are used.")
    missing = []
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        paths = [
            checkpoint_dir / f"epoch_{epoch:03d}.pt",
            baseline_dir / f"global_{epoch:03d}.mat",
            baseline_dir / f"pixelwise_{epoch:03d}.mat",
        ]
        missing.extend([p for p in paths if not p.exists()])
    if missing:
        print("\nMissing required files:")
        for path in missing:
            print(" ", path)
        raise FileNotFoundError("Cannot start inference.")
    manifest = {
        "after_file": str(after_path),
        "checkpoint_dir": str(checkpoint_dir),
        "baseline_dir": str(baseline_dir),
        "split_mat": str(SPLIT_MAT),
        "split_sha256": split_hash,
        "modes": ["global", "pixelwise"],
        "epochs": [EPOCH_START, EPOCH_END],
        "patch": PATCH,
        "stride": STRIDE,
        "smoothing_applied": False,
        "selection_performed": False,
    }
    with (output_dir / "inference_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        print("\n" + "=" * 80)
        print(f"INFERENCE — EPOCH {epoch:03d}")
        print("=" * 80)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        global_baseline_path = baseline_dir / f"global_{epoch:03d}.mat"
        pixelwise_baseline_path = baseline_dir / f"pixelwise_{epoch:03d}.mat"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_split_hash = checkpoint.get("split_sha256")
        if checkpoint_split_hash is not None and checkpoint_split_hash != split_hash:
            raise RuntimeError(f"Epoch {epoch:03d} checkpoint was trained with a different split.")
        model = ConvAE(in_ch=IN_CH, base=BASE).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        error_after = compute_scalar_error_map(model, after_image, device)
        global_data = loadmat(global_baseline_path)
        global_mean = load_scalar(global_data, "mean", global_baseline_path)
        global_std = load_scalar(global_data, "std", global_baseline_path)
        if global_std <= EPS:
            raise RuntimeError(f"Epoch {epoch}: global std too small.")
        global_z = (error_after - global_mean) / (global_std + EPS)
        if not USE_HIGH_ERROR:
            global_z = -global_z
        global_probability = sigmoid(global_z)
        global_output = output_dir / f"change_prob_global_{epoch:03d}.mat"
        save_probability(
            global_output,
            global_probability,
            epoch,
            "global",
            raw_z=global_z,
            error_map=error_after,
        )
        pixelwise_data = loadmat(pixelwise_baseline_path)
        pixelwise_mean = load_map(
            pixelwise_data, "mean_map", pixelwise_baseline_path, expected_shape
        )
        pixelwise_std = load_map(pixelwise_data, "std_map", pixelwise_baseline_path, expected_shape)
        if np.any(pixelwise_std < 0):
            raise RuntimeError(f"Epoch {epoch}: pixelwise std contains negative values.")
        pixelwise_z = (error_after - pixelwise_mean) / (pixelwise_std + EPS)
        if not USE_HIGH_ERROR:
            pixelwise_z = -pixelwise_z
        pixelwise_probability = sigmoid(pixelwise_z)
        pixelwise_output = output_dir / f"change_prob_pixelwise_{epoch:03d}.mat"
        save_probability(
            pixelwise_output,
            pixelwise_probability,
            epoch,
            "pixelwise",
            raw_z=pixelwise_z,
            error_map=error_after,
        )
        print("GLOBAL:")
        print(f"  baseline mean/std = {global_mean:.8f} / {global_std:.8f}")
        print(
            f"  prob min/mean/max = {float(global_probability.min()):.6f} / {float(global_probability.mean()):.6f} / {float(global_probability.max()):.6f}"
        )
        print("PIXELWISE:")
        print(
            f"  prob min/mean/max = {float(pixelwise_probability.min()):.6f} / {float(pixelwise_probability.mean()):.6f} / {float(pixelwise_probability.max()):.6f}"
        )
        print("Saved:")
        print(" ", global_output)
        print(" ", pixelwise_output)
        del model
        del error_after
        del global_z
        del global_probability
        del pixelwise_mean
        del pixelwise_std
        del pixelwise_z
        del pixelwise_probability
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\n" + "=" * 80)
    print("SAR AFTER INFERENCE FINISHED")
    print("=" * 80)
    print("Outputs saved in:")
    print(output_dir)
    print("\nNext step:")
    print(
        "Validation-only selection of GLOBAL vs PIXELWISE, epoch, sigma and threshold; then one final TEST evaluation."
    )


if __name__ == "__main__":
    main()
    record_outputs(globals(), "unsupervised", "infer")
