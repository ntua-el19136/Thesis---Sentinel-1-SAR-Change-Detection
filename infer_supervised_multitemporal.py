"""Generate unsmoothed per-epoch maps using the matching checkpoint run.

Usage: python infer_supervised_multitemporal.py --config config.json
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

PATCH = 256
STRIDE = 128
BASE = 32
IN_CH = 4
EPOCH_START = 1
EPOCH_END = 50
INFERENCE_BATCH = 12
AUTO_RESUME = True
EPS = 1e-06
globals().update(
    configure(
        ARGS,
        "supervised_multitemporal",
        "infer",
        {k: v for k, v in globals().copy().items() if k.isupper()},
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


def configured_run():
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
    return RUN_DIR


def load_img_mat(path: Path) -> np.ndarray:
    data = loadmat(path)
    if IMAGE_KEY not in data:
        keys = [key for key in data.keys() if not key.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Keys: {keys}")
    image = np.squeeze(data[IMAGE_KEY])
    if image.ndim != 3 or image.shape[2] != 2:
        raise ValueError(f"{path.name}: expected (H,W,2), got {image.shape}")
    image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)):
        raise RuntimeError(f"{path.name} contains NaN/Inf.")
    return image


def normalize_with_saved_stats(image: np.ndarray, stats: list[dict]) -> np.ndarray:
    if len(stats) != 2:
        raise ValueError("Expected exactly two SAR-channel normalization records.")
    output = np.empty_like(image, dtype=np.float32)
    for channel in range(2):
        current = stats[channel]
        low = float(current["low"])
        high = float(current["high"])
        mean = float(current["mean"])
        std = float(current["std"])
        if high <= low:
            raise RuntimeError(f"Channel {channel}: invalid clipping range.")
        if std <= EPS:
            raise RuntimeError(f"Channel {channel}: invalid std.")
        clipped = np.clip(image[channel], low, high)
        output[channel] = (clipped - mean) / (std + EPS)
    return output


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


def axis_positions(length: int) -> list[int]:
    if length < PATCH:
        raise ValueError(f"Axis length {length} smaller than PATCH={PATCH}.")
    positions = list(range(0, length - PATCH + 1, STRIDE))
    final_position = length - PATCH
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


@torch.no_grad()
def predict_full_multitemporal(
    model: nn.Module, before_stack: np.ndarray, after: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    temporal_count = before_stack.shape[0]
    if temporal_count != EXPECTED_BEFORE_COUNT:
        raise ValueError(f"Expected {EXPECTED_BEFORE_COUNT} BEFORE images, got {temporal_count}.")
    _, _, height, width = before_stack.shape
    if after.shape != (2, height, width):
        raise ValueError("AFTER shape does not match BEFORE stack.")
    tops = axis_positions(height)
    lefts = axis_positions(width)
    spatial_positions = [(top, left) for top in tops for left in lefts]
    total_samples = len(spatial_positions) * temporal_count
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    batch_inputs = []
    batch_metadata = []
    progress = tqdm(total=total_samples, desc="Temporal/spatial samples", leave=False)

    def flush_batch() -> None:
        nonlocal batch_inputs
        nonlocal batch_metadata
        if not batch_inputs:
            return
        batch_np = np.stack(batch_inputs, axis=0).astype(np.float32, copy=False)
        batch_tensor = torch.from_numpy(batch_np).to(device, non_blocking=True)
        logits = model(batch_tensor)
        probabilities = torch.sigmoid(logits)[:, 0].float().cpu().numpy()
        for probability, (top, left) in zip(probabilities, batch_metadata):
            probability_sum[top : top + PATCH, left : left + PATCH] += probability
            weight_sum[top : top + PATCH, left : left + PATCH] += 1.0
        progress.update(len(batch_inputs))
        batch_inputs = []
        batch_metadata = []

    for top, left in spatial_positions:
        after_patch = after[:, top : top + PATCH, left : left + PATCH]
        for before_index in range(temporal_count):
            before_patch = before_stack[before_index, :, top : top + PATCH, left : left + PATCH]
            input_patch = np.concatenate([before_patch, after_patch], axis=0)
            batch_inputs.append(input_patch)
            batch_metadata.append((top, left))
            if len(batch_inputs) == INFERENCE_BATCH:
                flush_batch()
    flush_batch()
    progress.close()
    if np.any(weight_sum <= 0):
        raise RuntimeError("Some image pixels were not covered during inference.")
    probability_map = probability_sum / weight_sum
    if not np.all(np.isfinite(probability_map)):
        raise RuntimeError("Final probability map contains NaN/Inf.")
    return (probability_map.astype(np.float32), weight_sum.astype(np.float32))


def main() -> None:
    preflight(globals(), "supervised_multitemporal", "infer")
    run_dir = configured_run()
    output_dir = OUTPUT_ROOT / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    require_file(SPLIT_MAT, "Frozen final SAR split")
    split_hash = sha256_file(SPLIT_MAT)
    first_checkpoint_path = run_dir / f"epoch_{EPOCH_START:03d}.pt"
    require_file(first_checkpoint_path, "Epoch 001 checkpoint")
    checkpoint_0 = torch.load(first_checkpoint_path, map_location="cpu", weights_only=False)
    before_names = checkpoint_0.get("before_names")
    after_name = checkpoint_0.get("after_name")
    if not isinstance(before_names, list):
        raise RuntimeError("Checkpoint missing multi-temporal 'before_names'.")
    if len(before_names) != EXPECTED_BEFORE_COUNT:
        raise RuntimeError(
            f"Checkpoint contains {len(before_names)} BEFORE files; expected {EXPECTED_BEFORE_COUNT}."
        )
    if not after_name:
        raise RuntimeError("Checkpoint missing AFTER filename.")
    checkpoint_split_hash = checkpoint_0.get("split_sha256")
    if checkpoint_split_hash is not None and checkpoint_split_hash != split_hash:
        raise RuntimeError("Checkpoint split hash does not match the frozen final split.")
    normalization_path = run_dir / "normalization_stats.json"
    require_file(normalization_path, "Saved training normalization statistics")
    with normalization_path.open("r", encoding="utf-8") as f:
        normalization = json.load(f)
    if normalization.get("before_files") != before_names:
        raise RuntimeError("normalization_stats.json BEFORE list does not match checkpoint.")
    if normalization.get("after_file") != after_name:
        raise RuntimeError("normalization_stats.json AFTER file does not match checkpoint.")
    before_stats_by_file = normalization.get("before")
    after_stats = normalization.get("after")
    if not isinstance(before_stats_by_file, dict):
        raise RuntimeError("Saved BEFORE normalization must be a dictionary keyed by filename.")
    print("\nLoading and normalizing all BEFORE acquisitions...")
    before_stack = None
    image_shape = None
    for before_index, before_name in enumerate(before_names):
        before_path = BEFORE_DIR / before_name
        require_file(before_path, f"BEFORE {before_index + 1:02d}")
        if before_name not in before_stats_by_file:
            raise RuntimeError(f"No saved normalization statistics for:\n{before_name}")
        print(f"[{before_index + 1:02d}/{EXPECTED_BEFORE_COUNT:02d}] {before_name}")
        raw_before = load_img_mat(before_path)
        normalized_before = normalize_with_saved_stats(
            raw_before, before_stats_by_file[before_name]
        )
        if before_stack is None:
            image_shape = normalized_before.shape[1:]
            before_stack = np.empty(
                (EXPECTED_BEFORE_COUNT, 2, image_shape[0], image_shape[1]), dtype=np.float32
            )
        if normalized_before.shape[1:] != image_shape:
            raise RuntimeError(f"{before_name}: spatial shape mismatch.")
        before_stack[before_index] = normalized_before
        del raw_before
        del normalized_before
    after_path = AFTER_DIR / after_name
    require_file(after_path, "AFTER MAT")
    print("\nLoading AFTER:")
    print(after_name)
    raw_after = load_img_mat(after_path)
    after = normalize_with_saved_stats(raw_after, after_stats)
    del raw_after
    if after.shape[1:] != image_shape:
        raise RuntimeError("AFTER spatial shape does not match BEFORE images.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL SUPERVISED SAR INFERENCE")
    print("=" * 80)
    print("Device:", device)
    print("Run directory:")
    print(run_dir)
    print("BEFORE acquisitions:", len(before_names))
    print("AFTER:")
    print(after_name)
    print("Image shape:", image_shape)
    print("Inference batch:", INFERENCE_BATCH)
    print("PATCH / STRIDE:", PATCH, "/", STRIDE)
    print("Temporal aggregation:")
    print(" equal-weight mean across all BEFORE predictions")
    print("Normalization:")
    print(" exact per-acquisition train_pool-fitted statistics saved during training")
    print("GT labels used:", False)
    print("Output directory:")
    print(output_dir)
    missing_checkpoints = []
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        checkpoint_path = run_dir / f"epoch_{epoch:03d}.pt"
        if not checkpoint_path.exists():
            missing_checkpoints.append(checkpoint_path)
    if missing_checkpoints:
        raise FileNotFoundError(
            f"Missing {len(missing_checkpoints)} checkpoints. First missing:\n{missing_checkpoints[0]}"
        )
    processed = 0
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        output_path = output_dir / f"supervised_change_prob_{epoch:03d}.mat"
        if AUTO_RESUME and output_path.exists():
            print(f"Skipping epoch {epoch:03d}: output already exists.")
            continue
        print("\n" + "=" * 80)
        print(f"PROCESSING EPOCH {epoch:03d}")
        print("=" * 80)
        checkpoint_path = run_dir / f"epoch_{epoch:03d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("before_names") != before_names:
            raise RuntimeError(f"Epoch {epoch:03d}: BEFORE list mismatch.")
        if checkpoint.get("after_name") != after_name:
            raise RuntimeError(f"Epoch {epoch:03d}: AFTER mismatch.")
        current_split_hash = checkpoint.get("split_sha256")
        if current_split_hash is not None and current_split_hash != split_hash:
            raise RuntimeError(f"Epoch {epoch:03d}: split hash mismatch.")
        model = SmallUNet(in_ch=IN_CH, base=BASE).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        probability_map, weight_sum = predict_full_multitemporal(model, before_stack, after, device)
        savemat(
            output_path,
            {
                "prob": probability_map.astype(np.float32),
                "weight_sum": weight_sum.astype(np.float32),
                "epoch": np.array(epoch, dtype=np.int32),
                "before_count": np.array(EXPECTED_BEFORE_COUNT, dtype=np.int32),
                "before_files": np.array(before_names, dtype=object),
                "after_file": np.array([after_name], dtype=object),
                "patch": np.array(PATCH, dtype=np.int32),
                "stride": np.array(STRIDE, dtype=np.int32),
                "temporal_aggregation": np.array(["equal_weight_mean"], dtype=object),
            },
            do_compression=True,
        )
        print("Saved:")
        print(output_path)
        print(
            f"Probability min/mean/max = {float(probability_map.min()):.6f} / {float(probability_map.mean()):.6f} / {float(probability_map.max()):.6f}"
        )
        processed += 1
        del model
        del checkpoint
        del probability_map
        del weight_sum
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest_path = output_dir / "inference_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "final supervised SAR multi-temporal",
                "checkpoint_run": run_dir.name,
                "before_files": before_names,
                "before_count": EXPECTED_BEFORE_COUNT,
                "after_file": after_name,
                "epochs": [EPOCH_START, EPOCH_END],
                "patch": PATCH,
                "stride": STRIDE,
                "inference_batch": INFERENCE_BATCH,
                "temporal_aggregation": "equal-weight mean across all BEFORE acquisitions",
                "normalization": "exact saved per-acquisition train_pool-fitted normalization",
                "split_mat": str(SPLIT_MAT),
                "split_sha256": split_hash,
                "ground_truth_used": False,
                "validation_selection_performed": False,
                "test_evaluation_performed": False,
            },
            f,
            indent=2,
        )
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL SUPERVISED INFERENCE FINISHED")
    print("=" * 80)
    print("Newly processed epochs:", processed)
    print("Outputs:")
    print(output_dir)
    print("\nNext step:")
    print(
        "Use the same validation-only evaluator protocol as bi-temporal: configured epochs, configured sigmas and thresholds, validation IoU, then TEST once."
    )


if __name__ == "__main__":
    main()
    record_outputs(globals(), "supervised_multitemporal", "infer")
