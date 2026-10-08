"""Generate unsmoothed per-epoch maps using the matching checkpoint run.

Usage: python infer_semisupervised_multitemporal.py --config config.json
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

BASE = 32
IN_CH = 4
PATCH = 256
STRIDE = 128
INFERENCE_BATCH = 12
EPOCH_START = 1
EPOCH_END = 50
AUTO_RESUME = True
EPS = 1e-06
globals().update(
    configure(
        ARGS,
        "semisupervised_multitemporal",
        "infer",
        {k: v for k, v in globals().copy().items() if k.isupper()},
    )
)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


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
    require_file(path, "SAR MAT")
    data = loadmat(path)
    if IMAGE_KEY not in data:
        keys = [k for k in data if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Keys: {keys}")
    x = np.squeeze(data[IMAGE_KEY])
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"{path.name}: expected (H,W,2), got {x.shape}")
    x = np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{path.name} contains NaN/Inf.")
    return x


def normalize_with_saved_stats(image: np.ndarray, stats: list[dict]) -> np.ndarray:
    if len(stats) != 2:
        raise ValueError("Expected two channel-stat records.")
    out = np.empty_like(image, dtype=np.float32)
    for c in range(2):
        lo = float(stats[c]["low"])
        hi = float(stats[c]["high"])
        mean = float(stats[c]["mean"])
        std = float(stats[c]["std"])
        if hi <= lo:
            raise RuntimeError(f"Invalid clipping range, channel {c}.")
        if std <= EPS:
            raise RuntimeError(f"Invalid std, channel {c}.")
        z = np.clip(image[c], lo, hi)
        out[c] = (z - mean) / (std + EPS)
    return out


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
    pos = list(range(0, length - PATCH + 1, STRIDE))
    final = length - PATCH
    if pos[-1] != final:
        pos.append(final)
    return pos


@torch.no_grad()
def predict_full_scene(
    model: nn.Module, before: np.ndarray, after: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    if before.shape != after.shape:
        raise ValueError("BEFORE/AFTER shapes differ.")
    _, h, w = before.shape
    positions = [(top, left) for top in axis_positions(h) for left in axis_positions(w)]
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    batch_x = []
    batch_pos = []
    progress = tqdm(total=len(positions), desc="Sliding-window inference", leave=False)

    def flush() -> None:
        nonlocal batch_x, batch_pos
        if not batch_x:
            return
        x_np = np.stack(batch_x, axis=0).astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        probs = torch.sigmoid(model(x))[:, 0].float().cpu().numpy()
        for p, (top, left) in zip(probs, batch_pos):
            prob_sum[top : top + PATCH, left : left + PATCH] += p
            weight_sum[top : top + PATCH, left : left + PATCH] += 1.0
        progress.update(len(batch_x))
        batch_x = []
        batch_pos = []

    for top, left in positions:
        xb = before[:, top : top + PATCH, left : left + PATCH]
        xa = after[:, top : top + PATCH, left : left + PATCH]
        batch_x.append(np.concatenate([xb, xa], axis=0).astype(np.float32, copy=False))
        batch_pos.append((top, left))
        if len(batch_x) == INFERENCE_BATCH:
            flush()
    flush()
    progress.close()
    if np.any(weight_sum <= 0):
        raise RuntimeError("Some pixels were not covered.")
    prob = prob_sum / weight_sum
    if not np.all(np.isfinite(prob)):
        raise RuntimeError("Probability map contains NaN/Inf.")
    return (prob.astype(np.float32), weight_sum.astype(np.float32))


def main() -> None:
    preflight(globals(), "semisupervised_multitemporal", "infer")
    run_dir = configured_run()
    output_dir = OUTPUT_ROOT / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    first_ckpt_path = run_dir / f"epoch_{EPOCH_START:03d}.pt"
    require_file(first_ckpt_path, "Epoch 001 checkpoint")
    first_ckpt = torch.load(first_ckpt_path, map_location="cpu", weights_only=False)
    before_names = first_ckpt.get("before_names")
    after_name = first_ckpt.get("after_name")
    if before_names is None:
        raise KeyError("Checkpoint missing 'before_names'.")
    if len(before_names) != EXPECTED_BEFORE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_BEFORE_COUNT} BEFORE names, found {len(before_names)}."
        )
    if after_name != EXPECTED_AFTER_NAME:
        raise RuntimeError(
            f"Unexpected AFTER acquisition:\nFound:    {after_name}\nExpected: {EXPECTED_AFTER_NAME}"
        )
    before_paths = [BEFORE_DIR / name for name in before_names]
    after_path = AFTER_DIR / after_name
    for path in before_paths:
        require_file(path, f"BEFORE MAT ({path.name})")
    require_file(after_path, "AFTER MAT")
    require_file(SPLIT_MAT, "Frozen split")
    split_hash = sha256_file(SPLIT_MAT)
    ckpt_split_hash = first_ckpt.get("split_sha256")
    if ckpt_split_hash is not None and ckpt_split_hash != split_hash:
        raise RuntimeError("Frozen split hash mismatch.")
    norm_path = run_dir / "normalization_stats.json"
    require_file(norm_path, "Normalization JSON")
    with norm_path.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    norm_before_names = norm.get("before_files")
    norm_after_name = norm.get("after_file")
    before_stats_dict = norm.get("before")
    after_stats = norm.get("after")
    if norm_before_names != before_names:
        raise RuntimeError("Normalization BEFORE file list mismatch.")
    if norm_after_name != after_name:
        raise RuntimeError("Normalization AFTER mismatch.")
    if not isinstance(before_stats_dict, dict):
        raise RuntimeError("Normalization JSON missing per-BEFORE stats dict.")
    if after_stats is None:
        raise RuntimeError("Normalization JSON missing AFTER stats.")
    print("\nLoading and normalizing all BEFORE acquisitions...")
    normalized_befores = []
    shape_hw = None
    for idx, (name, path) in enumerate(zip(before_names, before_paths), 1):
        print(f"[{idx:02d}/{len(before_names):02d}] {name}")
        if name not in before_stats_dict:
            raise RuntimeError(f"Missing normalization stats for {name}")
        raw_before = load_img_mat(path)
        norm_before = normalize_with_saved_stats(raw_before, before_stats_dict[name])
        del raw_before
        if shape_hw is None:
            shape_hw = norm_before.shape[1:]
        elif norm_before.shape[1:] != shape_hw:
            raise RuntimeError(f"{name}: shape mismatch.")
        normalized_befores.append(norm_before)
    print("\nLoading and normalizing AFTER:")
    print(after_name)
    raw_after = load_img_mat(after_path)
    after = normalize_with_saved_stats(raw_after, after_stats)
    del raw_after
    if shape_hw is None:
        raise RuntimeError("No BEFORE images loaded.")
    if after.shape[1:] != shape_hw:
        raise RuntimeError("AFTER shape mismatch with BEFORE images.")
    missing = []
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        p = run_dir / f"epoch_{epoch:03d}.pt"
        if not p.exists():
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} checkpoints. First missing:\n{missing[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL MEAN TEACHER SAR INFERENCE")
    print("=" * 80)
    print("Device:", device)
    print("Model for inference: EMA TEACHER")
    print("Run:")
    print(run_dir)
    print("Image shape:", shape_hw)
    print("Number of BEFORE acquisitions:", len(before_names))
    print("PATCH / STRIDE:", PATCH, "/", STRIDE)
    print("Inference batch:", INFERENCE_BATCH)
    print("GT labels used: False")
    print("Output:")
    print(output_dir)
    processed = 0
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        output_path = output_dir / f"mean_teacher_change_prob_{epoch:03d}.mat"
        if AUTO_RESUME and output_path.exists():
            print(f"Skipping epoch {epoch:03d}: output already exists.")
            continue
        print("\n" + "=" * 80)
        print(f"PROCESSING EPOCH {epoch:03d}")
        print("=" * 80)
        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        current_before_names = ckpt.get("before_names")
        current_after_name = ckpt.get("after_name")
        current_split_hash = ckpt.get("split_sha256")
        if current_before_names != before_names:
            raise RuntimeError(f"Epoch {epoch:03d}: BEFORE file list mismatch.")
        if current_after_name != after_name:
            raise RuntimeError(f"Epoch {epoch:03d}: AFTER mismatch.")
        if current_split_hash is not None and current_split_hash != split_hash:
            raise RuntimeError(f"Epoch {epoch:03d}: split hash mismatch.")
        if "teacher" not in ckpt:
            raise KeyError(f"Epoch {epoch:03d}: no teacher weights.")
        model = SmallUNet(in_ch=IN_CH, base=BASE).to(device)
        model.load_state_dict(ckpt["teacher"], strict=True)
        model.eval()
        averaged_prob = np.zeros(shape_hw, dtype=np.float32)
        reference_weight_sum = None
        for i, (before_name, before) in enumerate(zip(before_names, normalized_befores), 1):
            print(f"  BEFORE {i:02d}/{len(before_names):02d}: {before_name}")
            prob, weight_sum = predict_full_scene(model, before, after, device)
            averaged_prob += prob
            if reference_weight_sum is None:
                reference_weight_sum = weight_sum
            elif not np.array_equal(reference_weight_sum, weight_sum):
                raise RuntimeError("Sliding-window coverage mismatch across BEFOREs.")
            del prob
            del weight_sum
        averaged_prob /= float(len(before_names))
        if not np.all(np.isfinite(averaged_prob)):
            raise RuntimeError(f"Epoch {epoch:03d}: averaged probability has NaN/Inf.")
        savemat(
            output_path,
            {
                "prob": averaged_prob.astype(np.float32),
                "weight_sum": reference_weight_sum.astype(np.float32),
                "epoch": np.array(epoch, dtype=np.int32),
                "model_source": np.array(["ema_teacher"], dtype=object),
                "temporal_aggregation": np.array(["equal_average_over_befores"], dtype=object),
                "before_files": np.array(before_names, dtype=object),
                "after_file": np.array([after_name], dtype=object),
                "num_before": np.array(len(before_names), dtype=np.int32),
                "patch": np.array(PATCH, dtype=np.int32),
                "stride": np.array(STRIDE, dtype=np.int32),
            },
            do_compression=True,
        )
        print("Saved:")
        print(output_path)
        print(
            f"Averaged probability min/mean/max = {float(averaged_prob.min()):.6f} / {float(averaged_prob.mean()):.6f} / {float(averaged_prob.max()):.6f}"
        )
        processed += 1
        del model
        del ckpt
        del averaged_prob
        del reference_weight_sum
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest = {
        "experiment": "final multi-temporal semi-supervised Mean Teacher SAR change detection",
        "checkpoint_run": run_dir.name,
        "inference_model": "EMA teacher",
        "before_files": before_names,
        "after_file": after_name,
        "epochs": [EPOCH_START, EPOCH_END],
        "patch": PATCH,
        "stride": STRIDE,
        "inference_batch": INFERENCE_BATCH,
        "temporal_aggregation": "equal average over all BEFORE acquisitions",
        "normalization": "exact saved train_pool-fitted statistics",
        "split_mat": str(SPLIT_MAT),
        "split_sha256": split_hash,
        "ground_truth_used": False,
        "validation_selection_performed": False,
        "test_evaluation_performed": False,
    }
    with (output_dir / "inference_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\n" + "=" * 80)
    print("MULTI-TEMPORAL MEAN TEACHER INFERENCE FINISHED")
    print("=" * 80)
    print("Newly processed epochs:", processed)
    print("Outputs:")
    print(output_dir)
    print("\nNext step:")
    print(
        "Validation-only evaluator: configured epochs, configured sigmas and thresholds, select by validation IoU, then TEST once."
    )


if __name__ == "__main__":
    main()
    record_outputs(globals(), "semisupervised_multitemporal", "infer")
