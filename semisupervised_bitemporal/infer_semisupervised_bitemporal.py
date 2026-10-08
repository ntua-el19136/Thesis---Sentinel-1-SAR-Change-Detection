"""Generate unsmoothed per-epoch maps using the matching checkpoint run.

Usage: python "semisupervised_bitemporal/infer_semisupervised_bitemporal.py" --config config.json
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
        "semisupervised_bitemporal",
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
    d = loadmat(path)
    if IMAGE_KEY not in d:
        keys = [k for k in d if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Keys: {keys}")
    x = np.squeeze(d[IMAGE_KEY])
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
    preflight(globals(), "semisupervised_bitemporal", "infer")
    run_dir = configured_run()
    output_dir = OUTPUT_ROOT / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    first_ckpt_path = run_dir / f"epoch_{EPOCH_START:03d}.pt"
    require_file(first_ckpt_path, "Epoch 001 checkpoint")
    first_ckpt = torch.load(first_ckpt_path, map_location="cpu", weights_only=False)
    before_name = first_ckpt.get("before_name", first_ckpt.get("labeled_before_name"))
    after_name = first_ckpt.get("after_name")
    if before_name != EXPECTED_BEFORE_NAME:
        raise RuntimeError(f"Unexpected BEFORE acquisition:\n{before_name}")
    if after_name != EXPECTED_AFTER_NAME:
        raise RuntimeError(f"Unexpected AFTER acquisition:\n{after_name}")
    before_path = BEFORE_DIR / before_name
    after_path = AFTER_DIR / after_name
    require_file(before_path, "BEFORE MAT")
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
    if norm.get("before_file") != before_name:
        raise RuntimeError("Normalization BEFORE mismatch.")
    if norm.get("after_file") != after_name:
        raise RuntimeError("Normalization AFTER mismatch.")
    print("\nLoading BEFORE:")
    print(before_name)
    before = normalize_with_saved_stats(load_img_mat(before_path), norm["before"])
    print("Loading AFTER:")
    print(after_name)
    after = normalize_with_saved_stats(load_img_mat(after_path), norm["after"])
    if before.shape != after.shape:
        raise RuntimeError("Normalized image shapes differ.")
    missing = []
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        p = run_dir / f"epoch_{epoch:03d}.pt"
        if not p.exists():
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} checkpoints. First missing:\n{missing[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("BI-TEMPORAL MEAN TEACHER SAR INFERENCE")
    print("=" * 80)
    print("Device:", device)
    print("Model for inference: EMA TEACHER")
    print("Run:")
    print(run_dir)
    print("Image shape:", before.shape[1:])
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
        current_before = ckpt.get("before_name", ckpt.get("labeled_before_name"))
        if current_before != before_name:
            raise RuntimeError(f"Epoch {epoch:03d}: BEFORE mismatch.")
        if ckpt.get("after_name") != after_name:
            raise RuntimeError(f"Epoch {epoch:03d}: AFTER mismatch.")
        current_split_hash = ckpt.get("split_sha256")
        if current_split_hash is not None and current_split_hash != split_hash:
            raise RuntimeError(f"Epoch {epoch:03d}: split hash mismatch.")
        if "teacher" not in ckpt:
            raise KeyError(f"Epoch {epoch:03d}: no teacher weights.")
        model = SmallUNet(in_ch=IN_CH, base=BASE).to(device)
        model.load_state_dict(ckpt["teacher"], strict=True)
        model.eval()
        prob, weights = predict_full_scene(model, before, after, device)
        savemat(
            output_path,
            {
                "prob": prob.astype(np.float32),
                "weight_sum": weights.astype(np.float32),
                "epoch": np.array(epoch, dtype=np.int32),
                "model_source": np.array(["ema_teacher"], dtype=object),
                "before_file": np.array([before_name], dtype=object),
                "after_file": np.array([after_name], dtype=object),
                "patch": np.array(PATCH, dtype=np.int32),
                "stride": np.array(STRIDE, dtype=np.int32),
            },
            do_compression=True,
        )
        print("Saved:")
        print(output_path)
        print(
            f"Probability min/mean/max = {float(prob.min()):.6f} / {float(prob.mean()):.6f} / {float(prob.max()):.6f}"
        )
        processed += 1
        del model, ckpt, prob, weights
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest = {
        "experiment": "final bi-temporal semi-supervised Mean Teacher SAR change detection",
        "checkpoint_run": run_dir.name,
        "inference_model": "EMA teacher",
        "before_file": before_name,
        "after_file": after_name,
        "epochs": [EPOCH_START, EPOCH_END],
        "patch": PATCH,
        "stride": STRIDE,
        "inference_batch": INFERENCE_BATCH,
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
    print("BI-TEMPORAL MEAN TEACHER INFERENCE FINISHED")
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
    record_outputs(globals(), "semisupervised_bitemporal", "infer")
