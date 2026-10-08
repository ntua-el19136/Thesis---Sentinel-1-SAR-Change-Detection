"""Generate unsmoothed per-epoch maps using the matching checkpoint run.

Usage: python infer_supervised_bitemporal.py --config config.json
See README.md for input formats, outputs and execution order.
"""

from __future__ import annotations
from pipeline_config import parse_arguments, configure
from pipeline_checks import preflight, record_outputs

ARGS = parse_arguments()
from pathlib import Path
import json
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
AUTO_RESUME = True
EPS = 1e-06
globals().update(
    configure(
        ARGS,
        "supervised_bitemporal",
        "infer",
        {k: v for k, v in globals().copy().items() if k.isupper()},
    )
)


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def load_img(path: Path) -> np.ndarray:
    d = loadmat(path)
    if IMAGE_KEY not in d:
        raise KeyError(f"{path.name} missing croppedImg")
    x = np.squeeze(d[IMAGE_KEY])
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"{path.name}: expected (H,W,2), got {x.shape}")
    x = np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{path.name} contains NaN/Inf")
    return x


def normalize_saved(image: np.ndarray, stats: list[dict]) -> np.ndarray:
    out = np.empty_like(image, dtype=np.float32)
    for c in range(2):
        low = float(stats[c]["low"])
        high = float(stats[c]["high"])
        mean = float(stats[c]["mean"])
        std = float(stats[c]["std"])
        if high <= low or std <= EPS:
            raise RuntimeError(f"Invalid saved normalization for channel {c}")
        z = np.clip(image[c], low, high)
        out[c] = (z - mean) / (std + EPS)
    return out


def conv_block(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):

    def __init__(self, in_ch=IN_CH, base=BASE):
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

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.p1(x1))
        x3 = self.e3(self.p2(x2))
        y = self.u2(x3)
        y = self.d2(torch.cat([y, x2], dim=1))
        y = self.u1(y)
        y = self.d1(torch.cat([y, x1], dim=1))
        return self.out(y)


def axis_positions(length: int):
    if length < PATCH:
        raise ValueError("Image dimensions must be at least PATCH.")
    positions = list(range(0, length - PATCH + 1, STRIDE))
    final_position = length - PATCH
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


@torch.no_grad()
def predict_full(model: nn.Module, x: np.ndarray, device: torch.device):
    _, h, w = x.shape
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    tops = axis_positions(h)
    lefts = axis_positions(w)
    for top in tqdm(tops, desc="Rows", leave=False):
        for left in lefts:
            patch = x[:, top : top + PATCH, left : left + PATCH]
            t = torch.from_numpy(patch[None]).to(device, non_blocking=True)
            prob = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
            prob_sum[top : top + PATCH, left : left + PATCH] += prob
            weight_sum[top : top + PATCH, left : left + PATCH] += 1.0
    if np.any(weight_sum <= 0):
        raise RuntimeError("Some pixels were not covered")
    prob_map = prob_sum / weight_sum
    if not np.all(np.isfinite(prob_map)):
        raise RuntimeError("Probability map contains NaN/Inf")
    return prob_map.astype(np.float32)


def main():
    preflight(globals(), "supervised_bitemporal", "infer")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_file(RUN_DIR / f"epoch_{EPOCH_START:03d}.pt", "epoch 001 checkpoint")
    ckpt0 = torch.load(
        RUN_DIR / f"epoch_{EPOCH_START:03d}.pt", map_location="cpu", weights_only=False
    )
    before_name = ckpt0["labeled_before_name"]
    after_name = ckpt0["after_name"]
    if before_name != EXPECTED_BEFORE:
        raise RuntimeError(f"Wrong BEFORE.\nExpected: {EXPECTED_BEFORE}\nFound: {before_name}")
    if after_name != EXPECTED_AFTER:
        raise RuntimeError(f"Wrong AFTER.\nExpected: {EXPECTED_AFTER}\nFound: {after_name}")
    before_path = BEFORE_DIR / before_name
    after_path = AFTER_DIR / after_name
    require_file(before_path, "BEFORE MAT")
    require_file(after_path, "AFTER MAT")
    stats_path = RUN_DIR / "normalization_stats.json"
    require_file(stats_path, "normalization_stats.json")
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    if stats["before_file"] != before_name:
        raise RuntimeError("Saved BEFORE normalization file mismatch")
    if stats["after_file"] != after_name:
        raise RuntimeError("Saved AFTER normalization file mismatch")
    print("Loading BEFORE:")
    print(before_name)
    xb = normalize_saved(load_img(before_path), stats["before"])
    print("Loading AFTER:")
    print(after_name)
    xa = normalize_saved(load_img(after_path), stats["after"])
    if xb.shape[1:] != xa.shape[1:]:
        raise RuntimeError("BEFORE/AFTER shape mismatch")
    x_full = np.concatenate([xb, xa], axis=0).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("BI-TEMPORAL SUPERVISED SAR INFERENCE")
    print("=" * 80)
    print("Device:", device)
    print("Run:", RUN_DIR)
    print("Input shape:", x_full.shape)
    print("Output:", OUTPUT_DIR)
    print("Using exact saved training normalization.")
    print("No GT labels are used.\n")
    missing = [
        RUN_DIR / f"epoch_{epoch:03d}.pt"
        for epoch in range(EPOCH_START, EPOCH_END + 1)
        if not (RUN_DIR / f"epoch_{epoch:03d}.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint: {missing[0]}")
    processed = 0
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        ckpt_path = RUN_DIR / f"epoch_{epoch:03d}.pt"
        out_path = OUTPUT_DIR / f"supervised_change_prob_{epoch:03d}.mat"
        if AUTO_RESUME and out_path.exists():
            print(f"Skipping epoch {epoch:03d}: output already exists.")
            continue
        print("\n" + "=" * 80)
        print(f"PROCESSING EPOCH {epoch:03d}")
        print("=" * 80)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["labeled_before_name"] != before_name or ckpt["after_name"] != after_name:
            raise RuntimeError(f"Epoch {epoch:03d}: image-pair mismatch")
        model = SmallUNet().to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        prob = predict_full(model, x_full, device)
        savemat(
            out_path,
            {
                "prob": prob,
                "epoch": np.array(epoch, dtype=np.int32),
                "before_file": np.array([before_name], dtype=object),
                "after_file": np.array([after_name], dtype=object),
                "patch": np.array(PATCH, dtype=np.int32),
                "stride": np.array(STRIDE, dtype=np.int32),
            },
            do_compression=True,
        )
        print("Saved:", out_path.name)
        print(f"Probability min/mean/max = {prob.min():.6f} / {prob.mean():.6f} / {prob.max():.6f}")
        processed += 1
        del model
        del ckpt
        del prob
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (OUTPUT_DIR / "inference_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "final supervised SAR bi-temporal",
                "checkpoint_run": RUN_DIR.name,
                "before_file": before_name,
                "after_file": after_name,
                "epochs": [EPOCH_START, EPOCH_END],
                "patch": PATCH,
                "stride": STRIDE,
                "normalization": "exact saved train_pool-fitted training normalization",
                "ground_truth_used": False,
                "selection_performed": False,
            },
            f,
            indent=2,
        )
    print("\nDone.")
    print("Newly processed epochs:", processed)
    print("Outputs:", OUTPUT_DIR)
    print("\nNext: validation-only selection of epoch, sigma and threshold, then TEST once.")


if __name__ == "__main__":
    main()
    record_outputs(globals(), "supervised_bitemporal", "infer")
