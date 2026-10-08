"""Select parameters on validation and report the frozen prediction on test and full AOI.

Usage: python "unsupervised/evaluate_unsupervised.py" --config config.json --stage validate|test
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
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter

MODES = ("global", "pixelwise")
SIGMAS = (0.0, 1.0, 2.0)
THRESHOLDS = np.linspace(0.01, 0.99, 199)
EPS = 1e-12
globals().update(
    configure(
        ARGS, "unsupervised", "evaluate", {k: v for k, v in globals().copy().items() if k.isupper()}
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(data: dict, key: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Required mask missing: {key}")
    arr = np.squeeze(data[key]).astype(bool)
    if arr.ndim != 2 or (shape is not None and arr.shape != shape):
        raise ValueError(f"Invalid shape for {key}: {arr.shape}; expected {shape}")
    return arr


def load_gt_and_split(gt_path: Path, split_path: Path):
    if not gt_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Missing ground truth or split: {gt_path}; {split_path}")
    gt_data = loadmat(gt_path)
    if "ground_truth" not in gt_data:
        raise KeyError("ground_truth absent from ground-truth MAT")
    gt = np.squeeze(gt_data["ground_truth"]).astype(np.uint8)
    if gt.ndim != 2 or not np.all(np.isin(np.unique(gt), (0, 1))):
        raise ValueError("ground_truth must be a 2-D binary 0/1 array")
    gt_valid = load_mask(gt_data, "valid_mask", gt.shape)
    split_data = loadmat(split_path)
    split = {
        key: load_mask(split_data, key, gt.shape)
        for key in ("train_pool", "validation", "test", "buffer_mask", "valid_mask")
    }
    if not np.array_equal(gt_valid, split["valid_mask"]):
        raise ValueError("Ground truth and split valid masks do not match")
    for a, b in (
        ("train_pool", "validation"),
        ("train_pool", "test"),
        ("validation", "test"),
        ("buffer_mask", "train_pool"),
        ("buffer_mask", "validation"),
        ("buffer_mask", "test"),
    ):
        if np.any(split[a] & split[b]):
            raise ValueError(f"Spatial split overlaps: {a} and {b}")
    rebuilt = np.logical_or.reduce(
        [split["train_pool"], split["validation"], split["test"], split["buffer_mask"]]
    )
    if not np.array_equal(rebuilt, gt_valid):
        raise ValueError("Split masks do not reconstruct the valid GT mask")
    val_labels = gt[split["validation"]]
    if val_labels.size == 0 or val_labels.min() == val_labels.max():
        raise ValueError("Validation must contain both classes")
    return (gt, split)


def read_high_map(
    path: Path,
    expected_shape: tuple[int, int],
    expected_mode: str | None = None,
    expected_epoch: int | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing UNSMOOTHED original HIGH probability map: {path}")
    data = loadmat(path)
    if "prob" not in data:
        raise KeyError(f"Missing prob in {path}")
    prob = np.squeeze(data["prob"]).astype(np.float32)
    if prob.shape != expected_shape or not np.isfinite(prob).all():
        raise ValueError(f"Invalid map shape or non-finite values: {path}")
    if prob.min() < -1e-06 or prob.max() > 1 + 1e-06:
        raise ValueError(f"Probability values outside [0,1] in {path}")
    if expected_epoch is not None and "epoch" in data:
        if int(np.squeeze(data["epoch"])) != expected_epoch:
            raise ValueError(f"Epoch metadata mismatch: {path}")
    if expected_mode is not None and "mode" in data:
        raw = data["mode"]
        while isinstance(raw, np.ndarray) and raw.size == 1:
            raw = raw.item()
        if str(raw) != expected_mode:
            raise ValueError(f"Mode metadata mismatch: {path}; {raw!r}")
    return np.clip(prob, 0, 1)


def abs_from_high(prob_high: np.ndarray) -> np.ndarray:
    return np.maximum(prob_high, np.float32(1.0) - prob_high)


def maybe_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return arr
    return gaussian_filter(arr, sigma=sigma, mode="nearest").astype(np.float32)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def metrics_from_arrays(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    y, p = (gt[mask].astype(bool), pred[mask].astype(bool))
    return metrics_from_counts(
        np.count_nonzero(y & p),
        np.count_nonzero(~y & p),
        np.count_nonzero(y & ~p),
        np.count_nonzero(~y & ~p),
    )


def choose_threshold_from_validation(score: np.ndarray, gt: np.ndarray, validation: np.ndarray):
    s = score[validation].astype(np.float64, copy=False)
    y = gt[validation].astype(bool, copy=False)
    pos, neg = (np.sort(s[y]), np.sort(s[~y]))
    pos_below = np.searchsorted(pos, THRESHOLDS, side="left")
    neg_below = np.searchsorted(neg, THRESHOLDS, side="left")
    tp = len(pos) - pos_below
    fp = len(neg) - neg_below
    fn = len(pos) - tp
    tn = len(neg) - fp
    iou = tp / (tp + fp + fn + EPS)
    idx = int(np.argmax(iou))
    return (float(THRESHOLDS[idx]), metrics_from_counts(tp[idx], fp[idx], fn[idx], tn[idx]))


def verify_source_manifest(pred_dir: Path, split_path: Path) -> None:
    manifest_path = pred_dir / "inference_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("smoothing_applied") is not False:
            raise ValueError("Original inference manifest does not confirm unsmoothed maps")
        expected_hash = manifest.get("split_sha256")
        if expected_hash and expected_hash != sha256_file(split_path):
            raise ValueError("Original inference used a DIFFERENT split file")
    else:
        print(
            "WARNING: inference_manifest.json is absent. Confirm source maps are unsmoothed HIGH maps, not final_prediction.mat."
        )


def select_validation(
    pred_dir: Path, out_dir: Path, gt_path: Path, split_path: Path, epochs: range
) -> None:
    gt, split = load_gt_and_split(gt_path, split_path)
    verify_source_manifest(pred_dir, split_path)

    if (out_dir / "frozen_abs_configuration.json").exists():
        raise FileExistsError(
            "Frozen configuration already exists. Use a NEW output directory; never silently overwrite selection."
        )
    missing = [
        pred_dir / f"change_prob_{mode}_{ep:03d}.mat"
        for mode in MODES
        for ep in epochs
        if not (pred_dir / f"change_prob_{mode}_{ep:03d}.mat").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} ORIGINAL per-epoch prediction maps.\nExample: {missing[0]}\nRun inference first; evaluation requires original unsmoothed per-epoch maps."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validation: {split['validation'].sum():,} pixels. Test not accessed for tuning.")
    print("ABS extension: validation-only search over configured epochs, sigmas and thresholds.")
    rows, best = ([], None)
    for mode in MODES:
        for ep in epochs:
            source = pred_dir / f"change_prob_{mode}_{ep:03d}.mat"
            p_high = read_high_map(source, gt.shape, mode, ep)
            p_abs = abs_from_high(p_high)
            del p_high
            for sigma in SIGMAS:
                score = maybe_smooth(p_abs, sigma)
                threshold, metrics = choose_threshold_from_validation(
                    score, gt, split["validation"]
                )
                row = {
                    "mode": mode,
                    "epoch": ep,
                    "sigma": sigma,
                    "threshold": threshold,
                    "validation_iou": metrics["iou"],
                    "validation_f1": metrics["f1"],
                    "validation_precision": metrics["precision"],
                    "validation_recall": metrics["recall"],
                    "validation_specificity": metrics["specificity"],
                    "validation_accuracy": metrics["accuracy"],
                }
                rows.append(row)
                if best is None or row["validation_iou"] > best["validation_iou"]:
                    best = row.copy()
                del score
            print(f"{mode:10s} epoch {ep:03d} processed", flush=True)
            del p_abs
    with (out_dir / "abs_validation_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    selected_path = pred_dir / f"change_prob_{best['mode']}_{best['epoch']:03d}.mat"
    frozen = {
        "study_type": "validation-selected ABS evaluation",
        "source_run": pred_dir.name,
        "source_prediction_dir": str(pred_dir),
        "source_selected_prediction_file": selected_path.name,
        "selected_source_sha256": sha256_file(selected_path),
        "ground_truth_sha256": sha256_file(gt_path),
        "spatial_split_sha256": sha256_file(split_path),
        "score_definition": "sigmoid(abs(z)) = max(p_high, 1-p_high), before smoothing",
        "parameters_chosen_from": "validation only",
        "selection_metric": "IoU",
        "tested_modes": list(MODES),
        "tested_epochs": [epochs.start, epochs.stop - 1],
        "tested_sigma": list(SIGMAS),
        "thresholds": {
            "min": float(min(THRESHOLDS)),
            "max": float(max(THRESHOLDS)),
            "num": len(THRESHOLDS),
        },
        "frozen_selection": best,
        "test_metrics_calculated": False,
    }
    (out_dir / "frozen_abs_configuration.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nVALIDATION-ONLY ABS SELECTION COMPLETE")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Frozen selection: {out_dir / 'frozen_abs_configuration.json'}")
    print(
        "To evaluate the frozen configuration only, run --stage test with the SAME input and output directories."
    )


def evaluate_frozen_test(
    pred_dir: Path, out_dir: Path, gt_path: Path, split_path: Path, save_map: bool
) -> None:
    frozen_path = out_dir / "frozen_abs_configuration.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run --stage validate before --stage test")
    if (out_dir / "abs_final_metrics.json").exists():
        raise FileExistsError(
            "Test metrics already saved. Do not repeatedly evaluate or tune on test."
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["spatial_split_sha256"] != sha256_file(split_path):
        raise ValueError("Split changed since validation selection")
    if frozen["ground_truth_sha256"] != sha256_file(gt_path):
        raise ValueError("GT changed since validation selection")
    if frozen["source_run"] != pred_dir.name:
        raise ValueError("Source prediction run differs from frozen selection")
    verify_source_manifest(pred_dir, split_path)
    conf = frozen["frozen_selection"]
    source = pred_dir / frozen["source_selected_prediction_file"]
    if frozen["selected_source_sha256"] != sha256_file(source):
        raise ValueError("Selected source probability map changed since validation selection")
    gt, split = load_gt_and_split(gt_path, split_path)
    p_high = read_high_map(source, gt.shape, conf["mode"], int(conf["epoch"]))
    score = maybe_smooth(abs_from_high(p_high), float(conf["sigma"]))
    pred = score >= float(conf["threshold"])
    val_m = metrics_from_arrays(gt, pred, split["validation"])
    if abs(val_m["iou"] - conf["validation_iou"]) > 1e-10:
        raise ValueError("Frozen validation IoU did not reproduce; abort test result")
    test_m = metrics_from_arrays(gt, pred, split["test"])
    payload = {
        "scope": "frozen ABS evaluation",
        "frozen_selection": conf,
        "validation": val_m,
        "test": test_m,
        "test_positive_prevalence": float(gt[split["test"]].mean()),
        "test_was_used_for_abs_model_selection": False,
        "test_parameters_frozen_before_evaluation": True,
    }
    (out_dir / "abs_final_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "abs_final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "region",
                "mode",
                "epoch",
                "sigma",
                "threshold",
                "iou",
                "f1",
                "precision",
                "recall",
                "specificity",
                "accuracy",
                "tp",
                "fp",
                "fn",
                "tn",
            ],
        )
        writer.writeheader()
        for region, ms in (("validation", val_m), ("test", test_m)):
            writer.writerow(
                {
                    "region": region,
                    "mode": conf["mode"],
                    "epoch": conf["epoch"],
                    "sigma": conf["sigma"],
                    "threshold": conf["threshold"],
                    **ms,
                }
            )
    if save_map:
        savemat(
            out_dir / "abs_final_prediction.mat",
            {
                "prob": score.astype(np.float32),
                "prediction": pred.astype(np.uint8),
                "epoch": np.int32(conf["epoch"]),
                "sigma": np.float32(conf["sigma"]),
                "threshold": np.float32(conf["threshold"]),
                "mode": np.array([conf["mode"]], dtype=object),
            },
            do_compression=True,
        )
    print("\nFROZEN ABS CONFIGURATION TEST EVALUATION COMPLETE")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir / 'abs_final_metrics.json'}")
    _save_aoi_metrics(
        gt, pred, split["valid_mask"], split["test"], out_dir, "AOI", conf, "strict_held_out_test"
    )


def _full_aoi_valid(gt, *paths):
    from scipy.io import loadmat as _loadmat

    masks = []
    for path in paths:
        data = _loadmat(path, variable_names=["valid_mask"])
        if "valid_mask" in data:
            mask = np.squeeze(data["valid_mask"])
            if mask.shape != gt.shape or not np.all(np.isin(mask, [0, 1])):
                raise ValueError("Invalid full-AOI valid_mask: " + str(path))
            masks.append(mask.astype(bool))
    if not masks:
        raise KeyError("Full AOI requires an authoritative valid_mask; no split-union fallback")
    valid = masks[0].copy()
    for mask in masks[1:]:
        valid &= mask
    if not np.any(valid) or not np.all(np.isin(gt[valid], [0, 1])):
        raise ValueError("Full AOI is empty or contains ignored/nonbinary GT")
    return valid


def _save_aoi_metrics(
    gt, pred, valid, strict, directory, aoi, selection, strict_name="strict_held_out_test"
):
    import json as _json
    import csv as _csv
    from pathlib import Path as _Path

    valid, strict = (np.asarray(valid, bool), np.asarray(strict, bool))
    if not gt.shape == pred.shape == valid.shape == strict.shape:
        raise ValueError("Full AOI reporting shape mismatch")
    if np.any(strict & ~valid):
        raise ValueError("Strict evaluation contains pixels outside full valid AOI")

    def measure(mask):
        if not mask.any():
            raise ValueError("Empty evaluation region")
        y, p = (gt[mask], pred[mask])
        if not np.all(np.isin(y, [0, 1])) or not np.all(np.isin(p, [0, 1])):
            raise ValueError("Evaluation requires finite binary GT and predictions")
        y, p = (y.astype(bool), p.astype(bool))
        tp, fp, fn, tn = (int(np.count_nonzero(v)) for v in (y & p, ~y & p, y & ~p, ~y & ~p))

        def ratio(n, d):
            return float(n / d) if d else 0.0

        n = tp + fp + fn + tn
        return dict(
            pixels=n,
            prevalence=ratio(tp + fn, n),
            predicted_positive_fraction=ratio(tp + fp, n),
            iou=ratio(tp, tp + fp + fn),
            f1=ratio(2 * tp, 2 * tp + fp + fn),
            precision=ratio(tp, tp + fp),
            recall=ratio(tp, tp + fn),
            specificity=ratio(tn, tn + fp),
            accuracy=ratio(tp + tn, n),
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        )

    regions = {strict_name: measure(strict), "full_aoi": measure(valid)}
    payload = dict(
        aoi=aoi,
        selection=selection,
        regions=regions,
        full_aoi_used_for_selection=False,
        full_aoi_scope="All valid AOI pixels, including train/validation/test/buffers; ignored GT excluded. Descriptive evaluation, not held-out generalization.",
    )
    directory = _Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _Path(__file__).stem + "_aoi_metrics"
    (directory / (stem + ".json")).write_text(
        _json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (directory / (stem + ".csv")).open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["region", "aoi"] + list(regions["full_aoi"]))
        writer.writeheader()
        for region, metrics in regions.items():
            writer.writerow(dict(region=region, aoi=aoi, **metrics))
    print("\nADDITIONAL AOI METRICS — same frozen prediction and threshold")
    print(_json.dumps(payload, indent=2, ensure_ascii=False))
    print("Saved:", directory / (stem + ".json"))
    _save_aoi_confusion_maps(gt, pred, valid, strict, directory, aoi, selection, strict_name)


def _save_aoi_confusion_maps(gt, pred, valid, strict, directory, aoi, selection, strict_name):
    from pathlib import Path as _Path
    from PIL import Image as _Image
    from matplotlib.figure import Figure as _Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _Canvas
    from matplotlib.patches import Patch as _Patch
    import json as _json

    directory = _Path(directory)
    stem = _Path(__file__).stem
    palette = {
        1: (255, 255, 255),
        2: (255, 0, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        255: (235, 235, 235),
    }
    labels = {
        1: "TN: correct unchanged",
        2: "FP: false alarm",
        3: "FN: missed change",
        4: "TP: correct change",
        255: "Ignored / outside region",
    }
    geo_profile = None
    reference = globals().get("REFERENCE_TIF")
    if reference is not None and _Path(reference).is_file():
        import rasterio as _rio

        with _rio.open(reference) as src:
            if (src.height, src.width) != gt.shape:
                raise ValueError("Confusion map reference dimensions differ from GT")
            geo_profile = dict(
                driver="GTiff",
                height=gt.shape[0],
                width=gt.shape[1],
                count=1,
                dtype="uint8",
                crs=src.crs,
                transform=src.transform,
                nodata=255,
                compress="deflate",
            )
    for region, mask in ((strict_name, strict), ("full_aoi", valid)):
        mask = np.asarray(mask, bool)
        y, p = (gt == 1, pred == 1)
        classes = np.full(gt.shape, 255, dtype=np.uint8)
        classes[mask & ~y & ~p] = 1
        classes[mask & ~y & p] = 2
        classes[mask & y & ~p] = 3
        classes[mask & y & p] = 4
        rgb = np.empty((*gt.shape, 3), dtype=np.uint8)
        for value, color in palette.items():
            rgb[classes == value] = color
        prefix = directory / (stem + "_" + region + "_confusion")
        _Image.fromarray(rgb).save(str(prefix) + ".png")
        _Image.fromarray(classes).save(str(prefix) + "_classes.png")
        fig = _Figure(figsize=(10, 8))
        _Canvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(aoi + " — " + region)
        handles = [
            _Patch(facecolor=np.asarray(palette[v]) / 255, edgecolor="gray", label=labels[v])
            for v in (4, 1, 2, 3, 255)
        ]
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False
        )
        fig.savefig(str(prefix) + "_legend.png", dpi=300, bbox_inches="tight")
        fig.clear()
        if geo_profile is not None:
            with _rio.open(str(prefix) + ".tif", "w", **geo_profile) as dst:
                dst.write(classes, 1)
                dst.write_colormap(1, {v: (*c, 255) for v, c in palette.items()})
                dst.update_tags(TN="1", FP="2", FN="3", TP="4", IGNORE="255")
        (directory / (stem + "_" + region + "_confusion_legend.json")).write_text(
            _json.dumps(
                dict(
                    region=region,
                    aoi=aoi,
                    selection=selection,
                    classes={
                        str(v): dict(label=labels[v], rgb=list(c)) for v, c in palette.items()
                    },
                    georeferenced_tiff_saved=geo_profile is not None,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Saved confusion map:", str(prefix) + "_legend.png")


def main():
    preflight(globals(), "unsupervised", "evaluate")
    if ARGS.stage == "validate":
        if (RESULTS_DIR / "frozen_abs_configuration.json").exists():
            raise FileExistsError("ABS validation selection already frozen.")
        select_validation(
            PREDICTION_DIR, RESULTS_DIR, GT_MAT, SPLIT_MAT, range(EPOCH_START, EPOCH_END + 1)
        )
    elif ARGS.stage == "test":
        evaluate_frozen_test(PREDICTION_DIR, RESULTS_DIR, GT_MAT, SPLIT_MAT, True)
    else:
        raise ValueError("Specify --stage validate or --stage test.")


if __name__ == "__main__":
    main()
    record_outputs(globals(), "unsupervised", "evaluate")
