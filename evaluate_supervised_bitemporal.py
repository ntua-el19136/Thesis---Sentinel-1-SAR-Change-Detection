"""Select parameters on validation and report the frozen prediction on test and full AOI.

Usage: python evaluate_supervised_bitemporal.py --config config.json
See README.md for input formats, outputs and execution order.
"""

from __future__ import annotations
from pipeline_config import parse_arguments, configure
from pipeline_checks import preflight, record_outputs

ARGS = parse_arguments()
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter

EPOCHS = list(range(1, 51))
SIGMAS = [0.0, 1.0, 2.0]
THRESHOLDS = np.linspace(0.01, 0.99, 199, dtype=np.float64)
EPS = 1e-12
globals().update(
    configure(
        ARGS,
        "supervised_bitemporal",
        "evaluate",
        {k: v for k, v in globals().copy().items() if k.isupper()},
    )
)


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def require_dir(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def load_mask(data, key, source, expected_shape=None):
    if key not in data:
        raise KeyError(f"{source.name} missing '{key}'")
    mask = np.squeeze(data[key]).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"{key} must be 2-D, got {mask.shape}")
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"{key} shape {mask.shape} != {expected_shape}")
    return mask


def load_ground_truth():
    require_file(GT_MAT, "Verified SAR GT")
    data = loadmat(GT_MAT)
    if "ground_truth" not in data:
        raise KeyError(f"{GT_MAT.name} missing ground_truth")
    gt = np.squeeze(data["ground_truth"]).astype(np.uint8)
    if gt.ndim != 2:
        raise ValueError(f"ground_truth must be 2-D, got {gt.shape}")
    if not np.all(np.isin(np.unique(gt), [0, 1])):
        raise ValueError("ground_truth must contain only 0 and 1")
    valid = load_mask(data, "valid_mask", GT_MAT, gt.shape)
    return (gt, valid)


def load_split(shape):
    require_file(SPLIT_MAT, "Frozen final SAR split")
    data = loadmat(SPLIT_MAT)
    names = [
        "train_pool",
        "labeled_train",
        "unlabeled_train",
        "validation",
        "test",
        "buffer_mask",
        "valid_mask",
    ]
    masks = {name: load_mask(data, name, SPLIT_MAT, shape) for name in names}
    if not np.array_equal(masks["labeled_train"] | masks["unlabeled_train"], masks["train_pool"]):
        raise RuntimeError("labeled_train + unlabeled_train do not reconstruct train_pool")
    if np.any(masks["train_pool"] & masks["validation"]):
        raise RuntimeError("train_pool overlaps validation")
    if np.any(masks["train_pool"] & masks["test"]):
        raise RuntimeError("train_pool overlaps test")
    if np.any(masks["validation"] & masks["test"]):
        raise RuntimeError("validation overlaps test")
    rebuilt = masks["train_pool"] | masks["validation"] | masks["test"] | masks["buffer_mask"]
    if not np.array_equal(rebuilt, masks["valid_mask"]):
        raise RuntimeError("train/validation/test/buffer do not reconstruct valid_mask")
    return masks


def load_probability(path, shape):
    require_file(path, "Probability map")
    data = loadmat(path)
    if "prob" not in data:
        raise KeyError(f"{path.name} missing prob")
    prob = np.squeeze(data["prob"]).astype(np.float32)
    if prob.shape != shape:
        raise ValueError(f"{path.name}: {prob.shape} != expected {shape}")
    if not np.all(np.isfinite(prob)):
        raise RuntimeError(f"{path.name} contains NaN/Inf")
    if prob.min() < -1e-06 or prob.max() > 1.0 + 1e-06:
        raise ValueError(f"{path.name}: probability outside [0,1]")
    return np.clip(prob, 0.0, 1.0)


def metrics_from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = (int(tp), int(fp), int(fn), int(tn))
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
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


def evaluate_region(gt, prediction, mask):
    y = gt[mask].astype(bool, copy=False)
    p = prediction[mask].astype(bool, copy=False)
    tp = np.count_nonzero(p & y)
    fp = np.count_nonzero(p & ~y)
    fn = np.count_nonzero(~p & y)
    tn = np.count_nonzero(~p & ~y)
    return metrics_from_counts(tp, fp, fn, tn)


def find_best_threshold(score_map, gt, validation_mask):
    scores = score_map[validation_mask].astype(np.float64, copy=False)
    labels = gt[validation_mask].astype(bool, copy=False)
    positive_scores = np.sort(scores[labels])
    negative_scores = np.sort(scores[~labels])
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise RuntimeError("Validation region must contain both classes")
    pos_below = np.searchsorted(positive_scores, THRESHOLDS, side="left")
    neg_below = np.searchsorted(negative_scores, THRESHOLDS, side="left")
    tp = positive_scores.size - pos_below
    fp = negative_scores.size - neg_below
    fn = positive_scores.size - tp
    tn = negative_scores.size - fp
    iou = tp / (tp + fp + fn + EPS)
    best_i = int(np.argmax(iou))
    return (
        float(THRESHOLDS[best_i]),
        metrics_from_counts(tp[best_i], fp[best_i], fn[best_i], tn[best_i]),
    )


def smooth(probability, sigma):
    if sigma == 0.0:
        return probability
    return np.clip(gaussian_filter(probability, sigma=sigma, mode="nearest"), 0.0, 1.0).astype(
        np.float32
    )


def main():
    preflight(globals(), "supervised_bitemporal", "evaluate")
    require_dir(PREDICTION_DIR, "Bi-temporal supervised prediction directory")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if (RESULTS_DIR / "best_validation_configuration.json").exists():
        raise FileExistsError(
            "Validation selection already frozen. Preserve these results; do not reselect after inspecting test."
        )
    gt, gt_valid = load_ground_truth()
    split = load_split(gt.shape)
    if not np.array_equal(gt_valid, split["valid_mask"]):
        mismatch = int(np.count_nonzero(gt_valid != split["valid_mask"]))
        raise RuntimeError(f"GT valid_mask and split valid_mask differ at {mismatch:,} pixels")
    val = split["validation"]
    test = split["test"]
    val_pixels = int(val.sum())
    test_pixels = int(test.sum())
    val_positive = int(gt[val].sum())
    print("\n" + "=" * 80)
    print("BI-TEMPORAL SUPERVISED SAR EVALUATION")
    print("=" * 80)
    print("Run:", RUN_NAME)
    print("Raster shape:", gt.shape)
    print(f"Validation pixels: {val_pixels:,}")
    print(f"Validation changed: {val_positive:,} ({100.0 * val_positive / val_pixels:.2f}%)")
    print(f"Test pixels:       {test_pixels:,}")
    print("\nVALIDATION SEARCH ONLY")
    print(f"Epochs:     {EPOCHS[0]}..{EPOCHS[-1]}")
    print(f"Sigmas:     {SIGMAS}")
    print(f"Thresholds: {len(THRESHOLDS)} ({THRESHOLDS[0]:.2f}..{THRESHOLDS[-1]:.2f})")
    print("Criterion:  IoU")
    print("TEST labels are not used during search.\n")
    missing = []
    for epoch in EPOCHS:
        path = PREDICTION_DIR / f"supervised_change_prob_{epoch:03d}.mat"
        if not path.exists():
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} prediction files. First missing:\n{missing[0]}"
        )
    rows = []
    best = None
    for epoch in EPOCHS:
        probability = load_probability(
            PREDICTION_DIR / f"supervised_change_prob_{epoch:03d}.mat", gt.shape
        )
        epoch_best = None
        for sigma in SIGMAS:
            score = smooth(probability, sigma)
            threshold, m = find_best_threshold(score, gt, val)
            row = {
                "epoch": int(epoch),
                "sigma": float(sigma),
                "threshold": float(threshold),
                "validation_iou": m["iou"],
                "validation_f1": m["f1"],
                "validation_precision": m["precision"],
                "validation_recall": m["recall"],
                "validation_specificity": m["specificity"],
                "validation_accuracy": m["accuracy"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
            rows.append(row)
            if epoch_best is None or row["validation_iou"] > epoch_best["validation_iou"]:
                epoch_best = row.copy()
            if best is None or row["validation_iou"] > best["validation_iou"]:
                best = row.copy()
        print(
            f"epoch {epoch:03d}: best val IoU={epoch_best['validation_iou']:.4f}, sigma={epoch_best['sigma']:.1f}, thr={epoch_best['threshold']:.4f}"
        )
    search_csv = RESULTS_DIR / "validation_search.csv"
    with search_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n" + "=" * 80)
    print("VALIDATION SELECTION COMPLETE")
    print("=" * 80)
    print("Selected epoch:    ", best["epoch"])
    print("Selected sigma:    ", best["sigma"])
    print("Selected threshold:", f"{best['threshold']:.6f}")
    print("Validation IoU:    ", f"{best['validation_iou']:.6f}")
    print("Validation F1:     ", f"{best['validation_f1']:.6f}")
    config_path = RESULTS_DIR / "best_validation_configuration.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "final bi-temporal supervised SAR change detection",
                "selection_region": "validation only",
                "selection_metric": "IoU",
                "epoch": int(best["epoch"]),
                "sigma": float(best["sigma"]),
                "threshold": float(best["threshold"]),
                "validation_iou": float(best["validation_iou"]),
                "validation_f1": float(best["validation_f1"]),
                "test_used_for_selection": False,
            },
            f,
            indent=2,
        )
    selected_probability = load_probability(
        PREDICTION_DIR / f"supervised_change_prob_{best['epoch']:03d}.mat", gt.shape
    )
    final_probability = smooth(selected_probability, best["sigma"])
    final_prediction = final_probability >= best["threshold"]
    validation_metrics = evaluate_region(gt, final_prediction, val)
    test_metrics = evaluate_region(gt, final_prediction, test)
    test_positive = int(gt[test].sum())
    print("\n" + "=" * 80)
    print("HELD-OUT TEST RESULT")
    print("=" * 80)
    print(f"Test changed prevalence: {100.0 * test_positive / test_pixels:.2f}%")
    for key in ["iou", "f1", "precision", "recall", "specificity", "accuracy"]:
        print(f"{key:12s}: {test_metrics[key]:.6f}")
    print(
        f"TP={test_metrics['tp']:,}  FP={test_metrics['fp']:,}  FN={test_metrics['fn']:,}  TN={test_metrics['tn']:,}"
    )
    with (RESULTS_DIR / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "final bi-temporal supervised SAR change detection",
                "run": RUN_NAME,
                "selection": {
                    "region": "validation only",
                    "criterion": "IoU",
                    "epoch": int(best["epoch"]),
                    "sigma": float(best["sigma"]),
                    "threshold": float(best["threshold"]),
                },
                "validation": validation_metrics,
                "test": test_metrics,
                "test_used_for_selection": False,
            },
            f,
            indent=2,
        )
    metrics_csv = RESULTS_DIR / "final_metrics.csv"
    fields = [
        "region",
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
    ]
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for region, m in [("validation", validation_metrics), ("test", test_metrics)]:
            writer.writerow(
                {
                    "region": region,
                    "epoch": int(best["epoch"]),
                    "sigma": float(best["sigma"]),
                    "threshold": float(best["threshold"]),
                    **m,
                }
            )
    savemat(
        RESULTS_DIR / "final_prediction.mat",
        {
            "prob": final_probability.astype(np.float32),
            "prediction": final_prediction.astype(np.uint8),
            "epoch": np.array(int(best["epoch"]), dtype=np.int32),
            "sigma": np.array(float(best["sigma"]), dtype=np.float32),
            "threshold": np.array(float(best["threshold"]), dtype=np.float32),
        },
        do_compression=True,
    )
    test_pred_vis = np.full(gt.shape, 255, dtype=np.uint8)
    test_pred_vis[test] = final_prediction[test].astype(np.uint8)
    gt_test_vis = np.full(gt.shape, 255, dtype=np.uint8)
    gt_test_vis[test] = gt[test].astype(np.uint8)
    savemat(
        RESULTS_DIR / "final_test_prediction.mat",
        {
            "test_prediction_with_ignore": test_pred_vis,
            "ground_truth_test_with_ignore": gt_test_vis,
            "test_mask": test.astype(np.uint8),
        },
        do_compression=True,
    )
    print("\n" + "=" * 80)
    print("BI-TEMPORAL SUPERVISED EVALUATION FINISHED")
    print("=" * 80)
    print("Results saved in:")
    print(RESULTS_DIR)
    print("\nKey files:")
    print(" validation_search.csv")
    print(" best_validation_configuration.json")
    print(" final_metrics.json")
    print(" final_metrics.csv")
    print(" final_prediction.mat")
    print(" final_test_prediction.mat")
    _save_aoi_metrics(
        gt,
        final_prediction,
        _full_aoi_valid(gt, GT_MAT, SPLIT_MAT),
        test,
        RESULTS_DIR,
        "AOI",
        best,
        "strict_held_out_test",
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


if __name__ == "__main__":
    main()
    record_outputs(globals(), "supervised_bitemporal", "evaluate")
