"""Create buffered spatial regions and a labeled subset using fixed class-balance constraints.

Usage: python "ground_truth_and_spatial_split/create_spatial_split.py" --config config.json
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

ARGS = parse_arguments()
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.io import loadmat, savemat

TRAIN_TARGET = 0.6
VAL_TARGET = 0.2
TEST_TARGET = 0.2
TARGETS = np.array([TRAIN_TARGET, VAL_TARGET, TEST_TARGET], dtype=np.float64)
BUFFER_PIXELS = 128
PATCH_SIZE = 256
MIN_REGION_WIDTH = 512
SEARCH_STEP = 8
TRAIN_RATIO_TOL = 0.025
VAL_RATIO_TOL = 0.025
TEST_RATIO_TOL = 0.025
RATIO_TOLS = np.array([TRAIN_RATIO_TOL, VAL_RATIO_TOL, TEST_RATIO_TOL], dtype=np.float64)
MIN_MINORITY_CLASS_FRACTION = 0.2
MAX_POSITIVE_PREVALENCE_DEVIATION = 0.1
RATIO_COST_WEIGHT = 1.0
PREVALENCE_COST_WEIGHT = 1.0
BUFFER_COST_WEIGHT = 0.1
BUFFER_NORMALIZATION = 0.1
LABELED_TARGET = 0.2
LABEL_BLOCK_SIZE = 256
LABEL_HASH_MOD = 10
LABEL_HASH_GROUPS_TO_SELECT = 2
LABEL_HASH_A = 3
LABEL_HASH_B = 7
LABEL_SIZE_TOL = 0.02
LABEL_PREVALENCE_TOL = 0.05
LABEL_SIZE_COST_WEIGHT = 1.0
LABEL_PREVALENCE_COST_WEIGHT = 1.0
NODATA_VALUE = 255
globals().update(
    configure(
        ARGS, "spatial_split", "split", {k: v for k, v in globals().copy().items() if k.isupper()}
    )
)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")


def load_2d(data: dict, key: str, path: Path) -> np.ndarray:
    if key not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{key}'. Available keys: {keys}")
    arr = np.squeeze(data[key])
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: '{key}' must be 2-D, got {arr.shape}")
    return arr


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * float(count) / float(total)


def make_prefix(values: np.ndarray) -> np.ndarray:
    prefix = np.zeros(values.size + 1, dtype=np.int64)
    prefix[1:] = np.cumsum(values.astype(np.int64, copy=False))
    return prefix


def interval_sum(prefix: np.ndarray, start: int, end: int) -> int:
    return int(prefix[end] - prefix[start])


def candidate_passes_hard_constraints(
    ratios: np.ndarray, prevalences: np.ndarray, overall_prevalence: float
) -> tuple[bool, list[str]]:
    reasons = []
    ratio_errors = np.abs(ratios - TARGETS)
    if ratio_errors[0] > TRAIN_RATIO_TOL:
        reasons.append("train ratio")
    if ratio_errors[1] > VAL_RATIO_TOL:
        reasons.append("validation ratio")
    if ratio_errors[2] > TEST_RATIO_TOL:
        reasons.append("test ratio")
    for name, prevalence in zip(("train", "validation", "test"), prevalences):
        minority = min(float(prevalence), 1.0 - float(prevalence))
        if minority < MIN_MINORITY_CLASS_FRACTION:
            reasons.append(f"{name} class balance")
        if abs(float(prevalence) - overall_prevalence) > MAX_POSITIVE_PREVALENCE_DEVIATION:
            reasons.append(f"{name} prevalence")
    return (len(reasons) == 0, reasons)


def internal_selection_cost(
    ratios: np.ndarray, prevalences: np.ndarray, overall_prevalence: float, buffer_fraction: float
) -> tuple[float, float, float, float]:
    normalized_ratio_errors = (ratios - TARGETS) / RATIO_TOLS
    ratio_cost = float(np.sqrt(np.mean(normalized_ratio_errors**2)))
    normalized_prevalence_errors = (
        prevalences - overall_prevalence
    ) / MAX_POSITIVE_PREVALENCE_DEVIATION
    prevalence_cost = float(np.sqrt(np.mean(normalized_prevalence_errors**2)))
    buffer_cost = float(buffer_fraction / BUFFER_NORMALIZATION)
    total_cost = (
        RATIO_COST_WEIGHT * ratio_cost
        + PREVALENCE_COST_WEIGHT * prevalence_cost
        + BUFFER_COST_WEIGHT * buffer_cost
    )
    return (float(total_cost), ratio_cost, prevalence_cost, buffer_cost)


def search_orientation(
    valid_mask: np.ndarray, ground_truth: np.ndarray, axis: str, reverse: bool
) -> dict:
    if axis == "x":
        valid_per_axis = valid_mask.sum(axis=0)
        positive_per_axis = (valid_mask & ground_truth).sum(axis=0)
    elif axis == "y":
        valid_per_axis = valid_mask.sum(axis=1)
        positive_per_axis = (valid_mask & ground_truth).sum(axis=1)
    else:
        raise ValueError(f"Unknown axis '{axis}'.")
    if reverse:
        valid_per_axis = valid_per_axis[::-1]
        positive_per_axis = positive_per_axis[::-1]
    length = int(valid_per_axis.size)
    valid_prefix = make_prefix(valid_per_axis)
    positive_prefix = make_prefix(positive_per_axis)
    total_valid = int(valid_per_axis.sum())
    total_positive = int(positive_per_axis.sum())
    overall_prevalence = total_positive / total_valid
    best_feasible = None
    feasible_count = 0
    evaluated_count = 0
    best_fallback = None
    cut1_min = MIN_REGION_WIDTH
    cut1_max = length - (2 * BUFFER_PIXELS + 2 * MIN_REGION_WIDTH)
    if cut1_max <= cut1_min:
        raise RuntimeError("Raster axis is too short for configured region widths.")
    for cut1 in range(cut1_min, cut1_max + 1, SEARCH_STEP):
        val_start = cut1 + BUFFER_PIXELS
        cut2_min = val_start + MIN_REGION_WIDTH
        cut2_max = length - (BUFFER_PIXELS + MIN_REGION_WIDTH)
        for cut2 in range(cut2_min, cut2_max + 1, SEARCH_STEP):
            evaluated_count += 1
            test_start = cut2 + BUFFER_PIXELS
            train_count = interval_sum(valid_prefix, 0, cut1)
            buffer1_count = interval_sum(valid_prefix, cut1, val_start)
            val_count = interval_sum(valid_prefix, val_start, cut2)
            buffer2_count = interval_sum(valid_prefix, cut2, test_start)
            test_count = interval_sum(valid_prefix, test_start, length)
            usable_count = train_count + val_count + test_count
            if train_count <= 0 or val_count <= 0 or test_count <= 0 or (usable_count <= 0):
                continue
            counts = np.array([train_count, val_count, test_count], dtype=np.float64)
            ratios = counts / float(usable_count)
            train_positive = interval_sum(positive_prefix, 0, cut1)
            val_positive = interval_sum(positive_prefix, val_start, cut2)
            test_positive = interval_sum(positive_prefix, test_start, length)
            prevalences = np.array(
                [
                    train_positive / train_count,
                    val_positive / val_count,
                    test_positive / test_count,
                ],
                dtype=np.float64,
            )
            buffer_count = buffer1_count + buffer2_count
            buffer_fraction = buffer_count / total_valid
            selection_cost, ratio_cost, prevalence_cost, buffer_cost = internal_selection_cost(
                ratios, prevalences, overall_prevalence, buffer_fraction
            )
            passes, reasons = candidate_passes_hard_constraints(
                ratios, prevalences, overall_prevalence
            )
            candidate = {
                "axis": axis,
                "reverse": reverse,
                "cut1": int(cut1),
                "cut2": int(cut2),
                "length": length,
                "train_count": int(train_count),
                "val_count": int(val_count),
                "test_count": int(test_count),
                "buffer_count": int(buffer_count),
                "usable_count": int(usable_count),
                "train_ratio": float(ratios[0]),
                "val_ratio": float(ratios[1]),
                "test_ratio": float(ratios[2]),
                "train_prev": float(prevalences[0]),
                "val_prev": float(prevalences[1]),
                "test_prev": float(prevalences[2]),
                "overall_prev": float(overall_prevalence),
                "selection_cost": float(selection_cost),
                "ratio_cost": float(ratio_cost),
                "prevalence_cost": float(prevalence_cost),
                "buffer_cost": float(buffer_cost),
                "passes": bool(passes),
                "rejection_reasons": reasons,
            }
            if (
                best_fallback is None
                or candidate["selection_cost"] < best_fallback["selection_cost"]
            ):
                best_fallback = candidate
            if passes:
                feasible_count += 1
                if (
                    best_feasible is None
                    or candidate["selection_cost"] < best_feasible["selection_cost"]
                ):
                    best_feasible = candidate
    if best_fallback is None:
        raise RuntimeError(f"No candidate generated for axis={axis}, reverse={reverse}.")
    return {
        "best_feasible": best_feasible,
        "best_fallback": best_fallback,
        "feasible_count": int(feasible_count),
        "evaluated_count": int(evaluated_count),
    }


def make_region_masks(valid_mask: np.ndarray, split: dict):
    axis = split["axis"]
    reverse = split["reverse"]
    cut1 = split["cut1"]
    cut2 = split["cut2"]
    length = split["length"]
    val_start = cut1 + BUFFER_PIXELS
    test_start = cut2 + BUFFER_PIXELS
    axis_code = np.zeros(length, dtype=np.uint8)
    axis_code[:cut1] = 1
    axis_code[cut1:val_start] = 2
    axis_code[val_start:cut2] = 3
    axis_code[cut2:test_start] = 4
    axis_code[test_start:] = 5
    if reverse:
        axis_code = axis_code[::-1]
    if axis == "x":
        spatial_code = np.broadcast_to(axis_code[None, :], valid_mask.shape)
    else:
        spatial_code = np.broadcast_to(axis_code[:, None], valid_mask.shape)
    train_pool = valid_mask & (spatial_code == 1)
    buffer_mask = valid_mask & ((spatial_code == 2) | (spatial_code == 4))
    validation = valid_mask & (spatial_code == 3)
    test = valid_mask & (spatial_code == 5)
    return (train_pool, validation, test, buffer_mask)


def create_labeled_subset(train_pool: np.ndarray, ground_truth: np.ndarray):
    height, width = train_pool.shape
    row_blocks = np.arange(height, dtype=np.int32) // LABEL_BLOCK_SIZE
    col_blocks = np.arange(width, dtype=np.int32) // LABEL_BLOCK_SIZE
    hash_grid = (
        (LABEL_HASH_A * row_blocks[:, None] + LABEL_HASH_B * col_blocks[None, :]) % LABEL_HASH_MOD
    ).astype(np.uint8)
    train_count = int(train_pool.sum())
    train_positive = int((train_pool & ground_truth).sum())
    train_prevalence = train_positive / train_count
    group_counts = []
    group_positive_counts = []
    for group in range(LABEL_HASH_MOD):
        mask = train_pool & (hash_grid == group)
        group_counts.append(int(mask.sum()))
        group_positive_counts.append(int((mask & ground_truth).sum()))
    best = None
    for groups in itertools.combinations(range(LABEL_HASH_MOD), LABEL_HASH_GROUPS_TO_SELECT):
        labeled_count = sum((group_counts[group] for group in groups))
        labeled_positive = sum((group_positive_counts[group] for group in groups))
        if labeled_count <= 0:
            continue
        labeled_fraction = labeled_count / train_count
        labeled_prevalence = labeled_positive / labeled_count
        size_cost = abs(labeled_fraction - LABELED_TARGET) / LABEL_SIZE_TOL
        prevalence_cost = abs(labeled_prevalence - train_prevalence) / LABEL_PREVALENCE_TOL
        total_cost = (
            LABEL_SIZE_COST_WEIGHT * size_cost + LABEL_PREVALENCE_COST_WEIGHT * prevalence_cost
        )
        candidate = {
            "groups": groups,
            "count": int(labeled_count),
            "fraction": float(labeled_fraction),
            "positive_count": int(labeled_positive),
            "prevalence": float(labeled_prevalence),
            "size_cost": float(size_cost),
            "prevalence_cost": float(prevalence_cost),
            "total_cost": float(total_cost),
        }
        if best is None or candidate["total_cost"] < best["total_cost"]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not construct labeled subset.")
    selected = np.isin(hash_grid, np.asarray(best["groups"], dtype=np.uint8))
    labeled_train = train_pool & selected
    unlabeled_train = train_pool & ~selected
    return (labeled_train, unlabeled_train, best, group_counts, group_positive_counts)


def region_stats(name: str, mask: np.ndarray, ground_truth: np.ndarray, denominator: int):
    count = int(mask.sum())
    positive = int((mask & ground_truth).sum())
    negative = count - positive
    prevalence = positive / count if count > 0 else float("nan")
    return {
        "name": name,
        "pixels": count,
        "fraction": count / denominator if denominator > 0 else float("nan"),
        "changed": positive,
        "unchanged": negative,
        "positive_prevalence": float(prevalence),
    }


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            "Spatial-split output exists. Use a new output_root or archive the existing directory first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("CREATE IMPROVED SAR SPATIAL SPLIT")
    print("=" * 80)
    require_file(GT_MAT, "Verified SAR GT MAT")
    require_file(REFERENCE_TIF, "Verified SAR reference TIFF")
    data = loadmat(GT_MAT)
    ground_truth = load_2d(data, "ground_truth", GT_MAT).astype(bool)
    valid_mask = load_2d(data, "valid_mask", GT_MAT).astype(bool)
    if ground_truth.shape != valid_mask.shape:
        raise ValueError("ground_truth and valid_mask shapes differ.")
    height, width = ground_truth.shape
    total_valid = int(valid_mask.sum())
    total_positive = int((valid_mask & ground_truth).sum())
    total_negative = total_valid - total_positive
    overall_prevalence = total_positive / total_valid
    print(f"Shape                 : {height} x {width}")
    print(f"Valid GT pixels       : {total_valid:,}")
    print(f"Changed                : {total_positive:,}")
    print(f"Unchanged              : {total_negative:,}")
    print(f"Positive prevalence       : {100.0 * overall_prevalence:.2f}%")
    print(f"Spatial buffer        : {BUFFER_PIXELS} pixels")
    print("\nHard candidate constraints:")
    print(f"  Train target 60% ± {100 * TRAIN_RATIO_TOL:.1f} pp")
    print(f"  Val target   20% ± {100 * VAL_RATIO_TOL:.1f} pp")
    print(f"  Test target  20% ± {100 * TEST_RATIO_TOL:.1f} pp")
    print(f"  Minimum minority class in each region: {100 * MIN_MINORITY_CLASS_FRACTION:.1f}%")
    print(
        f"  Max positive-prevalence deviation from overall: ±{100 * MAX_POSITIVE_PREVALENCE_DEVIATION:.1f} pp"
    )
    print("\nSearching all four geographic orientations...")
    orientation_results = []
    for axis, reverse in (("x", False), ("x", True), ("y", False), ("y", True)):
        search_result = search_orientation(valid_mask, ground_truth, axis, reverse)
        orientation_results.append({"axis": axis, "reverse": reverse, **search_result})
        direction = "reverse" if reverse else "forward"
        print("\n" + "-" * 72)
        print(f"axis={axis}, direction={direction}")
        print(f"evaluated candidates : {search_result['evaluated_count']:,}")
        print(f"feasible candidates  : {search_result['feasible_count']:,}")
        candidate = search_result["best_feasible"]
        if candidate is None:
            fallback = search_result["best_fallback"]
            print("NO candidate passed all hard constraints.")
            print("Best fallback:")
            print(
                f"  split = {100 * fallback['train_ratio']:.2f}% / {100 * fallback['val_ratio']:.2f}% / {100 * fallback['test_ratio']:.2f}%"
            )
            print(
                f"  positive  = {100 * fallback['train_prev']:.2f}% / {100 * fallback['val_prev']:.2f}% / {100 * fallback['test_prev']:.2f}%"
            )
            print(f"  rejected because: {', '.join(fallback['rejection_reasons'])}")
        else:
            print("Best FEASIBLE candidate:")
            print(f"  internal cost = {candidate['selection_cost']:.6f}")
            print(f"    ratio component      = {candidate['ratio_cost']:.6f}")
            print(f"    prevalence component = {candidate['prevalence_cost']:.6f}")
            print(f"    buffer component     = {candidate['buffer_cost']:.6f}")
            print(
                f"  split among usable = {100 * candidate['train_ratio']:.2f}% / {100 * candidate['val_ratio']:.2f}% / {100 * candidate['test_ratio']:.2f}%"
            )
            print(
                f"  positive prevalence    = {100 * candidate['train_prev']:.2f}% / {100 * candidate['val_prev']:.2f}% / {100 * candidate['test_prev']:.2f}%"
            )
    feasible_candidates = [
        item["best_feasible"] for item in orientation_results if item["best_feasible"] is not None
    ]
    if not feasible_candidates:
        raise RuntimeError(
            "\nNo spatial split passed the configured hard constraints.\nDo NOT weaken constraints blindly. Inspect the printed fallback candidates first."
        )
    best_split = min(feasible_candidates, key=lambda item: item["selection_cost"])
    print("\n" + "=" * 80)
    print("SELECTED GEOGRAPHIC LAYOUT")
    print("=" * 80)
    print(f"Axis                    : {best_split['axis']}")
    print(f"Reverse                 : {best_split['reverse']}")
    print(f"Cut 1                   : {best_split['cut1']}")
    print(f"Cut 2                   : {best_split['cut2']}")
    print(f"Internal selection cost : {best_split['selection_cost']:.6f}")
    print(
        f"Split                    : {100 * best_split['train_ratio']:.2f}% / {100 * best_split['val_ratio']:.2f}% / {100 * best_split['test_ratio']:.2f}%"
    )
    print(
        f"Positive prevalence          : {100 * best_split['train_prev']:.2f}% / {100 * best_split['val_prev']:.2f}% / {100 * best_split['test_prev']:.2f}%"
    )
    train_pool, validation, test, buffer_mask = make_region_masks(valid_mask, best_split)
    usable_mask = train_pool | validation | test
    ignored_original = ~valid_mask
    unused_mask = ~usable_mask
    (
        labeled_train,
        unlabeled_train,
        label_selection,
        label_group_counts,
        label_group_positive_counts,
    ) = create_labeled_subset(train_pool, ground_truth)
    if not np.array_equal(labeled_train | unlabeled_train, train_pool):
        raise RuntimeError("Labeled + unlabeled do not reconstruct train_pool.")
    if np.any(labeled_train & unlabeled_train):
        raise RuntimeError("Labeled and unlabeled overlap.")
    major_masks = {"train_pool": train_pool, "validation": validation, "test": test}
    names = list(major_masks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = int((major_masks[names[i]] & major_masks[names[j]]).sum())
            if overlap:
                raise RuntimeError(f"{names[i]} overlaps {names[j]} by {overlap:,} pixels.")
    if np.any(buffer_mask & usable_mask):
        raise RuntimeError("Spatial buffer overlaps usable regions.")
    reconstructed_valid = usable_mask | buffer_mask
    if not np.array_equal(reconstructed_valid, valid_mask):
        missing = int((valid_mask & ~reconstructed_valid).sum())
        extra = int((reconstructed_valid & ~valid_mask).sum())
        raise RuntimeError(
            f"Usable + buffer do not reconstruct valid mask. Missing={missing:,}, extra={extra:,}"
        )
    usable_count = int(usable_mask.sum())
    buffer_count = int(buffer_mask.sum())
    train_count = int(train_pool.sum())
    validation_count = int(validation.sum())
    test_count = int(test.sum())
    labeled_count = int(labeled_train.sum())
    unlabeled_count = int(unlabeled_train.sum())
    stats = []
    for name, mask in (
        ("Training pool", train_pool),
        ("Labeled training", labeled_train),
        ("Unlabeled training", unlabeled_train),
        ("Validation", validation),
        ("Test", test),
        ("Spatial buffer", buffer_mask),
    ):
        stats.append(region_stats(name, mask, ground_truth, total_valid))
    print("\n" + "=" * 80)
    print("SPLIT STATISTICS")
    print("=" * 80)
    print(f"Original valid pixels : {total_valid:,}")
    print(
        f"Usable pixels         : {usable_count:,} ({pct(usable_count, total_valid):.2f}% of valid)"
    )
    print(
        f"Buffer pixels         : {buffer_count:,} ({pct(buffer_count, total_valid):.2f}% of valid)"
    )
    print("\nMAIN SPLIT — denominator = USABLE pixels")
    print(f"Training pool         : {train_count:,} ({pct(train_count, usable_count):.2f}%)")
    print(
        f"Validation            : {validation_count:,} ({pct(validation_count, usable_count):.2f}%)"
    )
    print(f"Test                  : {test_count:,} ({pct(test_count, usable_count):.2f}%)")
    print("\nTRAINING SUBDIVISION")
    print(
        f"Labeled               : {labeled_count:,} ({pct(labeled_count, train_count):.2f}% of train)"
    )
    print(
        f"Unlabeled             : {unlabeled_count:,} ({pct(unlabeled_count, train_count):.2f}% of train)"
    )
    print(f"Selected block groups : {label_selection['groups']}")
    print(f"Labeled positive prevalence: {100 * label_selection['prevalence']:.2f}%")
    train_prevalence = float(ground_truth[train_pool].mean())
    print(f"Whole train positive prevalence: {100 * train_prevalence:.2f}%")
    print("\nCLASS DISTRIBUTION")
    for item in stats:
        if item["pixels"] > 0:
            prevalence_text = f"{100 * item['positive_prevalence']:.2f}% changed"
        else:
            prevalence_text = "n/a"
        print(
            f"{item['name']:20s} {item['pixels']:>11,} | changed={item['changed']:>11,} | unchanged={item['unchanged']:>11,} | {prevalence_text}"
        )
    final_ratios = np.array(
        [train_count / usable_count, validation_count / usable_count, test_count / usable_count],
        dtype=np.float64,
    )
    final_prevalences = np.array(
        [
            float(ground_truth[train_pool].mean()),
            float(ground_truth[validation].mean()),
            float(ground_truth[test].mean()),
        ],
        dtype=np.float64,
    )
    passes, rejection_reasons = candidate_passes_hard_constraints(
        final_ratios, final_prevalences, overall_prevalence
    )
    if not passes:
        raise RuntimeError(
            "Final generated split unexpectedly fails hard constraints: "
            + ", ".join(rejection_reasons)
        )
    labeled_fraction = labeled_count / train_count
    labeled_prev = float(ground_truth[labeled_train].mean())
    if abs(labeled_fraction - LABELED_TARGET) > 0.03:
        raise RuntimeError(
            "Labeled fraction is more than 3 percentage points away from the 20% target."
        )
    if abs(labeled_prev - train_prevalence) > 0.08:
        raise RuntimeError(
            "Labeled subset positive prevalence differs from whole training pool by more than 8 percentage points."
        )
    print("\nACCEPTANCE CHECK: PASSED.")
    output_mat = OUTPUT_DIR / "spatial_split.mat"
    savemat(
        output_mat,
        {
            "ground_truth": ground_truth.astype(np.uint8),
            "valid_mask": valid_mask.astype(np.uint8),
            "usable_mask": usable_mask.astype(np.uint8),
            "train_pool": train_pool.astype(np.uint8),
            "labeled_train": labeled_train.astype(np.uint8),
            "unlabeled_train": unlabeled_train.astype(np.uint8),
            "validation": validation.astype(np.uint8),
            "test": test.astype(np.uint8),
            "buffer_mask": buffer_mask.astype(np.uint8),
            "ignored_original": ignored_original.astype(np.uint8),
            "unused_mask": unused_mask.astype(np.uint8),
            "target_train_fraction": np.array([[TRAIN_TARGET]], dtype=np.float64),
            "target_validation_fraction": np.array([[VAL_TARGET]], dtype=np.float64),
            "target_test_fraction": np.array([[TEST_TARGET]], dtype=np.float64),
            "target_labeled_fraction_within_train": np.array([[LABELED_TARGET]], dtype=np.float64),
            "buffer_pixels": np.array([[BUFFER_PIXELS]], dtype=np.int32),
            "patch_size": np.array([[PATCH_SIZE]], dtype=np.int32),
            "selected_axis": best_split["axis"],
            "selected_reverse": np.array([[int(best_split["reverse"])]], dtype=np.uint8),
            "selected_cut1": np.array([[best_split["cut1"]]], dtype=np.int32),
            "selected_cut2": np.array([[best_split["cut2"]]], dtype=np.int32),
            "selected_label_hash_groups": np.array(label_selection["groups"], dtype=np.int32),
            "internal_selection_cost": np.array([[best_split["selection_cost"]]], dtype=np.float64),
        },
        do_compression=True,
    )
    split_code = np.full(ground_truth.shape, NODATA_VALUE, dtype=np.uint8)
    split_code[buffer_mask] = 5
    split_code[labeled_train] = 1
    split_code[unlabeled_train] = 2
    split_code[validation] = 3
    split_code[test] = 4
    with rasterio.open(REFERENCE_TIF) as src:
        profile = src.profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": NODATA_VALUE,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
    )
    output_tif = OUTPUT_DIR / "spatial_split.tif"
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(split_code, 1)
    preview = np.zeros(ground_truth.shape, dtype=np.uint8)
    preview[labeled_train] = 1
    preview[unlabeled_train] = 2
    preview[validation] = 3
    preview[test] = 4
    preview[buffer_mask] = 5
    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(preview)
    ax.set_title(
        "Improved Final AOI Spatial Split — Verified GT\nSpatial 60/20/20 target; 20% of training pool labeled"
    )
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3, 4, 5])
    colorbar.ax.set_yticklabels(
        ["Ignored", "Labeled train", "Unlabeled train", "Validation", "Test", "Buffer"]
    )
    plt.tight_layout()
    preview_path = OUTPUT_DIR / "split_preview.png"
    plt.savefig(preview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    balance_names = ["Overall GT", "Train", "Labeled", "Unlabeled", "Validation", "Test"]
    balance_values = [
        100.0 * overall_prevalence,
        100.0 * float(ground_truth[train_pool].mean()),
        100.0 * float(ground_truth[labeled_train].mean()),
        100.0 * float(ground_truth[unlabeled_train].mean()),
        100.0 * float(ground_truth[validation].mean()),
        100.0 * float(ground_truth[test].mean()),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(balance_names, balance_values)
    ax.axhline(100.0 * overall_prevalence, linestyle="--")
    ax.set_ylabel("Changed pixels (%)")
    ax.set_title("Improved AOI Split — Class Distribution")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    balance_path = OUTPUT_DIR / "split_class_balance.png"
    plt.savefig(balance_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    stats_path = OUTPUT_DIR / "split_stats.txt"
    direction_text = "reverse" if best_split["reverse"] else "forward"
    lines = [
        "IMPROVED SAR SPATIAL SPLIT — VERIFIED GT",
        "=" * 62,
        "",
        "Input",
        "-----",
        str(GT_MAT),
        "",
        "Protocol",
        "--------",
        "Target usable split: 60% train / 20% validation / 20% test",
        "Inside training: 20% labeled / 80% unlabeled",
        "Test is excluded from all training and model selection.",
        "",
        "Hard acceptance constraints",
        "---------------------------",
        f"Train tolerance: ±{100 * TRAIN_RATIO_TOL:.2f} percentage points",
        f"Validation tolerance: ±{100 * VAL_RATIO_TOL:.2f} percentage points",
        f"Test tolerance: ±{100 * TEST_RATIO_TOL:.2f} percentage points",
        f"Minimum minority class fraction: {100 * MIN_MINORITY_CLASS_FRACTION:.2f}%",
        f"Maximum positive-prevalence deviation from overall: ±{100 * MAX_POSITIVE_PREVALENCE_DEVIATION:.2f} percentage points",
        "",
        "Selected spatial layout",
        "-----------------------",
        f"Axis: {best_split['axis']}",
        f"Direction: {direction_text}",
        f"Cut 1: {best_split['cut1']}",
        f"Cut 2: {best_split['cut2']}",
        f"Buffer: {BUFFER_PIXELS} pixels",
        "",
        "Internal engineering selection cost",
        "-----------------------------------",
        "This is NOT a model-performance metric.",
        f"Total cost: {best_split['selection_cost']:.8f}",
        f"Ratio component: {best_split['ratio_cost']:.8f}",
        f"Prevalence component: {best_split['prevalence_cost']:.8f}",
        f"Buffer component: {best_split['buffer_cost']:.8f}",
        "",
        "Original verified GT",
        "--------------------",
        f"Valid pixels: {total_valid}",
        f"Changed: {total_positive}",
        f"Unchanged: {total_negative}",
        f"Overall positive prevalence: {100 * overall_prevalence:.8f}%",
        "",
        "Usable vs spatial buffer",
        "------------------------",
        f"Usable pixels: {usable_count}",
        f"Usable / original valid: {pct(usable_count, total_valid):.8f}%",
        f"Buffer pixels: {buffer_count}",
        f"Buffer / original valid: {pct(buffer_count, total_valid):.8f}%",
        "",
        "Main split — denominator = usable pixels",
        "----------------------------------------",
        f"Training: {train_count} ({pct(train_count, usable_count):.8f}%)",
        f"Validation: {validation_count} ({pct(validation_count, usable_count):.8f}%)",
        f"Test: {test_count} ({pct(test_count, usable_count):.8f}%)",
        "",
        "Training subdivision",
        "--------------------",
        f"Labeled: {labeled_count} ({pct(labeled_count, train_count):.8f}% of train)",
        f"Unlabeled: {unlabeled_count} ({pct(unlabeled_count, train_count):.8f}% of train)",
        f"Selected label hash groups: {label_selection['groups']}",
        f"Training positive prevalence: {100 * train_prevalence:.8f}%",
        f"Labeled positive prevalence: {100 * labeled_prev:.8f}%",
        "",
        "Class distribution",
        "------------------",
    ]
    for item in stats:
        lines.extend(
            [
                "",
                item["name"],
                f"Pixels: {item['pixels']}",
                f"Changed: {item['changed']}",
                f"Unchanged: {item['unchanged']}",
                (
                    f"Positive prevalence: {100 * item['positive_prevalence']:.8f}%"
                    if np.isfinite(item["positive_prevalence"])
                    else "Positive prevalence: n/a"
                ),
            ]
        )
    lines.extend(["", "Final acceptance check", "----------------------", "PASSED"])
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print(output_mat)
    print(output_tif)
    print(preview_path)
    print(balance_path)
    print(stats_path)
    print("\nDo NOT start model training until this improved split has been inspected.")
    print("\nDONE.")


if __name__ == "__main__":
    main()
