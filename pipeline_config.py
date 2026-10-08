"""Shared paths and run settings for the command-line pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METHODS = (
    "unsupervised",
    "supervised_bitemporal",
    "supervised_multitemporal",
    "semisupervised_bitemporal",
    "semisupervised_multitemporal",
)


PARAMETER_KEYS = {
    "semisupervised_bitemporal": [
        "BASE",
        "BATCH_L",
        "BATCH_U",
        "CONF_HIGH_END",
        "CONF_HIGH_START",
        "CONF_LOW_END",
        "CONF_LOW_START",
        "EMA_DECAY",
        "INFERENCE_BATCH",
        "LABELED_SAMPLES_PER_EPOCH",
        "LAMBDA_MAX",
        "LR",
        "MIN_LABELED_PIXELS_IN_PATCH",
        "MIN_UNLABELED_PIXELS_IN_PATCH",
        "NOISE_STD",
        "NUM_WORKERS",
        "PATCH",
        "PERSISTENT_WORKERS",
        "PREFETCH_FACTOR",
        "RAMPUP_EPOCHS",
        "SEED",
        "SIGMAS",
        "STRIDE",
        "THRESHOLDS",
        "UNLABELED_SAMPLES_PER_EPOCH",
        "WEIGHT_DECAY",
    ],
    "supervised_bitemporal": [
        "BASE",
        "BATCH",
        "LABELED_SAMPLES_PER_EPOCH",
        "LR",
        "NUM_WORKERS",
        "PATCH",
        "PERSISTENT_WORKERS",
        "PIN_MEMORY",
        "PREFETCH_FACTOR",
        "SEED",
        "SIGMAS",
        "STRIDE",
        "THRESHOLDS",
        "WEIGHT_DECAY",
    ],
    "unsupervised": [
        "BASE",
        "BATCH",
        "CANDIDATE_STRIDE",
        "LR",
        "NUM_WORKERS",
        "PATCH",
        "PERSISTENT_WORKERS",
        "PREFETCH_FACTOR",
        "SAMPLES_PER_EPOCH",
        "SAVE_ERROR_MAP",
        "SAVE_RAW_Z",
        "SEED",
        "SIGMAS",
        "STRIDE",
        "THRESHOLDS",
    ],
    "supervised_multitemporal": [
        "BASE",
        "BATCH",
        "INFERENCE_BATCH",
        "LABELED_SAMPLES_PER_EPOCH",
        "LR",
        "NUM_WORKERS",
        "PATCH",
        "PERSISTENT_WORKERS",
        "PIN_MEMORY",
        "PREFETCH_FACTOR",
        "SEED",
        "SIGMAS",
        "STRIDE",
        "THRESHOLDS",
        "WEIGHT_DECAY",
    ],
    "ground_truth": ["ALL_TOUCHED", "MIN_CHANNEL_CORRELATION", "MIN_FULL_CORRELATION"],
    "semisupervised_multitemporal": [
        "BASE",
        "BATCH_L",
        "BATCH_U",
        "CONF_HIGH_END",
        "CONF_HIGH_START",
        "CONF_LOW_END",
        "CONF_LOW_START",
        "EMA_DECAY",
        "INFERENCE_BATCH",
        "LABELED_SAMPLES_PER_EPOCH",
        "LAMBDA_MAX",
        "LR",
        "MIN_LABELED_PIXELS_IN_PATCH",
        "MIN_UNLABELED_PIXELS_IN_PATCH",
        "NOISE_STD",
        "NUM_WORKERS",
        "PATCH",
        "PERSISTENT_WORKERS",
        "PREFETCH_FACTOR",
        "RAMPUP_EPOCHS",
        "SEED",
        "SIGMAS",
        "STRIDE",
        "THRESHOLDS",
        "UNLABELED_SAMPLES_PER_EPOCH",
        "WEIGHT_DECAY",
    ],
    "spatial_split": [
        "BUFFER_COST_WEIGHT",
        "BUFFER_NORMALIZATION",
        "BUFFER_PIXELS",
        "LABELED_TARGET",
        "LABEL_BLOCK_SIZE",
        "LABEL_HASH_A",
        "LABEL_HASH_B",
        "LABEL_HASH_GROUPS_TO_SELECT",
        "LABEL_HASH_MOD",
        "LABEL_PREVALENCE_COST_WEIGHT",
        "LABEL_PREVALENCE_TOL",
        "LABEL_SIZE_COST_WEIGHT",
        "LABEL_SIZE_TOL",
        "MAX_POSITIVE_PREVALENCE_DEVIATION",
        "MIN_MINORITY_CLASS_FRACTION",
        "MIN_REGION_WIDTH",
        "PATCH_SIZE",
        "PREVALENCE_COST_WEIGHT",
        "RATIO_COST_WEIGHT",
        "SEARCH_STEP",
        "TEST_RATIO_TOL",
        "TEST_TARGET",
        "TRAIN_RATIO_TOL",
        "TRAIN_TARGET",
        "VAL_RATIO_TOL",
        "VAL_TARGET",
    ],
    "shared": [
        "ALL_TOUCHED",
        "BASE",
        "BATCH",
        "BATCH_L",
        "BATCH_U",
        "BUFFER_COST_WEIGHT",
        "BUFFER_NORMALIZATION",
        "BUFFER_PIXELS",
        "CANDIDATE_STRIDE",
        "CONF_HIGH_END",
        "CONF_HIGH_START",
        "CONF_LOW_END",
        "CONF_LOW_START",
        "EMA_DECAY",
        "INFERENCE_BATCH",
        "LABELED_SAMPLES_PER_EPOCH",
        "LABELED_TARGET",
        "LABEL_BLOCK_SIZE",
        "LABEL_HASH_A",
        "LABEL_HASH_B",
        "LABEL_HASH_GROUPS_TO_SELECT",
        "LABEL_HASH_MOD",
        "LABEL_PREVALENCE_COST_WEIGHT",
        "LABEL_PREVALENCE_TOL",
        "LABEL_SIZE_COST_WEIGHT",
        "LABEL_SIZE_TOL",
        "LAMBDA_MAX",
        "LR",
        "MAX_POSITIVE_PREVALENCE_DEVIATION",
        "MIN_CHANNEL_CORRELATION",
        "MIN_FULL_CORRELATION",
        "MIN_LABELED_PIXELS_IN_PATCH",
        "MIN_MINORITY_CLASS_FRACTION",
        "MIN_REGION_WIDTH",
        "MIN_UNLABELED_PIXELS_IN_PATCH",
        "NOISE_STD",
        "NUM_WORKERS",
        "PATCH",
        "PATCH_SIZE",
        "PERSISTENT_WORKERS",
        "PIN_MEMORY",
        "PREFETCH_FACTOR",
        "PREVALENCE_COST_WEIGHT",
        "RAMPUP_EPOCHS",
        "RATIO_COST_WEIGHT",
        "SAMPLES_PER_EPOCH",
        "SAVE_ERROR_MAP",
        "SAVE_RAW_Z",
        "SEARCH_STEP",
        "SEED",
        "SIGMAS",
        "STRIDE",
        "TEST_RATIO_TOL",
        "TEST_TARGET",
        "THRESHOLDS",
        "TRAIN_RATIO_TOL",
        "TRAIN_TARGET",
        "UNLABELED_SAMPLES_PER_EPOCH",
        "VAL_RATIO_TOL",
        "VAL_TARGET",
        "WEIGHT_DECAY",
    ],
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run one stage of the SAR change-detection pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON configuration; relative paths resolve beside this file.",
    )
    parser.add_argument(
        "--stage", choices=("validate", "test"), help="Required for the ABS evaluator only."
    )
    parser.add_argument(
        "--resume", type=Path, help="Trusted training checkpoint. RNG state is not restored."
    )
    return parser.parse_args()


def configure(args, method, stage, defaults):
    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    allowed = {
        "data",
        "output_root",
        "run_name",
        "epochs",
        "epoch_start",
        "epoch_end",
        "parameters",
        "ground_truth",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    data = config.get("data", {})
    unknown_data = set(data) - {"before_dir", "after", "bitemporal_before", "image_key"}
    if unknown_data:
        raise ValueError(f"Unknown data fields: {sorted(unknown_data)}")
    gt_keys = {
        "reference_tif",
        "reference_mat",
        "aoi_vector",
        "event_vector",
        "label_field",
        "positive_labels",
        "ignore_labels",
    }
    if set(config.get("ground_truth", {})) - gt_keys:
        raise ValueError("Unknown ground_truth fields.")
    if args.stage is not None and not (method == "unsupervised" and stage == "evaluate"):
        raise ValueError("--stage is only supported by evaluate_unsupervised.py.")
    if args.resume is not None and stage != "train":
        raise ValueError("--resume is only supported by training scripts.")

    def path(value):
        p = Path(value).expanduser()
        return (config_path.parent / p).resolve() if not p.is_absolute() else p.resolve()

    before_dir = path(data.get("before_dir", "data/before"))
    after_file = path(data.get("after", "data/after/after.mat"))
    before_file = path(data.get("bitemporal_before", "data/before/before.mat"))
    root = path(config.get("output_root", "outputs"))
    run_name = config.get("run_name", "experiment_01")
    if (
        not isinstance(run_name, str)
        or not run_name
        or Path(run_name).name != run_name
        or run_name in {".", ".."}
        or "\\" in run_name
    ):
        raise ValueError("run_name must be a single nonempty directory name.")
    model_root = root / method
    checkpoints = model_root / "checkpoints"
    predictions = model_root / "predictions"
    results = model_root / "results"
    baseline = model_root / "baselines"
    run_dir = checkpoints / run_name
    gt_dir = root / "ground_truth"
    split_dir = root / "spatial_split"
    for key in ("epochs", "epoch_start", "epoch_end"):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f"{key} must be a positive integer.")
    epochs = int(config.get("epochs", 50))
    first = int(config.get("epoch_start", 1))
    last = int(config.get("epoch_end", epochs))
    if not 1 <= first <= last <= epochs:
        raise ValueError("Require 1 <= epoch_start <= epoch_end <= epochs.")
    parameters = config.get("parameters", {})
    known_groups = {"shared", "spatial_split", "ground_truth", *METHODS}
    if set(parameters) - known_groups:
        raise ValueError("Unknown parameters group.")
    for group, values in parameters.items():
        unknown_keys = set(values) - set(PARAMETER_KEYS[group])
        if unknown_keys:
            raise ValueError(f"Unknown/unsupported parameters in {group}: {sorted(unknown_keys)}")
        for key, value in values.items():
            sequence = value if isinstance(value, list) else [value]
            if not sequence or any(
                not isinstance(x, (int, float, bool)) or not math.isfinite(x) for x in sequence
            ):
                raise ValueError(f"{group}.{key} must contain finite numeric/boolean values.")
    overrides = {**parameters.get("shared", {}), **parameters.get(method, {})}
    # Only declared numerical/boolean algorithm settings are configurable here.
    forbidden = {
        k
        for k in overrides
        if not k.isupper()
        or k.endswith(("_DIR", "_ROOT", "_PATH", "_MAT", "_NAME"))
        or k
        in {
            "EPOCHS",
            "EPOCH_START",
            "EPOCH_END",
            "IN_CH",
            "EPS",
            "EXPECTED_BEFORE_COUNT",
            "RESUME",
            "TARGETS",
            "RATIO_TOLS",
            "NODATA_VALUE",
        }
    }
    if forbidden:
        raise ValueError(f"Use the top-level/data fields for these settings: {sorted(forbidden)}")
    settings = {k: v for k, v in overrides.items() if k in defaults}
    for key, value in settings.items():
        default = defaults[key]
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean.")
        if (
            isinstance(default, int)
            and not isinstance(default, bool)
            and (not isinstance(value, int) or isinstance(value, bool))
        ):
            raise ValueError(f"{key} must be an integer.")
    positive_keys = {
        "BASE",
        "PATCH",
        "PATCH_SIZE",
        "STRIDE",
        "BATCH",
        "BATCH_L",
        "BATCH_U",
        "INFERENCE_BATCH",
        "PREFETCH_FACTOR",
        "CANDIDATE_STRIDE",
        "SEARCH_STEP",
        "MIN_REGION_WIDTH",
        "LABEL_BLOCK_SIZE",
        "LABEL_HASH_MOD",
        "LABEL_HASH_GROUPS_TO_SELECT",
        "SAMPLES_PER_EPOCH",
        "LABELED_SAMPLES_PER_EPOCH",
        "UNLABELED_SAMPLES_PER_EPOCH",
        "RAMPUP_EPOCHS",
        "TRAIN_RATIO_TOL",
        "VAL_RATIO_TOL",
        "TEST_RATIO_TOL",
        "MAX_POSITIVE_PREVALENCE_DEVIATION",
        "BUFFER_NORMALIZATION",
    }
    for key in positive_keys & settings.keys():
        if not isinstance(settings[key], (int, float)) or settings[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if "NUM_WORKERS" in settings and settings["NUM_WORKERS"] < 0:
        raise ValueError("NUM_WORKERS must be nonnegative.")
    if "SIGMAS" in settings and (
        not isinstance(settings["SIGMAS"], list) or min(settings["SIGMAS"]) < 0
    ):
        raise ValueError("SIGMAS must be a nonempty list of nonnegative numbers.")
    if "THRESHOLDS" in settings:
        import numpy as np

        settings["THRESHOLDS"] = np.asarray(settings["THRESHOLDS"], dtype=float)
        if settings["THRESHOLDS"].ndim != 1 or not np.all(
            (settings["THRESHOLDS"] > 0) & (settings["THRESHOLDS"] < 1)
        ):
            raise ValueError("THRESHOLDS must be a list strictly between zero and one.")
    settings.update(
        {
            "ROOT": root,
            "THESIS_ROOT": root,
            "BEFORE_DIR": before_dir,
            "AFTER_DIR": after_file.parent,
            "AFTER_PATH": after_file,
            "BEFORE_PATH": before_file,
            "LABELED_BEFORE_PATH": before_file,
            "GT_MAT": gt_dir / "ground_truth.mat",
            "SPLIT_MAT": split_dir / "spatial_split.mat",
            "REFERENCE_TIF": gt_dir / "ground_truth.tif",
            "OUT_ROOT": checkpoints,
            "CKPT_ROOT": checkpoints,
            "CHECKPOINT_ROOT": checkpoints,
            "CKPT_DIR": run_dir,
            "RUN_DIR": run_dir,
            "RUN_NAME": run_name,
            "BASELINE_ROOT": baseline,
            "OUTPUT_ROOT": predictions,
            "PREDICTION_ROOT": predictions,
            "PRED_ROOT": predictions,
            "PREDICTION_DIR": predictions / run_name,
            "RESULT_ROOT": results,
            "RESULTS_ROOT": results,
            "RESULTS_DIR": results / run_name,
            "OUTPUT_DIR": split_dir if stage == "split" else predictions / run_name,
            "EXPECTED_BEFORE": before_file.name,
            "EXPECTED_BEFORE_NAME": before_file.name,
            "EXPECTED_AFTER": after_file.name,
            "EXPECTED_AFTER_NAME": after_file.name,
            "EXPECTED_BEFORE_COUNT": len(list(before_dir.glob("*.mat"))),
            "EPOCH_START": first,
            "EPOCH_END": last,
            "EPOCHS": list(range(first, last + 1)) if stage == "evaluate" else epochs,
            "RESUME": args.resume is not None,
            "RESUME_CKPT_PATH": args.resume.expanduser().resolve() if args.resume else None,
            "NUM_WORKERS": overrides.get("NUM_WORKERS", 0),
            "PERSISTENT_WORKERS": overrides.get("PERSISTENT_WORKERS", False),
            "AUTO_RESUME": False,
            "IMAGE_KEY": data.get("image_key", "croppedImg"),
            "CONFIG_PATH": config_path,
            "PIPELINE_CONFIG": config,
        }
    )
    if stage == "split":
        import numpy as np

        settings["TARGETS"] = np.array(
            [settings.get(k, defaults[k]) for k in ("TRAIN_TARGET", "VAL_TARGET", "TEST_TARGET")]
        )
        if not np.isclose(settings["TARGETS"].sum(), 1):
            raise ValueError("Spatial split targets must sum to one.")
        settings["RATIO_TOLS"] = np.array(
            [
                settings.get(k, defaults[k])
                for k in ("TRAIN_RATIO_TOL", "VAL_RATIO_TOL", "TEST_RATIO_TOL")
            ]
        )
    patch = int(settings.get("PATCH", defaults.get("PATCH", 256)))
    stride = int(settings.get("STRIDE", defaults.get("STRIDE", 128)))
    if patch < 4 or patch % 4:
        raise ValueError("PATCH must be a positive multiple of four.")
    if stage in {"infer", "baseline"} and not 1 <= stride <= patch:
        raise ValueError("STRIDE must be between 1 and PATCH.")
    if stage in {"train", "infer", "baseline"}:
        before_files = sorted(before_dir.glob("*.mat"))
        if not before_files:
            raise FileNotFoundError(f"No BEFORE MAT files in {before_dir}")
        if method.endswith("multitemporal") or method == "unsupervised":
            if len(before_files) < 2:
                raise ValueError("At least two pre-event acquisitions are required.")
        if after_file.resolve() in [p.resolve() for p in before_files]:
            raise ValueError("AFTER must not be included in the BEFORE directory.")
        if method.endswith("bitemporal") and before_file.parent != before_dir:
            raise ValueError("bitemporal_before must be inside before_dir.")
        if method.endswith("bitemporal") and not before_file.is_file():
            raise FileNotFoundError(before_file)
    return settings
