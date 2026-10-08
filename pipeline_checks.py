"""Validate spatial masks and preserve provenance between pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_masks(split_path, gt_path=None):
    data = loadmat(split_path)
    names = (
        "train_pool",
        "labeled_train",
        "unlabeled_train",
        "validation",
        "test",
        "buffer_mask",
        "valid_mask",
    )
    masks = {}
    for name in names:
        if name not in data:
            raise KeyError(f"Missing spatial mask: {name}")
        raw = np.squeeze(data[name])
        if raw.ndim != 2 or not np.isin(raw, [0, 1]).all():
            raise ValueError(f"{name} must be a two-dimensional binary mask.")
        masks[name] = raw.astype(bool)
    if len({x.shape for x in masks.values()}) != 1:
        raise ValueError("Spatial masks have different dimensions.")
    major = [masks[k] for k in ("train_pool", "validation", "test", "buffer_mask")]
    for i, left in enumerate(major):
        for right in major[i + 1 :]:
            if (left & right).any():
                raise ValueError("Spatial regions overlap.")
    if not np.array_equal(np.logical_or.reduce(major), masks["valid_mask"]):
        raise ValueError("Spatial regions do not reconstruct valid_mask.")
    labeled, unlabeled = masks["labeled_train"], masks["unlabeled_train"]
    if (labeled & unlabeled).any() or not np.array_equal(labeled | unlabeled, masks["train_pool"]):
        raise ValueError("Labeled/unlabeled masks must partition train_pool.")
    for name in ("labeled_train", "unlabeled_train", "validation", "test"):
        if not masks[name].any():
            raise ValueError(f"Empty spatial region: {name}")
    if gt_path is not None:
        # Read validity only; held-out class labels are not needed by training.
        valid = np.squeeze(loadmat(gt_path, variable_names=["valid_mask"])["valid_mask"])
        if not np.isin(valid, [0, 1]).all() or not np.array_equal(
            valid.astype(bool), masks["valid_mask"]
        ):
            raise ValueError("Ground-truth and split validity masks differ.")
    return masks


def data_signature(g):
    files = sorted(g["BEFORE_DIR"].glob("*.mat")) + [g["AFTER_PATH"]]
    return {str(p.resolve()): sha256(p) for p in files}


def read_record(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pipeline provenance record: {path}. Run the preceding stage first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def preflight(g, method, stage):
    if g["ARGS"].resume and stage != "train":
        raise ValueError("--resume is only supported by training scripts.")
    validate_masks(g["SPLIT_MAT"], g["GT_MAT"])
    split_hash = sha256(g["SPLIT_MAT"])
    gt_hash = sha256(g["GT_MAT"])
    if stage == "train":
        initial = dict(
            method=method,
            split_sha256=split_hash,
            gt_sha256=gt_hash,
            data_sha256=data_signature(g),
            config=g["PIPELINE_CONFIG"],
        )
        initial_path = g["RUN_DIR"] / "pipeline_inputs.json"
        if g["RESUME"]:
            if read_record(initial_path) != initial:
                raise ValueError("Resume inputs/config differ from the initial training run.")
        for sample_key, batch_key in (
            ("LABELED_SAMPLES_PER_EPOCH", "BATCH_L"),
            ("UNLABELED_SAMPLES_PER_EPOCH", "BATCH_U"),
            ("LABELED_SAMPLES_PER_EPOCH", "BATCH"),
            ("SAMPLES_PER_EPOCH", "BATCH"),
        ):
            if sample_key in g and batch_key in g:
                if g[sample_key] < g[batch_key] or g[batch_key] < 1:
                    raise ValueError(
                        "Each loader needs at least one complete positive-sized batch."
                    )
        if (
            "BATCH_U" in g
            and g["LABELED_SAMPLES_PER_EPOCH"] // g["BATCH_L"]
            != g["UNLABELED_SAMPLES_PER_EPOCH"] // g["BATCH_U"]
        ):
            raise ValueError("Labeled and unlabeled loaders must have the same number of batches.")
        if g["RESUME"]:
            import torch

            checkpoint = torch.load(g["RESUME_CKPT_PATH"], map_location="cpu", weights_only=False)
            if Path(checkpoint["run_dir"]).resolve() != g["RUN_DIR"]:
                raise ValueError("Resume checkpoint does not belong to configured run_name.")
            for key, value in checkpoint.get("config", {}).items():
                if key in g and isinstance(value, (bool, int, float)) and g[key] != value:
                    raise ValueError(f"Resume setting differs from checkpoint: {key}")
        return
    record = read_record(g["RUN_DIR"] / "pipeline_run.json")
    if (
        record["method"] != method
        or record["split_sha256"] != split_hash
        or record["gt_sha256"] != gt_hash
    ):
        raise ValueError("Method, ground truth or spatial split differs from the training run.")
    if stage in {"infer", "baseline"}:
        for epoch in range(g["EPOCH_START"], g["EPOCH_END"] + 1):
            name = f"epoch_{epoch:03d}.pt"
            if record["checkpoints"].get(name) != sha256(g["RUN_DIR"] / name):
                raise ValueError(f"Checkpoint differs from completed training: {name}")
        if record["data_sha256"] != data_signature(g):
            raise ValueError("SAR inputs differ from the training run.")
        for key in ("BASE", "PATCH", "IN_CH", "IMAGE_KEY"):
            if record["settings"][key] != g[key]:
                raise ValueError(f"{key} differs from the training run.")
        if stage == "infer" and method == "unsupervised":
            baseline = read_record(g["BASELINE_ROOT"] / g["RUN_NAME"] / "pipeline_baselines.json")
            if (
                baseline["split_sha256"] != split_hash
                or baseline["patch"] != g["PATCH"]
                or baseline["stride"] != g["STRIDE"]
            ):
                raise ValueError(
                    "Baseline split or sliding-window settings do not match inference."
                )
            for name, digest in baseline["maps"].items():
                if sha256(g["BASELINE_ROOT"] / g["RUN_NAME"] / name) != digest:
                    raise ValueError(f"Baseline map changed: {name}")
            for epoch in range(g["EPOCH_START"], g["EPOCH_END"] + 1):
                name = f"epoch_{epoch:03d}.pt"
                if baseline["checkpoints"].get(name) != sha256(g["RUN_DIR"] / name):
                    raise ValueError(f"Baseline checkpoint differs: {name}")
    if stage == "evaluate":
        pred = read_record(g["PREDICTION_DIR"] / "pipeline_predictions.json")
        if (
            pred["split_sha256"] != split_hash
            or pred["gt_sha256"] != gt_hash
            or pred["method"] != method
        ):
            raise ValueError("Prediction provenance does not match the evaluation inputs.")
        for name, expected in pred["maps"].items():
            if sha256(g["PREDICTION_DIR"] / name) != expected:
                raise ValueError(f"Prediction map changed after inference: {name}")


def record_outputs(g, method, stage):
    common = dict(method=method, split_sha256=sha256(g["SPLIT_MAT"]), gt_sha256=sha256(g["GT_MAT"]))
    if stage == "train":
        payload = dict(
            common,
            checkpoints={p.name: sha256(p) for p in sorted(g["RUN_DIR"].glob("epoch_*.pt"))},
            data_sha256=data_signature(g),
            settings={k: g[k] for k in ("BASE", "PATCH", "IN_CH", "IMAGE_KEY")},
        )
        target = g["RUN_DIR"] / "pipeline_run.json"
    elif stage == "baseline":
        payload = dict(
            common,
            maps={
                p.name: sha256(p)
                for p in sorted((g["BASELINE_ROOT"] / g["RUN_NAME"]).glob("*.mat"))
            },
            patch=g["PATCH"],
            stride=g["STRIDE"],
            checkpoints={
                f"epoch_{ep:03d}.pt": sha256(g["RUN_DIR"] / f"epoch_{ep:03d}.pt")
                for ep in range(g["EPOCH_START"], g["EPOCH_END"] + 1)
            },
        )
        target = g["BASELINE_ROOT"] / g["RUN_NAME"] / "pipeline_baselines.json"
    elif stage == "infer":
        directory = g["PREDICTION_DIR"]
        payload = dict(common, maps={p.name: sha256(p) for p in sorted(directory.glob("*.mat"))})
        target = directory / "pipeline_predictions.json"
    else:
        return
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_training_inputs(g, method):
    """Persist input identity before the first epoch, including interrupted runs."""
    payload = dict(
        method=method,
        split_sha256=sha256(g["SPLIT_MAT"]),
        gt_sha256=sha256(g["GT_MAT"]),
        data_sha256=data_signature(g),
        config=g["PIPELINE_CONFIG"],
    )
    (g["RUN_DIR"] / "pipeline_inputs.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
