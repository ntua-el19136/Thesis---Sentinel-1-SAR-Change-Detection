"""Check a completed synthetic run; every temporary mutation is restored."""

from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline_checks import preflight, validate_masks


def verify(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    passed = []

    def load(script, config_file=config_path):
        old = sys.argv
        try:
            sys.argv = [script, "--config", str(config_file)]
            return runpy.run_path(str(REPO / script.split("_", 1)[1] / (script + ".py")))
        finally:
            sys.argv = old

    def rejected(label, action, fragment):
        try:
            action()
        except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as error:
            assert fragment in str(error), (label, str(error))
            passed.append(label)
        else:
            raise AssertionError(f"{label}: invalid input was accepted")

    @contextlib.contextmanager
    def altered(path):
        path = Path(path)
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed-for-integrity-check")
            yield
        finally:
            path.write_bytes(original)

    g = load("infer_supervised_bitemporal")
    with altered(g["AFTER_PATH"]):
        rejected(
            "changed SAR data",
            lambda: preflight(g, "supervised_bitemporal", "infer"),
            "SAR inputs differ",
        )
    with altered(g["RUN_DIR"] / "epoch_001.pt"):
        rejected(
            "changed checkpoint",
            lambda: preflight(g, "supervised_bitemporal", "infer"),
            "Checkpoint differs",
        )
    original = g["SPLIT_MAT"].read_bytes()
    try:
        masks = loadmat(g["SPLIT_MAT"])
        clean = {k: v for k, v in masks.items() if not k.startswith("__")}
        clean["verification_marker"] = np.array([1])
        savemat(g["SPLIT_MAT"], clean)
        rejected(
            "different valid split file",
            lambda: preflight(g, "supervised_bitemporal", "infer"),
            "differs from the training run",
        )
        clean["test"] = clean["test"].copy()
        y, x = np.argwhere(clean["train_pool"] > 0)[0]
        clean["test"][y, x] = 1
        savemat(g["SPLIT_MAT"], clean)
        rejected("overlapping train/test masks", lambda: validate_masks(g["SPLIT_MAT"]), "overlap")
    finally:
        g["SPLIT_MAT"].write_bytes(original)
    bad_g = {**g, "BASE": g["BASE"] + 1}
    rejected(
        "incompatible architecture",
        lambda: preflight(bad_g, "supervised_bitemporal", "infer"),
        "BASE differs",
    )

    for label, modify, fragment in [
        (
            "unknown parameter",
            lambda c: c["parameters"]["shared"].update(PATC=16),
            "Unknown/unsupported",
        ),
        (
            "AFTER among BEFOREs",
            lambda c: c["data"].update(after=c["data"]["bitemporal_before"]),
            "AFTER must not",
        ),
        ("noninteger epoch", lambda c: c.update(epochs=1.5), "positive integer"),
    ]:
        c = copy.deepcopy(config)
        modify(c)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=config_path.parent, delete=False
        ) as file:
            json.dump(c, file)
            path = Path(file.name)
        try:
            rejected(label, lambda: load("infer_supervised_bitemporal", path), fragment)
        finally:
            path.unlink()

    ae = load("infer_unsupervised")
    with altered(ae["BASELINE_ROOT"] / ae["RUN_NAME"] / "global_001.mat"):
        rejected(
            "changed AE baseline",
            lambda: preflight(ae, "unsupervised", "infer"),
            "Baseline map changed",
        )
    ev = load("evaluate_unsupervised")
    pred = ev["PREDICTION_DIR"] / "change_prob_global_001.mat"
    with altered(pred):
        rejected(
            "changed prediction",
            lambda: preflight(ev, "unsupervised", "evaluate"),
            "Prediction map changed",
        )
    frozen_path = ev["RESULTS_DIR"] / "frozen_abs_configuration.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        (directory / frozen_path.name).write_bytes(frozen_bytes)
        selected = ev["PREDICTION_DIR"] / frozen["source_selected_prediction_file"]
        with altered(selected):
            rejected(
                "changed frozen ABS source",
                lambda: ev["evaluate_frozen_test"](
                    ev["PREDICTION_DIR"], directory, ev["GT_MAT"], ev["SPLIT_MAT"], False
                ),
                "Selected source probability map changed",
            )
        # Reproduce in an isolated temporary directory, without selecting again.
        with contextlib.redirect_stdout(io.StringIO()):
            ev["evaluate_frozen_test"](
                ev["PREDICTION_DIR"], directory, ev["GT_MAT"], ev["SPLIT_MAT"], False
            )
        repeated = json.loads((directory / "abs_final_metrics.json").read_text())
        actual = json.loads((ev["RESULTS_DIR"] / "abs_final_metrics.json").read_text())
        assert repeated["frozen_selection"] == actual["frozen_selection"]
        assert repeated["test"] == actual["test"]
        assert frozen_path.read_bytes() == frozen_bytes
        passed.append("frozen ABS selection reproduces without reselection")
    rejected(
        "repeat held-out ABS evaluation",
        lambda: ev["evaluate_frozen_test"](
            ev["PREDICTION_DIR"], ev["RESULTS_DIR"], ev["GT_MAT"], ev["SPLIT_MAT"], False
        ),
        "Test metrics already saved",
    )
    ev["ARGS"].stage = "validate"
    rejected("overwrite frozen ABS validation", ev["main"], "already frozen")

    methods = [
        "supervised_bitemporal",
        "supervised_multitemporal",
        "semisupervised_bitemporal",
        "semisupervised_multitemporal",
        "unsupervised",
    ]
    for method in methods:
        evaluator = load("evaluate_" + method)
        gt = loadmat(evaluator["GT_MAT"])["ground_truth"]
        masks = validate_masks(evaluator["SPLIT_MAT"], evaluator["GT_MAT"])
        changed_gt = gt.copy()
        changed_gt[masks["test"]] = 1 - changed_gt[masks["test"]]
        score = np.random.default_rng(42).random(gt.shape).astype(np.float32)
        function = (
            evaluator["choose_threshold_from_validation"]
            if method == "unsupervised"
            else evaluator["find_best_threshold"]
        )
        assert function(score, gt, masks["validation"]) == function(
            score, changed_gt, masks["validation"]
        )
        result_dir = evaluator["RESULTS_DIR"]
        aoi = json.loads((result_dir / ("evaluate_" + method + "_aoi_metrics.json")).read_text())
        assert aoi["regions"]["full_aoi"]["pixels"] == int(masks["valid_mask"].sum())
        for region, metrics in aoi["regions"].items():
            assert sum(metrics[k] for k in ("tp", "tn", "fp", "fn")) == metrics["pixels"]
        if method != "unsupervised":
            rejected("frozen selection guard: " + method, evaluator["main"], "already frozen")
        passed.append("held-out-label independence and valid AOI counts: " + method)

    # Exercise candidate generation itself, rather than only split membership.
    train = load("train_supervised_multitemporal")
    masks = validate_masks(train["SPLIT_MAT"])
    with contextlib.redirect_stdout(io.StringIO()):
        indices, width = train["build_labeled_candidates"](
            masks["labeled_train"], masks["train_pool"], train["PATCH"]
        )
    for flat in indices:
        y, x = divmod(int(flat), width)
        assert masks["train_pool"][y : y + train["PATCH"], x : x + train["PATCH"]].all()
        assert masks["labeled_train"][y : y + train["PATCH"], x : x + train["PATCH"]].any()
    passed.append("all supervised multitemporal candidate patches stay in train_pool")
    train["ARGS"].resume = train["RUN_DIR"] / "epoch_001.pt"
    train["RESUME"] = True
    train["RESUME_CKPT_PATH"] = train["ARGS"].resume
    preflight(train, "supervised_multitemporal", "train")
    passed.append("same-input resume preflight")
    with altered(train["AFTER_PATH"]):
        rejected(
            "resume with changed data",
            lambda: preflight(train, "supervised_multitemporal", "train"),
            "Resume inputs/config differ",
        )
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config), indent=2))
