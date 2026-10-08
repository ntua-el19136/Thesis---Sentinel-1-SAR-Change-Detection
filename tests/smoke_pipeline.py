"""Generate synthetic inputs and run all five CPU pipelines in an empty directory."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.io import savemat
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[1]


def make_inputs(root):
    (root / "data/before").mkdir(parents=True)
    (root / "data/after").mkdir(parents=True)
    rng = np.random.default_rng(17)
    shape = (64, 128)
    for i in range(3):
        image = rng.normal(size=(*shape, 2)).astype("float32")
        savemat(root / f"data/before/before_{i}.mat", {"croppedImg": image})
        if i == 0:
            with rasterio.open(
                root / "data/reference.tif",
                "w",
                driver="GTiff",
                height=shape[0],
                width=shape[1],
                count=2,
                dtype="float32",
                crs="EPSG:32634",
                transform=from_origin(0, 640, 10, 10),
            ) as dst:
                dst.write(image.transpose(2, 0, 1))
    after = rng.normal(size=(*shape, 2)).astype("float32")
    after[:32] += 0.7
    savemat(root / "data/after/after.mat", {"croppedImg": after})
    gpd.GeoDataFrame(
        {"notation": ["affected", "uncertain"]},
        geometry=[box(0, 320, 1280, 640), box(0, 0, 20, 20)],
        crs="EPSG:32634",
    ).to_file(root / "data/events.geojson", driver="GeoJSON")
    gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[box(0, 0, 1280, 640)], crs="EPSG:32634").to_file(
        root / "data/aoi.geojson", driver="GeoJSON"
    )
    config = {
        "data": {
            "before_dir": "data/before",
            "after": "data/after/after.mat",
            "bitemporal_before": "data/before/before_0.mat",
        },
        "output_root": "outputs",
        "run_name": "smoke",
        "epochs": 1,
        "ground_truth": {
            "reference_tif": "data/reference.tif",
            "reference_mat": "data/before/before_0.mat",
            "aoi_vector": "data/aoi.geojson",
            "event_vector": "data/events.geojson",
            "positive_labels": ["affected"],
            "ignore_labels": ["uncertain"],
        },
        "parameters": {
            "shared": {
                "BASE": 2,
                "PATCH": 16,
                "STRIDE": 16,
                "BATCH": 1,
                "BATCH_L": 1,
                "BATCH_U": 1,
                "LABELED_SAMPLES_PER_EPOCH": 2,
                "UNLABELED_SAMPLES_PER_EPOCH": 2,
                "SAMPLES_PER_EPOCH": 2,
                "NUM_WORKERS": 0,
                "INFERENCE_BATCH": 4,
                "CANDIDATE_STRIDE": 4,
            },
            "spatial_split": {
                "PATCH_SIZE": 16,
                "BUFFER_PIXELS": 4,
                "MIN_REGION_WIDTH": 16,
                "SEARCH_STEP": 1,
                "LABEL_BLOCK_SIZE": 8,
            },
        },
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="New or empty directory for generated synthetic data and outputs.",
    )
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty work directory; existing runs are preserved.")
    root.mkdir(parents=True, exist_ok=True)
    make_inputs(root)
    logs = root / "logs"
    logs.mkdir()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "CUDA_VISIBLE_DEVICES": "",
    }
    stages = [("create_ground_truth", []), ("create_spatial_split", [])]
    for method in [
        "supervised_bitemporal",
        "supervised_multitemporal",
        "semisupervised_bitemporal",
        "semisupervised_multitemporal",
        "unsupervised",
    ]:
        stages.append(("train_" + method, []))
        if method == "unsupervised":
            stages.append(("build_unsupervised_baselines", []))
        stages.extend(
            [
                ("infer_" + method, []),
                ("evaluate_" + method, ["--stage", "validate"] if method == "unsupervised" else []),
            ]
        )
        if method == "unsupervised":
            stages.append(("evaluate_unsupervised", ["--stage", "test"]))
    stages.append(("tests/verify_outputs", []))
    results = []
    for name, extra in stages:
        label = name.replace("/", "_") + ("_" + extra[-1] if extra else "")
        start = time.monotonic()
        log = logs / (label + ".log")
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / (name + ".py")),
                    "--config",
                    str(root / "config.json"),
                    *extra,
                ],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        row = dict(
            stage=label, returncode=result.returncode, seconds=round(time.monotonic() - start, 2)
        )
        results.append(row)
        (root / "smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(row, flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {log}")
    print("All CPU pipeline stages and integrity checks passed.")


if __name__ == "__main__":
    main()
