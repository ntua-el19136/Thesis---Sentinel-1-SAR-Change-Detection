"""Rasterize event labels on a verified Sentinel-1 reference grid.

Usage: python "ground_truth_and_spatial_split/create_ground_truth.py" --config config.json
The reference MAT and TIFF must represent the same acquisition and crop.
Vector layers may be shapefiles, GeoJSON or another GeoPandas-supported format.
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

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.io import loadmat, savemat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL_TOUCHED = False
MIN_FULL_CORRELATION = 0.85
MIN_CHANNEL_CORRELATION = 0.75
NODATA_VALUE = 255
globals().update(
    configure(
        ARGS,
        "ground_truth",
        "ground_truth",
        {k: v for k, v in globals().copy().items() if k.isupper()},
    )
)


def resolve_path(value):
    path = Path(value).expanduser()
    return (CONFIG_PATH.parent / path).resolve() if not path.is_absolute() else path.resolve()


def read_image(path):
    data = loadmat(path)
    if IMAGE_KEY not in data:
        raise KeyError(f"Missing {IMAGE_KEY!r} in {path}")
    image = np.asarray(data[IMAGE_KEY], dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 2:
        raise ValueError(f"Expected (H, W, 2) ordered VH/VV: {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite SAR values: {path}")
    return image


def verify_grid(reference_mat, reference_tif):
    image = read_image(reference_mat)
    with rasterio.open(reference_tif) as src:
        if src.count != 2 or src.crs is None:
            raise ValueError("Reference TIFF needs two bands and a CRS.")
        if image.shape[:2] != (src.height, src.width):
            raise ValueError("Reference MAT and TIFF dimensions differ.")
        bands = src.read().astype(np.float64)
        valid = np.all(src.read_masks() > 0, axis=0) & np.isfinite(bands).all(axis=0)
        if valid.sum() < 2:
            raise ValueError("Reference TIFF has too few valid pixels.")
        candidates = []
        for vh_band, vv_band in ((0, 1), (1, 0)):
            correlations = [
                float(np.corrcoef(image[:, :, c][valid], bands[b][valid])[0, 1])
                for c, b in ((0, vh_band), (1, vv_band))
            ]
            candidates.append(
                dict(
                    vh_band=vh_band + 1,
                    vv_band=vv_band + 1,
                    correlations=correlations,
                    combined=float(np.mean(correlations)),
                )
            )
        finite = [
            c
            for c in candidates
            if np.isfinite(c["combined"]) and np.isfinite(c["correlations"]).all()
        ]
        if not finite:
            raise ValueError("Reference correlation is undefined.")
        best = max(finite, key=lambda c: c["combined"])
        if (
            best["combined"] < MIN_FULL_CORRELATION
            or min(best["correlations"]) < MIN_CHANNEL_CORRELATION
        ):
            raise ValueError(f"Reference MAT/TIFF correlation check failed: {best}")
        return src.profile.copy(), valid, best


def read_polygons(path, crs):
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Vector layer has no CRS: {path}")
    frame = frame.to_crs(crs)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    invalid = ~frame.geometry.is_valid
    if invalid.any():
        frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].buffer(0)
    if frame.empty or not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"Expected nonempty polygon geometry: {path}")
    return frame


def raster_mask(frame, profile):
    shape = (profile["height"], profile["width"])
    if frame.empty:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        ((geom, 1) for geom in frame.geometry),
        out_shape=shape,
        transform=profile["transform"],
        all_touched=ALL_TOUCHED,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def save_tif(path, array, profile, nodata=None):
    output = profile.copy()
    output.update(driver="GTiff", count=1, dtype="uint8", compress="lzw", nodata=nodata)
    with rasterio.open(path, "w", **output) as dst:
        dst.write(array.astype(np.uint8), 1)


def main():
    if GT_MAT.parent.exists() and any(GT_MAT.parent.iterdir()):
        raise FileExistsError(
            "Ground-truth output exists. Use a new output_root or archive the existing directory first."
        )
    cfg = PIPELINE_CONFIG["ground_truth"]
    ref_tif = resolve_path(cfg["reference_tif"])
    ref_mat = resolve_path(cfg["reference_mat"])
    profile, image_valid, verification = verify_grid(ref_mat, ref_tif)
    # MAT arrays do not carry a CRS; their co-registration remains an input requirement.
    acquisitions = sorted(BEFORE_DIR.glob("*.mat")) + [AFTER_PATH]
    if len(acquisitions) < 3:
        raise ValueError("At least two BEFORE images and one AFTER image are required.")
    for path in acquisitions:
        image = read_image(path)
        if image.shape[:2] != image_valid.shape:
            raise ValueError(f"Acquisition grid dimensions differ: {path}")
    events = read_polygons(resolve_path(cfg["event_vector"]), profile["crs"])
    aoi = read_polygons(resolve_path(cfg["aoi_vector"]), profile["crs"])
    field = cfg.get("label_field", "notation")
    if field not in events:
        raise KeyError(f"Event layer is missing {field!r}")
    labels = events[field].astype(str).str.strip().str.casefold()
    positives = {str(v).strip().casefold() for v in cfg["positive_labels"]}
    ignored = {str(v).strip().casefold() for v in cfg.get("ignore_labels", [])}
    if not positives or positives & ignored:
        raise ValueError("Positive labels must be nonempty and disjoint from ignored labels.")
    positive_frame = events[labels.isin(positives)]
    if positive_frame.empty:
        raise ValueError(f"No positive polygons found. Available labels: {sorted(labels.unique())}")
    positive_mask = raster_mask(positive_frame, profile)
    ignore_mask = raster_mask(events[labels.isin(ignored)], profile)
    aoi_mask = raster_mask(aoi, profile)
    valid = image_valid & aoi_mask & ~ignore_mask
    gt = positive_mask.astype(np.uint8)
    if not valid.any() or np.unique(gt[valid]).size != 2:
        raise ValueError("Valid ground truth must contain both classes.")
    directory = GT_MAT.parent
    directory.mkdir(parents=True, exist_ok=True)
    encoded = np.full(valid.shape, NODATA_VALUE, dtype=np.uint8)
    encoded[valid] = gt[valid]
    signed = np.full(valid.shape, -1, dtype=np.int8)
    signed[valid] = gt[valid]
    epsg = profile["crs"].to_epsg()
    savemat(
        GT_MAT,
        dict(
            ground_truth=gt,
            valid_mask=valid.astype(np.uint8),
            ignore_mask=(~valid).astype(np.uint8),
            ground_truth_with_ignore=signed,
            aoi_mask=aoi_mask.astype(np.uint8),
            s1_valid_mask=image_valid.astype(np.uint8),
            crs_epsg=np.int32(epsg if epsg is not None else -1),
            crs_wkt=profile["crs"].to_wkt(),
            pixel_size_x=abs(profile["transform"].a),
            pixel_size_y=abs(profile["transform"].e),
        ),
        do_compression=True,
    )
    save_tif(directory / "ground_truth.tif", encoded, profile, NODATA_VALUE)
    save_tif(directory / "valid_mask.tif", valid, profile)
    save_tif(directory / "aoi_mask.tif", aoi_mask, profile)
    report = dict(
        verification=verification,
        valid_pixels=int(valid.sum()),
        positive_pixels=int((positive_mask & valid).sum()),
        ignored_pixels=int((~valid).sum()),
        positive_labels=sorted(positives),
        ignore_labels=sorted(ignored),
        crs=str(profile["crs"]),
        all_touched=ALL_TOUCHED,
    )
    (directory / "ground_truth_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    preview = np.where(valid, gt, 2)
    plot = ax.imshow(preview, vmin=0, vmax=2)
    ax.set_title("Ground truth: unchanged / changed / ignored")
    fig.colorbar(plot, ax=ax, ticks=[0, 1, 2])
    fig.savefig(directory / "ground_truth_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print("Saved:", GT_MAT)


if __name__ == "__main__":
    main()
