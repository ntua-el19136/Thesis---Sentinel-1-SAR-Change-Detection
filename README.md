# Sentinel-1 SAR change detection

Example code accompanying **Mapping Floods and Burned Areas Using Sentinel-1: Development of a Unified Framework for Comparing Convolutional Neural Network Architectures and Learning Strategies**.

Greek thesis title: «Χαρτογράφηση πλημμυρών και καμένων εκτάσεων με Sentinel-1: Ανάπτυξη ενιαίου πλαισίου για τη σύγκριση αρχιτεκτονικών και στρατηγικών μάθησης συνελικτικών νευρωνικών δικτύων».

The repository contains 18 pipeline scripts and two shared helpers. It adapts the supplied research scripts to explicit input paths and a common run configuration. No observations, study areas, ground-truth datasets or trained weights are distributed. The implementations were derived from a burned-area experiment; generic paths and labels do not establish experimental validity on floods or other areas.

## Setup

Use Python 3.10 or newer (tested with Python 3.12 on Linux CPU). Create a virtual environment, activate it and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The scripts automatically use CUDA when available. For a CUDA build, install PyTorch appropriate to your CUDA environment before the remaining requirements. The included checks used CPU only; GPU behavior and large-scene memory consumption were not validated. Checkpoints are loaded with `weights_only=False`: use only checkpoints you generated or otherwise trust.

## Input contract

Copy `config.example.json` to `config.json` and set the paths. Every relative path resolves next to the JSON file, regardless of the shell's working directory.

| Input | Required content |
|---|---|
| `data.before_dir` | Only pre-event `.mat` acquisitions; at least two for the complete five-method workflow. Every file in this directory is used by multitemporal and AE stages. |
| `data.after` | One explicit post-event `.mat`, outside the BEFORE directory. |
| `data.bitemporal_before` | One acquisition inside BEFORE, fixed for both bitemporal methods. |
| Each image MAT | SciPy-readable MATLAB file (not HDF5/v7.3); default key `croppedImg`, finite numeric array `(H, W, 2)`, channels **VH, VV**. `data.image_key` changes this key. |
| `ground_truth.reference_tif` | Georeferenced two-band TIFF with a CRS, the same acquisition and crop as `reference_mat`. TIFF band order is checked in both possible orders against the MAT. |
| `ground_truth.reference_mat` | Matching MAT used to check reference dimensions and per-channel correlation. |
| `ground_truth.aoi_vector` | Polygon or MultiPolygon layer with a CRS defining the area of interest. |
| `ground_truth.event_vector` | Polygon layer with a CRS and the configured label field. `positive_labels` denotes changed areas; `ignore_labels` excludes uncertain areas. Matching strips whitespace and ignores case. |

All acquisitions must have the same dimensions, pixel alignment, crop, orientation, polarization order and consistent SAR preprocessing/radiometric representation. Calibration, terrain correction, co-registration and MAT v7.3 conversion are upstream responsibilities. The scripts do not convert between linear power and dB. A MAT contains no georeferencing; equal dimensions and reference correlation do not prove co-registration of the other dates. Inspect alignment yourself.

The reference check uses full arrays, tests both TIFF band orders and requires mean correlation at least 0.85 and each channel at least 0.75 by default. It does not resample misaligned inputs. Nonfinite MAT values are rejected. Valid reference pixels inside the AOI, excluding ignored polygons, define validity. Positive polygons receive 1; all remaining valid pixels receive 0, including unlisted event labels. Use `ignore_labels` for every class that must not become negative. Ignore takes precedence over positive. Missing event coverage is therefore **not** automatically unknown. Both classes must exist. Inspect the ground-truth preview and report before continuing.

## Shared experimental protocol

The split is constructed once and reused across methods. It searches geographic orientations and buffered boundaries using class-balance constraints: this is **label-informed split construction**, frozen before model fitting or model selection. It is not optimization against test predictions. Default 60/20/20 targets refer to usable valid pixels **after removing buffers**, with a 2.5 percentage-point tolerance per region. The labeled subset targets 20% of the train pool (blockwise, within configured tolerances); the rest is unlabeled. Default minimum minority-class fraction is 20%, and prevalence deviation from the overall valid region is at most 10 percentage points. These constraints can be infeasible, especially for rare changes. Inspect the diagnostics and design any alternative split protocol before training; the code does not silently relax them.

Training candidate patches lie entirely within `train_pool`. Supervised losses use only `labeled_train`. Mean Teacher uses those same labeled pixels plus confidence-filtered consistency on `unlabeled_train`; teacher and student receive the same temporal pair and geometry, with Gaussian noise added to the student. AE trains on BEFORE acquisitions in the train pool without class-label losses. Ignored and held-out pixels are excluded from all training losses.

| Method | Input and prediction |
|---|---|
| Unsupervised AE | Two-channel BEFORE patches; post-event reconstruction error compared with global and pixelwise pre-event baselines. |
| Supervised bitemporal | Four channels: VH/VV from the selected BEFORE, then VH/VV AFTER; masked BCE plus Dice loss. |
| Supervised multitemporal | One randomly selected BEFORE per patch, fixed AFTER, four channels; inference averages probabilities equally across BEFORE acquisitions. |
| Semisupervised bitemporal | Four-channel fixed pair with Mean Teacher; inference uses the EMA teacher. |
| Semisupervised multitemporal | Random BEFORE per training patch, fixed AFTER; EMA-teacher inference averages probabilities equally over BEFOREs. |

Multitemporal here does **not** stack every date as input channels. Normalization is per acquisition/channel: clip at train-pool 1st/99th percentiles, then standardize using the clipped train-pool mean and standard deviation. Training statistics are saved and reused for the corresponding inputs. The AE AFTER image has its own normalization fitted only at train-pool locations. AE global error statistics use train-pool pixels; pixelwise baselines use pre-event observations at every inference location, including held-out locations, without held-out labels. This spatially transductive input protocol should be stated when reporting results.

Select epoch, Gaussian smoothing sigma and threshold by validation IoU only. Default search uses sigma 0/1/2 and 199 thresholds from 0.01 to 0.99. Strictly greater IoU updates the selection, retaining the first encountered tied choice. Each supervised/Mean Teacher evaluator writes its selected configuration before calculating held-out metrics in the same invocation. Existing selections are protected against accidental reselection. AE uses separate `validate` and `test` invocations with a frozen, hashed selection. Fix the search space before examining test results.

**AE score:** inference writes unsmoothed HIGH-direction probabilities; the ABS evaluator applies `max(p_high, 1 - p_high)` **before** smoothing and selects global/pixelwise baseline mode on validation. ABS originated as a post-hoc extension in the supplied research evaluation. The separated protocol in this generic package does not establish that the historical thesis decisions were preregistered or originally made without post-hoc inspection.

Full-AOI metrics and TP/TN/FP/FN maps use the same selected prediction. Full AOI includes valid training, validation, test and buffer pixels; it is descriptive and **not held-out generalization**. The full-AOI and strict-test maps exclude ignored pixels.

## Execution order

Run commands from this repository directory, using the same config and `run_name` throughout. Paths below are generic. First create and inspect preprocessing outputs:

```bash
python create_ground_truth.py --config config.json
python create_spatial_split.py --config config.json
```

Inspect `ground_truth_preview.png`, `ground_truth_report.json`, the split preview and `split_stats.txt`. Confirm the split has valid fully contained training patches for your patch size. Then run any or all methods:

```bash
python train_supervised_bitemporal.py --config config.json
python infer_supervised_bitemporal.py --config config.json
python evaluate_supervised_bitemporal.py --config config.json

python train_supervised_multitemporal.py --config config.json
python infer_supervised_multitemporal.py --config config.json
python evaluate_supervised_multitemporal.py --config config.json

python train_semisupervised_bitemporal.py --config config.json
python infer_semisupervised_bitemporal.py --config config.json
python evaluate_semisupervised_bitemporal.py --config config.json

python train_semisupervised_multitemporal.py --config config.json
python infer_semisupervised_multitemporal.py --config config.json
python evaluate_semisupervised_multitemporal.py --config config.json

python train_unsupervised.py --config config.json
python build_unsupervised_baselines.py --config config.json
python infer_unsupervised.py --config config.json
python evaluate_unsupervised.py --config config.json --stage validate
python evaluate_unsupervised.py --config config.json --stage test
```

AE baselines must precede AE inference and correspond to the same checkpoints, patch and stride. Inference processes every configured epoch; there is no automatic choice of a newest run. `epoch_start`/`epoch_end` restrict inference and selection, not training. If reducing `epochs` in the example, reduce `epoch_end` as well. The default training duration remains 50 epochs; smoke checks use one.

## Configuration and outputs

`parameters.shared` supplies cross-stage settings; a method group overrides them. A setting applies only to stages that declare it. Unknown names are rejected across all groups, while known settings irrelevant to the current stage are allowed. Supported names are listed in `PARAMETERS.md`. Use top-level/data fields for paths, epochs and image keys. `PATCH` must be divisible by four and fit the image; `STRIDE` must be from 1 to `PATCH`. Keep the splitter's `PATCH_SIZE` consistent with model `PATCH`. Mean Teacher labeled/unlabeled loaders must have equal complete-batch counts; each needs at least one complete batch. Adjust batch sizes to available RAM/VRAM.

Defaults are retained from the source algorithms: BASE 32, PATCH 256; AE 8,000 samples/epoch, supervised 8,000 labeled samples/epoch, Mean Teacher 2,000 labeled and 4,000 unlabeled samples/epoch. `NUM_WORKERS=0` is the portable default. For CPU checks, setting `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `MPLBACKEND=Agg` is useful.

| Location under `output_root` | Outputs |
|---|---|
| `ground_truth/` | `ground_truth.mat`, raster masks, preview and verification report. MAT keys include `ground_truth` (0/1), `valid_mask`, `ignore_mask`, `ground_truth_with_ignore` (-1/0/1), `aoi_mask`, `s1_valid_mask`. |
| `spatial_split/` | `spatial_split.mat` with `train_pool`, `labeled_train`, `unlabeled_train`, `validation`, `test`, `buffer_mask`, `valid_mask`; GeoTIFF, preview and statistics. |
| `<method>/checkpoints/<run_name>/` | Per-epoch weights, normalization/configuration, training logs, initial input manifest and completed-run provenance. |
| `unsupervised/baselines/<run_name>/` | Global and pixelwise baseline MATs for each epoch and baseline provenance. |
| `<method>/predictions/<run_name>/` | Full-scene probability MATs and prediction hashes. All probability maps use `prob`; AE additionally records baseline mode and epoch metadata. |
| `<method>/results/<run_name>/` | Validation search, selected configuration, held-out metrics, final prediction, full-AOI metrics and confusion maps. AE freeze: `frozen_abs_configuration.json`; test: `abs_final_metrics.json`. |

Training refuses to overwrite a run directory. Ground truth and split refuse nonempty output directories because they are shared by every method/run under that `output_root`. Choose a new output root for a new preprocessing experiment. Input, split, ground-truth, checkpoint, baseline and prediction checks detect important mismatches; they are consistency checks, not a security boundary against deliberate manifest edits. Keep complete run folders together. Paths in generated private run metadata may be absolute; generated outputs are excluded from this source distribution and should be reviewed before sharing.

For an interrupted run, use the same config and original run directory:

```bash
python train_supervised_bitemporal.py --config config.json --resume outputs/supervised_bitemporal/checkpoints/experiment_01/supervised_last.pt
```

Last-checkpoint names are `supervised_last.pt`, `mean_teacher_last.pt` and `ae_last.pt` for the corresponding methods. Resume validates original inputs/configuration and checkpoint ownership. It does not restore every RNG state and is not bit-for-bit equivalent to uninterrupted training. Completed-run provenance is written only when training finishes; inference of a partial run is rejected. Do not change data, total epochs or settings to resume. If a failure leaves partial preprocessing/results, preserve and inspect them before manually archiving that incomplete directory and retrying; never use test outcomes to guide a rerun or a new search.

## Adaptations and verification

The latest supplied five evaluators were used, including their full-AOI metrics/confusion maps and ABS-compatible AE evaluation. Older evaluator variants were not restored. CNN architectures, sampling, losses, EMA and numerical defaults remain based on the supplied scripts. Changes cover configurable paths/date counts, consistent run selection, diagnostics, provenance checks, overwrite/frozen-selection guards, GeoTIFF block profiles and runtime compatibility. Ground-truth generation is a larger rewrite: explicit vector labels/AOI, dynamic CRS and reference checks replace case-specific inputs. It needs visual checking on each real dataset and uses full-array memory.

See `VERIFICATION.md` for the checks actually performed and their limits. No thesis scores are reproduced or claimed here. Real-data end-to-end validation, realistic resource sizing and GPU execution remain necessary before using this package for reported scientific results. These scripts are command-line entry points, not a supported importable API. No author, contact or license has been inferred; choose an appropriate license before public distribution.
