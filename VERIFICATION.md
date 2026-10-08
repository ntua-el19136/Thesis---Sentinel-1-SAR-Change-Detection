# Verification record

This verifies software interoperability using synthetic inputs. It does not reproduce thesis accuracy, establish real-data usability without preparation, or measure generalization.

The distributed `tests/smoke_pipeline.py` was also executed end to end from a fresh work directory: all 20 subprocess stages succeeded, including all 26 integrity/protocol checks.

## Folder-layout verification (2026-10-08)

The final grouped layout was tested again after moving the 18 entry points into their workflow folders and renaming the preprocessing folder to `ground_truth_and_spatial_split`. All 20 subprocess stages and all 26 integrity/protocol checks passed on fresh synthetic CPU inputs. The harness was launched outside the repository root, so subprocesses also exercised shared-helper discovery independently of the working directory.

Only helper discovery, entry-point paths, test paths and documentation were adapted. An AST comparison against the uploaded archive confirmed that all top-level function and class definitions in the 18 pipeline scripts and two shared helpers were unchanged. README command targets and all four image links were checked. This remains a small synthetic software check, not real-data or GPU validation.

Versions used for this rerun: numpy 2.3.5, scipy 1.17.0, torch 2.6.0+cpu, rasterio 1.5.2, geopandas 1.2.0, fiona 1.10.1, matplotlib 3.10.8.

## Executed checks

- Ground truth generated from synthetic polygons and matching two-band MAT/TIFF: channel correlations 1.0 to floating-point precision; 8,188 valid pixels, 4,096 positive, 4 ignored.
- Buffered spatial split and GeoTIFF export completed: 4,604 training, 1,536 validation, 1,536 test and 512 buffer pixels. Labeled training: 896 pixels (19.46% of training).
- All five training → inference → evaluation chains completed on CPU. AE also completed baseline generation, separate validation selection and frozen test evaluation. Full-AOI metrics and confusion outputs were produced.
- Synthetic inputs: 64 × 128 pixels, three BEFORE acquisitions, one AFTER, two channels. Configuration: one epoch, BASE 2, PATCH/STRIDE 16, training batches 1, two samples per loader. This exercises execution paths with small tensors; it is not evidence of convergence.
- Syntax compilation and formatting completed. Model/loss comparisons against the supplied training sources are summarized below.

## Integrity and protocol checks

- changed SAR data.
- changed checkpoint.
- different valid split file.
- overlapping train/test masks.
- incompatible architecture.
- unknown parameter.
- AFTER among BEFOREs.
- noninteger epoch.
- changed AE baseline.
- changed prediction.
- changed frozen ABS source.
- frozen ABS selection reproduces without reselection.
- repeat held-out ABS evaluation.
- overwrite frozen ABS validation.
- frozen selection guard: supervised_bitemporal.
- held-out-label independence and valid AOI counts: supervised_bitemporal.
- frozen selection guard: supervised_multitemporal.
- held-out-label independence and valid AOI counts: supervised_multitemporal.
- frozen selection guard: semisupervised_bitemporal.
- held-out-label independence and valid AOI counts: semisupervised_bitemporal.
- frozen selection guard: semisupervised_multitemporal.
- held-out-label independence and valid AOI counts: semisupervised_multitemporal.
- held-out-label independence and valid AOI counts: unsupervised.
- all supervised multitemporal candidate patches stay in train_pool.
- same-input resume preflight.
- resume with changed data.

## Source comparison

Architectures and the listed loss/schedule components were compared as Python ASTs, excluding docstrings.

| Training script | Component | AST matches supplied source |
|---|---|---|
| train_supervised_bitemporal.py | SmallUNet | True |
| train_supervised_bitemporal.py | conv_block | True |
| train_supervised_bitemporal.py | masked_bce_with_logits | True |
| train_supervised_bitemporal.py | masked_dice_loss_with_logits | True |
| train_supervised_multitemporal.py | SmallUNet | True |
| train_supervised_multitemporal.py | conv_block | True |
| train_supervised_multitemporal.py | masked_bce_with_logits | True |
| train_supervised_multitemporal.py | masked_dice_loss_with_logits | True |
| train_semisupervised_bitemporal.py | SmallUNet | True |
| train_semisupervised_bitemporal.py | conv_block | True |
| train_semisupervised_bitemporal.py | masked_bce_with_logits | True |
| train_semisupervised_bitemporal.py | masked_dice_loss_with_logits | True |
| train_semisupervised_bitemporal.py | confidence_filtered_mse | True |
| train_semisupervised_bitemporal.py | update_ema | True |
| train_semisupervised_bitemporal.py | rampup | True |
| train_semisupervised_multitemporal.py | SmallUNet | True |
| train_semisupervised_multitemporal.py | conv_block | True |
| train_semisupervised_multitemporal.py | masked_bce_with_logits | True |
| train_semisupervised_multitemporal.py | masked_dice_loss_with_logits | True |
| train_semisupervised_multitemporal.py | confidence_filtered_mse | True |
| train_semisupervised_multitemporal.py | update_ema | True |
| train_semisupervised_multitemporal.py | rampup | True |
| train_unsupervised.py | ConvAE | True |

## Tested environment

Python 3.12.14 on Linux; CPU PyTorch, no CUDA execution. Dependency lower bounds in requirements are installation guidance, not a tested compatibility matrix.

| Dependency | Version actually used |
|---|---|
| numpy | 2.3.5 |
| scipy | 1.17.0 |
| torch | 2.14.1+cpu |
| matplotlib | 3.10.8 |
| PIL | 12.3.0 |
| tqdm | 4.70.1 |
| rasterio | 1.5.2 |
| geopandas | 1.2.0 |
| fiona | 1.10.1 |

## Reproduce with generated synthetic inputs

No generated data or checkpoints are included in this repository. The test utility creates its own polygons, random arrays and config in a new/empty directory:

```bash
python tests/smoke_pipeline.py --work-dir /path/to/new/synthetic-check
```

Use any new writable directory; on Windows supply a suitable local path. The utility forces CPU, runs all preprocessing/model stages, checks integrity and writes stage logs plus `smoke_results.json` under that directory. An existing nonempty directory is refused. To rerun only integrity checks on that synthetic run:

```bash
python tests/verify_outputs.py --config /path/to/new/synthetic-check/config.json
```

The integrity utility temporarily modifies and restores synthetic files to test rejection paths; use it only on disposable synthetic outputs. The two files in `tests/` are developer utilities, in addition to the 20 pipeline/helper Python files.

## Limits

Not tested: real scenes, GPU/mixed precision, multiworker loading, memory/performance at full Sentinel-1 scale, all GIS formats/CRSs, all configuration combinations, or an interrupted multi-epoch resume equivalence. Resume preflight was exercised with matching and changed inputs; RNG state is not restored. Reference matching uses full arrays and correlations, and does not verify co-registration of every acquisition. Fixed spatial/class-balance constraints may be infeasible on another dataset. One-epoch synthetic results must not be reported as scientific performance.

## Documentation and illustration update

The README preprocessing instructions were aligned with Chapter 3 of the thesis (Sections 3.1 and 3.4-3.6), and four author-supplied fire-case PNGs were added under `docs/images/`. These are illustrative historical outputs, not generated outputs of the synthetic checks. No pipeline or test Python files were changed in this documentation update.
