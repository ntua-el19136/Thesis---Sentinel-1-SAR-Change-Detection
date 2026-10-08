# Supported parameter overrides

Keys are case-sensitive. Put settings in `parameters.shared` or a named group below. Method/group values override shared values. A key is applied only where that stage declares it; it may be shared with other stages without being an error. Do not place path, epoch, channel-count or fixed ABS-score settings in these groups. The defaults shown are the source-script defaults before config overrides.

For example, `"parameters": {"shared": {"PATCH": 256, "STRIDE": 128}, "supervised_bitemporal": {"BATCH": 4}}`. The complete example is `config.example.json`.

Use JSON arrays for `SIGMAS` and `THRESHOLDS`, JSON booleans for flags, integers for counts and finite numbers for scalar settings. A grid must be nonempty; thresholds must be strictly between 0 and 1, sigmas nonnegative. Set all search options before test inspection.

## semisupervised_bitemporal

| Key | Default by applicable stage |
|---|---|
| `BASE` | `32` (infer, train) |
| `BATCH_L` | `4` (train) |
| `BATCH_U` | `8` (train) |
| `CONF_HIGH_END` | `0.85` (train) |
| `CONF_HIGH_START` | `0.6` (train) |
| `CONF_LOW_END` | `0.15` (train) |
| `CONF_LOW_START` | `0.4` (train) |
| `EMA_DECAY` | `0.99` (train) |
| `INFERENCE_BATCH` | `12` (infer) |
| `LABELED_SAMPLES_PER_EPOCH` | `2000` (train) |
| `LAMBDA_MAX` | `1.0` (train) |
| `LR` | `0.0003` (train) |
| `MIN_LABELED_PIXELS_IN_PATCH` | `1` (train) |
| `MIN_UNLABELED_PIXELS_IN_PATCH` | `1` (train) |
| `NOISE_STD` | `0.03` (train) |
| `NUM_WORKERS` | `0` (train) |
| `PATCH` | `256` (infer, train) |
| `PERSISTENT_WORKERS` | `False` (train) |
| `PREFETCH_FACTOR` | `2` (train) |
| `RAMPUP_EPOCHS` | `10` (train) |
| `SEED` | `42` (train) |
| `SIGMAS` | `[0.0, 1.0, 2.0]` (evaluate) |
| `STRIDE` | `128` (infer) |
| `THRESHOLDS` | `np.linspace(0.01, 0.99, 199, dtype=np.float64)` (evaluate) |
| `UNLABELED_SAMPLES_PER_EPOCH` | `4000` (train) |
| `WEIGHT_DECAY` | `0.0001` (train) |

## supervised_bitemporal

| Key | Default by applicable stage |
|---|---|
| `BASE` | `32` (infer, train) |
| `BATCH` | `12` (train) |
| `LABELED_SAMPLES_PER_EPOCH` | `8000` (train) |
| `LR` | `0.0003` (train) |
| `NUM_WORKERS` | `0` (train) |
| `PATCH` | `256` (infer, train) |
| `PERSISTENT_WORKERS` | `False` (train) |
| `PIN_MEMORY` | `True` (train) |
| `PREFETCH_FACTOR` | `2` (train) |
| `SEED` | `42` (train) |
| `SIGMAS` | `[0.0, 1.0, 2.0]` (evaluate) |
| `STRIDE` | `128` (infer) |
| `THRESHOLDS` | `np.linspace(0.01, 0.99, 199, dtype=np.float64)` (evaluate) |
| `WEIGHT_DECAY` | `0.0001` (train) |

## unsupervised

| Key | Default by applicable stage |
|---|---|
| `BASE` | `32` (build, infer, train) |
| `BATCH` | `12` (train) |
| `CANDIDATE_STRIDE` | `16` (train) |
| `LR` | `0.0003` (train) |
| `NUM_WORKERS` | `6` (train) |
| `PATCH` | `256` (build, infer, train) |
| `PERSISTENT_WORKERS` | `True` (train) |
| `PREFETCH_FACTOR` | `2` (train) |
| `SAMPLES_PER_EPOCH` | `8000` (train) |
| `SAVE_ERROR_MAP` | `False` (infer) |
| `SAVE_RAW_Z` | `False` (infer) |
| `SEED` | `42` (train) |
| `SIGMAS` | `(0.0, 1.0, 2.0)` (evaluate) |
| `STRIDE` | `128` (build, infer) |
| `THRESHOLDS` | `np.linspace(0.01, 0.99, 199)` (evaluate) |

## supervised_multitemporal

| Key | Default by applicable stage |
|---|---|
| `BASE` | `32` (infer, train) |
| `BATCH` | `12` (train) |
| `INFERENCE_BATCH` | `12` (infer) |
| `LABELED_SAMPLES_PER_EPOCH` | `8000` (train) |
| `LR` | `0.0003` (train) |
| `NUM_WORKERS` | `0` (train) |
| `PATCH` | `256` (infer, train) |
| `PERSISTENT_WORKERS` | `False` (train) |
| `PIN_MEMORY` | `True` (train) |
| `PREFETCH_FACTOR` | `2` (train) |
| `SEED` | `42` (train) |
| `SIGMAS` | `[0.0, 1.0, 2.0]` (evaluate) |
| `STRIDE` | `128` (infer) |
| `THRESHOLDS` | `np.linspace(0.01, 0.99, 199, dtype=np.float64)` (evaluate) |
| `WEIGHT_DECAY` | `0.0001` (train) |

## ground_truth

| Key | Default by applicable stage |
|---|---|
| `ALL_TOUCHED` | `False` (create) |
| `MIN_CHANNEL_CORRELATION` | `0.75` (create) |
| `MIN_FULL_CORRELATION` | `0.85` (create) |

## semisupervised_multitemporal

| Key | Default by applicable stage |
|---|---|
| `BASE` | `32` (infer, train) |
| `BATCH_L` | `4` (train) |
| `BATCH_U` | `8` (train) |
| `CONF_HIGH_END` | `0.85` (train) |
| `CONF_HIGH_START` | `0.6` (train) |
| `CONF_LOW_END` | `0.15` (train) |
| `CONF_LOW_START` | `0.4` (train) |
| `EMA_DECAY` | `0.99` (train) |
| `INFERENCE_BATCH` | `12` (infer) |
| `LABELED_SAMPLES_PER_EPOCH` | `2000` (train) |
| `LAMBDA_MAX` | `1.0` (train) |
| `LR` | `0.0003` (train) |
| `MIN_LABELED_PIXELS_IN_PATCH` | `1` (train) |
| `MIN_UNLABELED_PIXELS_IN_PATCH` | `1` (train) |
| `NOISE_STD` | `0.03` (train) |
| `NUM_WORKERS` | `0` (train) |
| `PATCH` | `256` (infer, train) |
| `PERSISTENT_WORKERS` | `False` (train) |
| `PREFETCH_FACTOR` | `2` (train) |
| `RAMPUP_EPOCHS` | `10` (train) |
| `SEED` | `42` (train) |
| `SIGMAS` | `[0.0, 1.0, 2.0]` (evaluate) |
| `STRIDE` | `128` (infer) |
| `THRESHOLDS` | `np.linspace(0.01, 0.99, 199, dtype=np.float64)` (evaluate) |
| `UNLABELED_SAMPLES_PER_EPOCH` | `4000` (train) |
| `WEIGHT_DECAY` | `0.0001` (train) |

## spatial_split

| Key | Default by applicable stage |
|---|---|
| `BUFFER_COST_WEIGHT` | `0.1` (create) |
| `BUFFER_NORMALIZATION` | `0.1` (create) |
| `BUFFER_PIXELS` | `128` (create) |
| `LABELED_TARGET` | `0.2` (create) |
| `LABEL_BLOCK_SIZE` | `256` (create) |
| `LABEL_HASH_A` | `3` (create) |
| `LABEL_HASH_B` | `7` (create) |
| `LABEL_HASH_GROUPS_TO_SELECT` | `2` (create) |
| `LABEL_HASH_MOD` | `10` (create) |
| `LABEL_PREVALENCE_COST_WEIGHT` | `1.0` (create) |
| `LABEL_PREVALENCE_TOL` | `0.05` (create) |
| `LABEL_SIZE_COST_WEIGHT` | `1.0` (create) |
| `LABEL_SIZE_TOL` | `0.02` (create) |
| `MAX_POSITIVE_PREVALENCE_DEVIATION` | `0.1` (create) |
| `MIN_MINORITY_CLASS_FRACTION` | `0.2` (create) |
| `MIN_REGION_WIDTH` | `512` (create) |
| `PATCH_SIZE` | `256` (create) |
| `PREVALENCE_COST_WEIGHT` | `1.0` (create) |
| `RATIO_COST_WEIGHT` | `1.0` (create) |
| `SEARCH_STEP` | `8` (create) |
| `TEST_RATIO_TOL` | `0.025` (create) |
| `TEST_TARGET` | `0.2` (create) |
| `TRAIN_RATIO_TOL` | `0.025` (create) |
| `TRAIN_TARGET` | `0.6` (create) |
| `VAL_RATIO_TOL` | `0.025` (create) |
| `VAL_TARGET` | `0.2` (create) |

`parameters.shared` accepts the union of these names. `AUTO_RESUME` is deliberately unsupported; use the explicit `--resume` CLI option. AE `ERROR_SMOOTH_SIGMA=0` and `USE_HIGH_ERROR=True` are fixed by the ABS-compatible input contract. `MIN_FULL_CORRELATION` is the mean of the two channel correlations. Split `PATCH_SIZE` is saved as metadata; actual training candidate containment is checked using model `PATCH`.
