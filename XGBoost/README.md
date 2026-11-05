# XGBoost input ablations: Ddc / full Yb / imaginary Yb / real Yb

Updated version of your gpu_xgboost_vmag.py and Case30.ipynb for the prepared v2 datasets. The original Downloads files remain unchanged. Keep all package files together on your GPU server.

## Data and environment

Use `dl_normalized_datasets/case30`, `case39`, or `case57`, schema `ddc_yb_train_normalized_v2`. Each folder contains dataset_train.npz, dataset_val.npz, dataset_test.npz. X has shape (N,3,B,B); y is standardized raw DQ; Vmag is standardized bus voltage. Raw target arrays and training-only statistics are read for reporting. No normalization is refitted.

Requires Python 3.10+, NumPy, and a GPU-capable XGBoost >=2.0 build compatible with your server's driver. Keep a working CUDA installation instead of blindly replacing it with a newer wheel. CPU checks used XGBoost 3.3.0. Jupyter is optional. The code does not require PyTorch, CuPy, joblib, or tqdm; XGBoost installs its own required runtime dependencies.

GPU configuration uses `tree_method=hist`, `device=cuda`, and QuantileDMatrix, as in the [official XGBoost GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/). A tiny startup probe verifies that requested CUDA training has not silently fallen back to CPU. Use `--device cpu` for CPU checks.

## Run your four ablations

From this code folder:

```bash
python -u gpu_xgboost_vmag.py \
  --data-dir /path/to/dl_normalized_datasets/case30 \
  --out-dir results_case30 \
  --device cuda --seeds 42 123 2026
```

All four variants run by default:

| Scenario | Variant | Selected normalized channels |
|---|---|---|
| Ddc -> DQ | ddc | 0 |
| Ddc + full Yb -> DQ | ddc_yb | 0,1,2 |
| Ddc + Im(Yb) -> DQ | ddc_imag | 0,2 |
| Ddc + Re(Yb) -> DQ | ddc_real | 0,1 |

For one scenario add, for example, `--variants ddc_yb`. Change input and output directories for case39 or case57. Optional `--case 30` checks the bus count; otherwise it is read from the arrays. To predict Vmag instead, add `--target vmag` and use a separate output directory.

Quick GPU execution check:

```bash
python gpu_xgboost_vmag.py --data-dir /path/to/dl_normalized_datasets/case30 \
  --out-dir smoke_case30 --device cuda --smoke
```

Smoke mode uses 64 train / 32 validation / 32 test samples, only 12 evenly spaced output entries, five boosting rounds, depth 2, and the first seed. Its metrics describe those selected entries, not the whole DQ matrix. They are not performance estimates. Use a fresh output directory for full training.

## Model strategy and expected workload

Preserves your independent-output strategy: each DQ matrix entry (or each Vmag bus) gets an independent scalar regressor. Inputs are the selected full matrices flattened per sample. This is not a row-by-row local-feature approximation and not an experimental multi-output-tree algorithm.

| Case | Scalar outputs per DQ trial | Up to models for 4 variants × 3 seeds |
|---|---:|---:|
| 30 | 900 | 10800 |
| 39 | 1521 | 18252 |
| 57 | 3249 | 38988 |

Full runs can take substantial time and disk space. Outputs that are exactly constant in training are represented by their training constant rather than a booster. Start with smoke mode, then `--seeds 42` for a single-seed full baseline if desired. Full Vmag runs require only B outputs per trial.

Outputs train sequentially on one selected GPU. The training and validation QuantileDMatrix objects are built once per variant/seed and reused across output fits. Validation quantiles reference the training matrix. Replacing labels does not refit cuts using targets. This avoids four concurrent GPU model fits and unnecessary copies from the original gpu_workers approach. `--threads` controls host/native CPU threads, not the number of simultaneous GPU models.

Each variant has a separate directory and checkpoint metadata. Excluded channels are actually omitted from the feature matrix; they cannot affect predictions. Unlike the neural networks, trees do not need a fixed three-slot input to hold neural parameter counts equal. Identical tree hyperparameters, output targets, split assignments and per-output seeds are used across variants. Fitted tree structures and selected numbers of trees may differ. The initial base score for each output is its training mean, identical across variants.

Default colsample_bytree is 1.0, so every available selected feature is eligible and additional channels do not change a random column-subsampling budget. This differs from your original 0.8. Subsample remains 0.8; an output's seed is deterministically derived from the requested run seed and output index. Use the same settings for all variants. XGBoost need not share a neural-network initialization or parameter count to form this input ablation.

## Training and validation

Defaults: at most 600 boosting rounds per scalar output, max_depth=6, learning_rate=0.05, reg_alpha=0, reg_lambda=1, max_bin=256, host threads=4, and early stopping after 20 rounds without validation RMSE improvement. There is no hidden per-case hyperparameter table. You can choose another depth/round budget on validation data, keeping it fixed across that case's input ablations.

The objective is squared error on standardized targets. Validation selects boosting rounds independently for each scalar output. Saved boosters are sliced to the chosen best iteration, following the [XGBoost guidance on early stopping](https://xgboost.readthedocs.io/en/stable/prediction.html#early-stopping); prediction does not accidentally use the later, worse trees. Test data is loaded only after all requested variants/seeds have finished model selection.

The input transforms are the same train-fitted transforms as for the neural-network experiments. Tree models generally do not need zero-mean/unit-variance inputs, but retaining the supplied representation keeps dataset handling consistent. No second scaler, sigmoid, or clipping is applied.

## Metrics: original units AND standardized targets

Each trial reports two clearly labelled sets:

- Primary `r2`, `rmse`, `mae`: inverse-transformed predictions against DQ_raw or Vmag_raw. These are comparable with the updated U-Net/transformer package on the same case/split/target.
- Secondary `standardized`: metrics directly against stored standardized y or Vmag.

For DQ, `RMSE_raw = RMSE_standardized * dq_scale`; R2 is unchanged apart from small floating-point target rounding. For Vmag, scales differ by bus, so pooled standardized and raw R2 can differ. Both RMSE types can exceed 1; standardization is not a [0,1] bound.

Metrics pool all sample/output entries rather than averaging R2 over outputs or chunks. Vmag per-output R2 is saved. When a complete DQ matrix is predicted, derived row/column-sum metrics are computed after inversion to raw DQ units. The older per-sample-unit-max DQ pattern task is a different target definition from standardized raw DQ.

## Checkpoints, outputs and resume

```text
results_case30/
  protocol.json
  summary.json
  comparison.csv
  ddc/seed_42/
    metadata.json
    state.json
    record_00001.json
    history_00001.json
    output_00001.ubj
    ...one UBJ for each nonconstant output...
    results.json
    validation_predictions.npz
    test_predictions.npz
  ...other variant/seed folders...
```

metadata.json records the target, feature selection, preprocessing, data hashes, XGBoost version and settings. Each completed output commits its own atomic record JSON; the state.json summary is refreshed at progress intervals. Boosting histories are separate history JSON files, so saving each new output does not repeatedly rewrite every previous boosting history. Completed records remain recoverable even if the summary write is interrupted. UBJ is XGBoost's native model format; Python model pickles are not used.

comparison.csv contains all per-seed raw and standardized train/validation/test metrics. summary.json adds mean and sample standard deviation across seeds. It does not select a seed by test score.

Predictions are saved with `prediction`, `target`, `prediction_standardized`, `target_standardized`, `source_indices`, and `output_indices`. Full DQ predictions have shape (N,1,B,B), Vmag has shape (N,B). Smoke predictions have shape (N,K), where K=12 selected outputs; inspect `complete_target` and output_indices. No missing smoke entries are filled with misleading zero predictions.

Repeat the identical command with `--resume` to recover. Completed outputs are retained and checked; an interrupted output is retrained from its beginning. Recovery is per output, not per boosting round. Keep data, target, seed, XGBoost version, device, thread count, and hyperparameters fixed. Use a fresh output directory for a different experiment. A crash after saving a UBJ but before committing its state entry is safely handled by retraining that output. Checkpoints are retained after training, without automatic cleanup.

The notebook Case30.ipynb uses the same runner. Set CASE to 30/39/57, DATA_ROOT to the normalized data parent, and CODE_DIR to this code folder. It prints the estimated number of scalar models before training. Device strings cuda and cuda:N are supported. Native CUDA training is required when CUDA is requested; no silent CPU fallback is accepted.

Original aliases `--target_key`, `--n_estimators`, `--max_depth`, `--learning_rate`, `--colsample_bytree`, `--reg_alpha`, `--reg_lambda`, and `--early_stopping_rounds` are accepted. Replace separate train/val/test path flags with `--data-dir`. Original gpu_workers, chunk_size, model pickle path and old checkpoint directories are not used by this runner.

## Inference

From a normalized NPZ, without reading target arrays:

```bash
python predict_ablation.py --model-dir results_case30/ddc_yb/seed_42 \
  --input /path/to/dl_normalized_datasets/case30/dataset_test.npz \
  --output predictions.npz --device cuda
```

From raw physical inputs:

```python
from predict_ablation import predict_raw
pred = predict_raw('results_case30/ddc_yb/seed_42', Ddc_raw, Yb_complex,
                   device='cuda')
```

Pass (N,B,B) or (B,B) arrays with the training bus order/units. Transpose MATLAB (B,B,N) first. Ddc-only needs no Yb. Partial variants accept full complex Yb or a real array containing only the required component. Preprocessing and inverse target transforms use saved training statistics; no ground-truth DQ/Vmag values or unknown target scale are required. Prediction can run on CPU using --device cpu even for models trained on CUDA.

## Verification

Local CPU checks passed all four variants on cases 30, 39, and 57 using smoke subsets and selected outputs. Checks cover strict channel selection, raw-input inference, prediction from NPZ files without labels, constant-output handling, best-iteration slicing, complete synthetic matrix/vector shapes, and interrupted-output resumption matching uninterrupted predictions exactly. The notebook was also executed in CPU smoke mode. See verification.json.

CUDA execution was not available locally; run the GPU smoke command on your server. This package contains source/notebook/verification only. It contains no data or pretrained accuracy results.
