# U-Net and Attention U-Net input ablations

Updated copies of your attached models, loader, metrics, and notebook. Your original Downloads files remain unchanged. Use this whole folder on the server, so Python imports these updated modules together.

## Required dataset and environment

Use the **dl_normalized_datasets** package from this conversation, schema `ddc_yb_train_normalized_v2`. Each case folder must contain dataset_train.npz, dataset_val.npz, dataset_test.npz. These files already contain fitted normalization parameters; normalization.json is useful for inspection but the runner reads parameters directly from the NPZ files. Do not use the earlier samples-last three-key or raw-Yb ablation packages.

Requires Python 3.10+, NumPy, and PyTorch 2.5+ installed with CUDA support appropriate for your server. Jupyter is needed only for the notebook. No SciPy, pandas, scikit-learn, or graph libraries are required by the training/inference paths. Keep an existing CUDA-enabled PyTorch installation; this package does not install a different build.

Expected structure, for example:

```text
work/
  unet_ablation_gpu/
    train_ablation.py
    dataset.py
    models.py
    unet_model.py
    attention_unet_model.py
    metrics.py
    normalization.py
    predict_ablation.py
    train_unet_case39.ipynb
  dl_normalized_datasets/
    case30/dataset_{train,val,test}.npz
    case57/dataset_{train,val,test}.npz
```

## Run all four DQ ablations

From inside this code folder, replace paths as needed:

```bash
python -u train_ablation.py \
  --data-dir /path/to/dl_normalized_datasets/case30 \
  --out-dir results/case30 \
  --models unet attention_unet \
  --device cuda --epochs 200 --batch-size 64 \
  --seeds 42 123 2026
```

Repeat for case57 with `--data-dir /path/to/dl_normalized_datasets/case57 --out-dir results/case57`. The command above trains 24 models per case: 2 architectures × 4 input settings × 3 seeds. Use `--models unet` for plain U-Net only or `--models attention_unet` for Attention U-Net only. By default the script trains only plain U-Net, all four variants, three seeds, and DQ targets.

| Scenario | CLI variant | Normalized input slots |
|---|---|---|
| Ddc -> DQ | ddc | 0 |
| Ddc + full Yb -> DQ | ddc_yb | 0,1,2 |
| Ddc + Im(Yb) -> DQ | ddc_imag | 0,2 |
| Ddc + Re(Yb) -> DQ | ddc_real | 0,1 |

All four run by default. For one experiment, add `--variants ddc_yb`, for example.

Quick GPU execution check, using a separate output folder:

```bash
python train_ablation.py --data-dir /path/to/dl_normalized_datasets/case30 \
  --models unet attention_unet --device cuda --smoke --out-dir smoke_case30
```

Smoke mode uses 16 training, 8 validation, and 8 test samples; two epochs; base width 8; batch size 4; and only the first seed. These scores do not estimate final predictive accuracy. For an additional smoke test with full default model width on a larger subset, run without `--smoke` and set your chosen epochs/seeds; smoke is intentionally a small compatibility test.

## What changed

- Both source architectures retain their encoder/decoder structure, convolutions, BatchNorm, and skip connections. Attention U-Net retains CBAM and bottleneck self-attention.
- The first convolution now accepts configurable `in_channels`. The runner uses three fixed input slots for every ablation and masks excluded slots AFTER normalization. This keeps parameter count and starting trainable weights identical across the four variants within each architecture/seed. Plain and attention models have different parameter counts from one another.
- The final matrix sigmoid was removed because standardized DQ is unbounded. Internal attention sigmoid gates remain. No prediction clipping is applied before metrics. The vector output head is also linear at its output.
- The loader reads three-channel NCHW X and both matrix/vector target formats in v2 files. Raw target arrays are used only for evaluation. There is no second normalization fit, no target-derived input, and no full-Yb adjacency supplied to Ddc-only or partial-Yb variants.
- The notebook is a configurable interface to the same CLI runner, so notebook and terminal experiments use the same implementation. Its retained filename contains case39 for continuity; choose case30 or case57 in its configuration.
- Checkpoints include architecture/task/input settings, training normalization, source fingerprints, and paired initialization hashes. Test data is loaded after all requested trials finish validation checkpoint selection.

## Training and metrics

Defaults: base_channels=32, AdamW lr=3e-4, weight decay=1e-4, combined MSE/MAE with alpha=0.5, ReduceLROnPlateau on raw validation RMSE (patience 8, factor 0.5), gradient clipping at 1, CUDA AMP, and early stopping after 30 epochs without validation RMSE improvement.

The loss runs on standardized targets. With `--loss-alpha 1`, training uses pure MSE, which for the globally standardized DQ target is proportional to raw-DQ MSE. Keep loss settings identical across all four input variants. Choose hyperparameters using validation data rather than test data.

Metrics are **pooled over every entry and sample**, accumulated with global error sums and a stable merged target variance. Batch R2 values are not averaged. Predictions are inverse-transformed using the saved training statistics before R2/RMSE/MAE are computed against raw targets. DQ metrics use the source DQ units; Vmag metrics use per unit. DQ is a dependency matrix here, not reactive loss Qloss.

Targets are raw DQ under a global standardization, unlike your original per-sample-unit-max DQ pattern target. Scores from those two tasks are not directly interchangeable. No unknown test-sample DQ maximum is required at inference.

History train metrics are online predictions during parameter updates. Final train/validation/test metrics in comparison.csv are all recomputed using the selected best checkpoint with evaluation mode enabled. These are the values to use for comparisons.

Statistics and sample indices must be consistent across splits. The runner checks the normalization, bus counts, target geometry, and disjoint indices. Validation selects the checkpoint with lowest raw RMSE independently per model/variant/seed. The summary reports mean and sample standard deviation across seeds, without choosing a winning seed using test scores. A training-mean baseline is included.

CUDA kernels may not be bitwise reproducible by default. Use `--deterministic` for stricter reproducibility on a fixed software/hardware stack. Same initialization and sample order do not guarantee identical results across different GPU types.

## Outputs

```text
results/case30/
  protocol.json
  summary.json
  comparison.csv
  unet/ddc/seed_42/
    best.pt
    latest.pt
    history.csv
    results.json
    validation_predictions.npz
    test_predictions.npz
  ...remaining model/variant/seed folders...
```

Saved predictions have keys prediction, target, source_indices. Prediction and target values are in original units, with `(N,1,B,B)` DQ or `(N,B)` Vmag layout. summary.json includes per-seed and aggregate results plus the training-mean baseline. All outputs use unique trial folders to avoid mixing ablations.

To resume, repeat the same command with `--resume`. Keep datasets, epoch budget, batch size, optimization settings, AMP, worker count, and architecture unchanged. Completed or early-stopped trials are re-evaluated; interrupted ones resume from the next epoch. Recovery is at epoch boundaries. Use a fresh `--out-dir` for a different experiment. Do not reuse smoke output folders for full training. If memory is insufficient, start with a smaller `--batch-size`. If AMP causes nonfinite loss, use a fresh run with `--no-amp`.

## Vmag instead of DQ

Add `--target vmag` and use a separate output directory. The vector head predicts one standardized voltage per bus, while metrics are inverse-transformed to p.u. Per-bus final validation/test metrics are also stored. Each run predicts one target type; this is not joint multitask training.

## Inference

Normalized NPZ input (targets are not read):

```bash
python predict_ablation.py \
  --checkpoint results/case30/unet/ddc_yb/seed_42/best.pt \
  --input /path/to/dl_normalized_datasets/case30/dataset_test.npz \
  --output dq_predictions.npz --device cuda
```

For raw physical input matrices:

```python
from predict_ablation import predict_raw
DQ_prediction = predict_raw(
    'results/case30/unet/ddc_yb/seed_42/best.pt',
    Ddc_raw, Yb_complex, device='cuda')
```

Pass raw arrays `(N,B,B)` or `(B,B)`, retaining the training bus order and units. For Ddc-only checkpoints, omit Yb. For partial-Yb checkpoints, pass full complex Yb or only the required real-valued component. The checkpoint controls preprocessing, input masking, target type, and inverse transform. Raw DQ predictions need no ground-truth target maximum. Returned values are raw DQ or Vmag p.u. Inference uses float32 (CUDA training evaluation may use AMP, so small numerical differences are possible). Only load checkpoints you trust.

## Validation performed locally

Both architectures completed two-epoch CPU training/evaluation for every DQ ablation on both 30- and 57-bus data. Optional Vmag training was checked on 57-bus data. Additional checks verified excluded-channel value/gradient invariance, paired initial weights/parameter counts, negative output support, inference matching saved predictions, globally pooled metrics under uneven batching, and interrupted resumption matching uninterrupted CPU weights exactly. See verification.json. CUDA was not available locally and must be checked on your server.

The supplied ZIP includes source, notebook, requirements, and verification results. Smoke weights/data are excluded; this is not a pretrained model or an accuracy claim.
