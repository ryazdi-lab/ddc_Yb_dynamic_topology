# Transformer input-ablation package

Updated copies of your attached transformer_model.py, dataset.py, metrics.py, train_transformer_imlive.py, and case30.ipynb. Original Downloads files are unchanged. Keep all files together in this extracted folder to avoid importing old modules.

## Data and setup

Use the previously prepared **dl_normalized_datasets** v2 case30, case39, or case57 files. Each directory contains dataset_train.npz, dataset_val.npz, dataset_test.npz. The runner reads stored training normalization directly from NPZ metadata; it never refits it. Older samples-last or raw-Yb packages are incompatible.

Requires Python 3.10+, NumPy, and CUDA-enabled PyTorch 2.5+ appropriate for your GPU server. Jupyter is optional for the notebook. Training and inference need no SciPy, pandas, or scikit-learn. Keep your server's existing CUDA-enabled PyTorch build.

## Run the four ablations

From this code folder:

```bash
python -u train_transformer_imlive.py \
  --data-dir /path/to/dl_normalized_datasets/case30 \
  --out-dir results_case30 \
  --models axial --device cuda \
  --seeds 42 123 2026 \
  --epochs 200 --batch-size 16
```

All four variants run automatically:

| Scenario | Variant | Input slots |
|---|---|---|
| Ddc -> DQ | ddc | 0 |
| Ddc + full Yb -> DQ | ddc_yb | 0,1,2 |
| Ddc + Im(Yb) -> DQ | ddc_imag | 0,2 |
| Ddc + Re(Yb) -> DQ | ddc_real | 0,1 |

Use case39 or case57 in the input path and a separate output directory for those cases. Bus count is read from the dataset; optional `--case 39` checks that it matches. Default target is DQ. One architecture × four variants × three seeds trains 12 models per case. For one scenario add, for example, `--variants ddc_yb`.

The global-attention architecture is also available as `--models dq`; to run both use `--models axial dq`. The name `dq` here selects the original global Transformer architecture, not the target. Global attention uses N² tokens and attention complexity grows as N⁴; use a small batch (e.g. `--batch-size 2`) if memory is tight. Axial is the default from your notebook.

Quick execution check in a separate directory:

```bash
python train_transformer_imlive.py --data-dir /path/to/dl_normalized_datasets/case30 \
  --models axial --device cuda --smoke --out-dir smoke_case30
```

Smoke uses 16 train / 8 val / 8 test samples, two epochs, batch 2, width 16, depth 1, four heads, and only the first seed. Smoke scores are not performance estimates. Use a fresh output directory for full training.

## Architecture and code changes

- Preserves your AxialTransformerDQNet row/column attention and DQTransformer global attention architecture, learned row/column embeddings, feedforward blocks, pre-normalization, and dropout. The main model is not replaced with the earlier bus-token transformer.
- The scalar embedding now accepts configurable input channels. The runner uses a fixed three-slot embedding for every scenario. An input mask sets excluded slots to zero AFTER the dataset's normalization. Within each architecture/seed, all four scenarios have the same parameter count, identical initial trainable weights, and paired shuffled sample order. Starting parameter hashes are checked.
- Both DQ and Vmag output sigmoids are removed. Targets are standardized and unbounded. Predictions are not clipped before evaluation.
- Axial attention computes each pre-attention LayerNorm once and uses `need_weights=False`, avoiding unused attention weight maps. These changes preserve the attention operation while allowing PyTorch's efficient attention implementation where available.
- The shared v2 loader and pooled physical-unit metric implementation follow the updated U-Net package. Raw target arrays never enter the model. No full-Yb adjacency is passed to Ddc-only or partial-Yb variants.
- The legacy auxiliary-centrality class remains in transformer_model.py, but the ablation runner supports axial/global DQ or Vmag prediction. It does not enable the unattached centrality_loss.py training path. Its legacy centrality sums would be in standardized coordinates, not physical units.

The notebook `case30.ipynb` is a configurable interface to the same runner. Set CASE to 30, 39, or 57, DATA_ROOT to the parent directory of the normalized case folders, and CODE_DIR to this code folder. Default MODELS is axial. It executes a subprocess to ensure imports resolve to these updated files.

## Training protocol

Defaults: d_model=96, depth=4, num_heads=6, mlp_ratio=4, dropout=0.1, batch 16, AdamW lr=3e-4 / weight decay=1e-4, combined standardized MSE+MAE alpha=0.5, five-epoch linear warmup followed by cosine decay, gradient clipping at 1, CUDA AMP, 200 epochs maximum, and early-stop patience 30.

`--loss-alpha 1` selects pure MSE. For globally standardized DQ this is proportional to raw-DQ MSE. Keep settings fixed across the input variants. Transformer warmup/cosine is retained; the U-Net package uses a different scheduler. Cross-architecture comparisons should account for any changed optimization settings.

Checkpoint selection uses lowest pooled raw-unit validation RMSE, independently per trial. For DQ under one fixed global affine scale, this ranks checkpoints identically to pooled validation R². Test data is loaded only after all requested validation selections finish. Test scores do not choose a seed or checkpoint.

Epoch history includes normalized loss and physical-unit R²/RMSE/MAE. Metrics aggregate all samples and entries using error sums and merged variance; they do not average batch R². Final train/validation/test metrics are recomputed using the best checkpoint in evaluation mode. Online train-history metrics reflect changing weights during the epoch. Raw-DQ targets differ from the older per-sample-unit-max DQ pattern task; compare the same target definition across models.

Each trial is paired by initialization and sample order. `--deterministic` requests deterministic algorithms on a fixed stack; it does not guarantee identical results across different hardware. CUDA deterministic attention support depends on the installed PyTorch/GPU backend.

The background heartbeat prints every 120 seconds; change with `--heartbeat-interval` (0 disables it). It is a console progress feature, not a scheduled task.

## Results and recovery

```text
results_case30/
  protocol.json
  comparison.csv
  summary.json
  axial/ddc/seed_42/
    best.pt
    latest.pt
    history.csv
    results.json
    validation_predictions.npz
    test_predictions.npz
  ...other input/seed/model folders...
```

comparison.csv contains per-trial raw-unit train/validation/test metrics. summary.json adds mean and sample standard deviation across seeds and a training-mean baseline. Saved prediction files have `prediction`, `target`, and `source_indices`, all in original DQ units or Vmag p.u. Shape: (N,1,B,B) for DQ or (N,B) for Vmag.

Resume using the same command plus `--resume`. Settings, data, normalization, batch size, epoch budget, worker count, and AMP must remain unchanged; dataset fingerprints are checked. latest.pt restores model, optimizer, scheduler, scaler and RNG states at the next epoch. Completed trials are re-evaluated. Mid-epoch recovery is not implemented in this updated runner; unlike the original partial-epoch checkpoint logic, samples are not silently replayed against partly updated weights.

Use a fresh output directory for changed settings. Reduce batch size for a new run if memory is insufficient. For nonfinite AMP loss, start fresh with `--no-amp`. Original `--model`, `--target_key y/Vmag`, `--batch_size`, `--d_model`, `--num_heads`, `--warmup_epochs`, `--heartbeat_interval`, and `--no_amp` spellings are accepted as aliases. Replace the original separate `--train/--val/--test` command with `--data-dir`. Old `--checkpoint`, `--resume_checkpoint`, `--checkpoint_every_n_batches`, and `--compile` flags are not used; trial output paths are managed by the runner.

## Vmag target

Add `--target vmag` and use a separate output directory. The original vector-head variants use row pooling and a linear head. Metrics are inverse-transformed to p.u.; per-bus final metrics are saved too. A run predicts one target type, not both jointly.

## Inference

From a normalized input NPZ (targets are not read):

```bash
python predict_ablation.py \
  --checkpoint results_case30/axial/ddc_yb/seed_42/best.pt \
  --input /path/to/dl_normalized_datasets/case30/dataset_test.npz \
  --output predictions.npz --device cuda --batch-size 16
```

From raw physical matrices:

```python
from predict_ablation import predict_raw
pred = predict_raw('results_case30/axial/ddc_yb/seed_42/best.pt',
                   Ddc_raw, Yb_complex, device='cuda', batch_size=16)
```

Use (N,B,B) or (B,B) input arrays with the training units and bus order; transpose MATLAB (B,B,N) first. Ddc-only needs no Yb. Partial variants accept a real array of only the required component or full complex Yb. Saved statistics preprocess inputs and inverse-transform predictions automatically; no DQ/Vmag target or sample-specific target maximum is needed. Inference uses float32, so CUDA AMP evaluation can have small numerical differences. Only load trusted checkpoints.

## Local checks

Both axial and global models completed two-epoch CPU training/evaluation for all four DQ variants on cases 30, 39, and 57. Checks also cover strict channel value/gradient exclusion, paired weights/parameter counts, negative output support, raw-input inference matching saved predictions, global metrics with uneven batches, and interrupted resumption matching uninterrupted CPU weights. Optional Vmag training was checked on case30. See verification.json.

CUDA execution was not available locally. This source package includes no data, trained weights, or performance claims. Upload the prepared normalized datasets separately.
