# Triangle/transformer input ablations with Im(Yb) pseudoinverse

## Added variant: `ddc_imag_pinv`

The fifth input setting is **Ddc + Moore–Penrose pseudo-inverse of raw Im(Yb)**. It replaces the imaginary-channel slot only for this variant. It does not add Re(Yb), invert the complex Yb, invert normalized features, or perform elementwise reciprocals.

```python
VARIANTS = ['ddc', 'ddc_yb', 'ddc_imag', 'ddc_real', 'ddc_imag_pinv']
```

Run only the new input, using your triangle/residual-gate model:

```bash
python -u train_transformer_imlive.py \
  --data-dir /path/to/dl_normalized_datasets/case30 \
  --out-dir results_case30_tri_pinv \
  --models tri --head residual_gate --variants ddc_imag_pinv \
  --device cuda --seeds 42 123 2026 --pinv-rcond 1e-8
```

For all five inputs, omit `--variants`. The notebook already includes the fifth variant and exposes `PINV_RCOND`. Use a fresh output directory for a new experiment. Original Downloads files and dataset NPZs are unchanged. This complete package bundles the previously missing dataset.py, metrics.py, transformer_model.py, normalization.py and prediction helper alongside your supplied six files.

### Pseudoinverse definition and scaling

For each scenario independently, `P = np.linalg.pinv(raw_ImYb, rcond=1e-8, hermitian=False)` is computed in float64 on CPU. This uses general SVD; no symmetry or invertibility is assumed. Singular values at or below `rcond * largest_singular_value` are discarded, per the [NumPy pinv definition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html). Change the cutoff with `--pinv-rcond`; choose any tuning using validation rather than test data.

The raw imaginary matrix is read from `X_raw[:,2]` in the v2 NPZ files. The inverse of a centered/asinh/standardized matrix is a different quantity and is deliberately not used.

For training pseudoinverses only, fit `s = median(abs(nonzero P))`, then the global mean and standard deviation of `asinh(P/s)`. The new feature is `(asinh(P/s)-mean)/std`, stored as float32 in slot 2. Near-zero/empty scale guards are included and no clipping is applied. Validation/test/inference reuse those same training coefficients. The existing Ddc normalization, all targets, the other four input representations, and the sample split assignments stay unchanged.

The three-slot model capacity is unchanged: the new setting activates slots 0 and 2 and masks slot 1. Triangle updates, residual-gate logic, and baseline evaluation are preserved. The preprocessing is deterministic feature engineering of Im(Yb), not an additional measured signal or an accuracy guarantee.

Pseudoinverses are precomputed once per split/run and reused across model seeds and architectures, rather than recomputed every batch. `--pinv-chunk-size` (default 64) controls CPU SVD batching; it does not alter the mathematical cutoff. Full runs fit coefficients on all training records; smoke runs fit on the documented smoke subset. Derived arrays occupy additional CPU memory; they are not written back into your datasets.

`pinv_preprocessing.json` in the output root and `metadata['input_transform']` in every new-variant checkpoint store the cutoff, signed asinh scale, train center/std, base-normalization signature, training count and training-index hash. Resume checks these parameters. Existing checkpoints remain usable for inference; start fresh for new training because the metadata contract now records input transformations.

### Inference for the new variant

```python
from predict_ablation import predict_raw
pred = predict_raw('results_case30_tri_pinv/tri/ddc_imag_pinv/seed_42/best.pt',
                   Ddc_raw, ImYb_raw, device='cuda')
# Passing full complex Yb instead of ImYb_raw is also supported.
```

Inputs use (N,B,B) or (B,B) layout. Saved preprocessing reconstructs the feature and returns DQ in original units (or Vmag p.u. for a plain vector-head run). No target values are needed. The NPZ prediction CLI also supports this variant, but requires X_raw in addition to the original normalized X so raw Im(Yb) is available exactly. Labels are not read. Do not feed already pseudoinverted data as if it were raw Im(Yb), and do not manually overwrite the original channel-2 normalization parameters.

### Checks completed for this addition

- Analytic pseudo-inverse checks for a singular Laplacian, a zero matrix, a nonsymmetric rank-one matrix and an explicit SVD cutoff.
- Train-only feature scaling, unchanged base arrays/targets, and target independence.
- Your supplied triangle tests, expanded automatically to all five variants, including both heads, masked gradients, paired parameter counts/initialization and a short batch-fit check.
- All five settings trained with axial and triangle residual-gate models in a two-epoch case30 CPU smoke check. The new variant also trained on case39 and case57 smoke subsets.
- Raw-component, complex-Yb and label-free NPZ inference agree with saved predictions.

See `pinv_verification.json`. CUDA was not tested locally. Smoke scores are not accuracy estimates. Additional test command:

```bash
python test_pinv_variant.py --data-dir /path/to/dl_normalized_datasets/case30
```


Updated copies of your attached transformer_model.py, dataset.py, metrics.py, train_transformer_imlive.py, and case30.ipynb. Original Downloads files are unchanged. Keep all files together in this extracted folder to avoid importing old modules.

## Data and setup

Use the previously prepared **dl_normalized_datasets** v2 case30, case39, or case57 files. Each directory contains dataset_train.npz, dataset_val.npz, dataset_test.npz. The runner reads stored training normalization directly from NPZ metadata; it never refits it. Older samples-last or raw-Yb packages are incompatible.

Requires Python 3.10+, NumPy, and CUDA-enabled PyTorch 2.5+ appropriate for your GPU server. Jupyter is optional for the notebook. Training and inference need no SciPy, pandas, or scikit-learn. Keep your server's existing CUDA-enabled PyTorch build.

## Run the five ablations

From this code folder:

```bash
python -u train_transformer_imlive.py \
  --data-dir /path/to/dl_normalized_datasets/case30 \
  --out-dir results_case30 \
  --models axial --device cuda \
  --seeds 42 123 2026 \
  --epochs 200 --batch-size 16
```

All five variants run automatically:

| Scenario | Variant | Input slots |
|---|---|---|
| Ddc -> DQ | ddc | 0 |
| Ddc + full Yb -> DQ | ddc_yb | 0,1,2 |
| Ddc + Im(Yb) -> DQ | ddc_imag | 0,2 |
| Ddc + Re(Yb) -> DQ | ddc_real | 0,1 |
| Ddc + pinv(Im(Yb)) -> DQ | ddc_imag_pinv | 0,2 (derived feature) |

Use case39 or case57 in the input path and a separate output directory for those cases. Bus count is read from the dataset; optional `--case 39` checks that it matches. Default target is DQ. One architecture × five variants × three seeds trains 15 models per case. For one scenario add, for example, `--variants ddc_yb`.

The global-attention architecture is also available as `--models dq`; to run both use `--models axial dq`. The name `dq` here selects the original global Transformer architecture, not the target. Global attention uses N² tokens and attention complexity grows as N⁴; use a small batch (e.g. `--batch-size 2`) if memory is tight. Axial is the default from your notebook.

Quick execution check in a separate directory:

```bash
python train_transformer_imlive.py --data-dir /path/to/dl_normalized_datasets/case30 \
  --models axial --device cuda --smoke --out-dir smoke_case30
```

Smoke uses 16 train / 8 val / 8 test samples, two epochs, batch 2, width 16, depth 1, four heads, and only the first seed. Smoke scores are not performance estimates. Use a fresh output directory for full training.

## Architecture and code changes

- Preserves your AxialTransformerDQNet row/column attention and DQTransformer global attention architecture, learned row/column embeddings, feedforward blocks, pre-normalization, and dropout. The main model is not replaced with the earlier bus-token transformer.
- The scalar embedding now accepts configurable input channels. The runner uses a fixed three-slot embedding for every scenario. An input mask sets excluded slots to zero AFTER the dataset's normalization. Within each architecture/seed, all five scenarios have the same parameter count, identical initial trainable weights, and paired shuffled sample order. Starting parameter hashes are checked.
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

## Triangle model and residual-gate head (added)

New architecture `--models tri` (`triangle_model.py`): each block applies an outgoing and an incoming
triangle multiplicative update (cell (i,j) mixed with (i,k),(j,k) and (k,i),(k,j) for every bus k), then
the existing axial row/column attention and feed-forward. Inputs also include the transposed matrix and
per-bus row/column means. Vmag uses the same backbone with the row-pooling head.

New output head `--head residual_gate` (DQ only, works with axial, dq and tri):
`DQ_hat = where(Ddc != 0, a*Ddc + b + network, 0)` in standardized units. On the supplied case30 data DQ is
exactly zero wherever Ddc is exactly zero, so the gate makes those entries exact. `a`, `b` are learned scalars
initialised from a train-only linear fit, and the last network layer starts at zero, so training starts at the
gated linear baseline (about R2 0.87 on val/test). The runner checks the zero assumption on the training split
and stops with an error if a dataset violates it (then use `--head plain`). The gate uses a tolerance of
1e-6 standardized units, because float32 normalization maps raw Ddc below ~3e-7 onto the same value as zero.

Example:

```bash
python -u train_transformer_imlive.py --data-dir /path/to/case30 --out-dir results_case30_tri_gate \
  --models axial tri --head residual_gate --device cuda --seeds 42 123 2026
```

Use a separate output directory per `--head` value. Results now also contain `nz_r2/nz_rmse/nz_mae`
(metrics on entries where Ddc != 0 only), and `summary.json` has `reference_baselines` (DQ = Ddc and a gated
linear fit of Ddc). A model is only informative if it beats these. `test_tri_model.py` runs CPU checks
(shapes, gradient exclusion, gate exactness, paired initialisation, a few training steps). The supplied triangle tests and new pseudoinverse checks were run locally on CPU. Run a GPU smoke test before full training.

## Design proposal and how it was implemented

The two additions above follow a design proposal made after inspecting the supplied case30 data
(DQ is close to Ddc, has the same zero pattern, and is asymmetric). The original proposal is reproduced
here, followed by where the implementation matches it and where it deliberately differs.

### Original proposal (verbatim summary)

**Stage 1: residual + zero-gate on the existing axial model.** The network only learns the correction on
top of a strong baseline:

    DQ_hat(i,j) = 1[Ddc(i,j) > 0] * ( a*Ddc(i,j) + b + f(Z)(i,j) )

- `a` and `b` are learned scalars, initialised near the linear fit.
- The gate makes the ~45% structural zeros exact, so no capacity is spent on them.
- The network starts at R2 of about 0.87 instead of 0, and every variant (ddc, ddc_yb, ddc_imag, ddc_real)
  gets the same skip and gate, so the ablation stays fair.
- The gate must be computed from raw Ddc, or from the normalized constant for zero
  (`-input_center/input_scale`, about -0.5075), because zero is not 0 after standardization.

**Stage 2: bus-pair network with triangle updates.** DQ(i,j) most likely depends on paths through
intermediate buses k, which row/column attention only approximates. A triangle multiplicative update
(from AlphaFold's Evoformer) expresses it directly:

    Z(i,j) <- Z(i,j) + W * LN( sum_k sigmoid(A*Z(i,k)) * (B*Z(j,k)) )

It costs O(N^3 * d), which is cheap for 30, 39 and 57 buses. Proposed block: axial row/column attention,
a triangle update with both outgoing (Z(i,k), Z(j,k)) and incoming (Z(k,i), Z(k,j)) paths (needed because
DQ is asymmetric), a node stream (one token per bus) that attends over buses with a learned bias from Z and
feeds back into Z, and a feed-forward layer. Initial Z embeds Ddc(i,j), Ddc(j,i), Re Yb(i,j), Im Yb(i,j)
masked per variant, plus row and column sums of Ddc as node features (closeness/betweenness-like summaries).

### Implementation status

| Proposal item | Status | Notes |
|---|---|---|
| Residual skip `a*Ddc + b` with learned scalars | Implemented | Works in standardized units; `a`, `b` are fitted on the training split at run start (case30: about 0.923 and 0.019), not hard-coded. The proposal's example values were approximate. |
| Zero gate | Implemented, with a changed tolerance | Ddc is continuous and DQ is nonzero for small positive Ddc, so the gate is `abs(x - zero_constant) > 1e-6` in standardized units rather than a loose threshold. A tolerance of 1e-3 would wrongly zero about 0.3% of real entries. Float32 normalization maps raw Ddc below about 3e-7 onto the zero value, which is why exactly 0 cannot be tested. |
| Same skip and gate for all four variants | Implemented | The gate reads slot 0 (Ddc), which every variant keeps. Initial weights and parameter counts remain paired. |
| Network starts at baseline | Implemented | The last layer is zero-initialised. Expected step-0 score on case30: R2 0.872 (val) and 0.869 (test); 0.836 on the Ddc-nonzero entries only. |
| Triangle update, outgoing and incoming | Implemented | Standard gated form (separate sigmoid gate, LayerNorm on the mixed result, output gate) rather than the single-line formula; the sum over k is computed in float32 even under AMP. |
| Block order | Differs | Implemented order is outgoing update, incoming update, then axial attention with feed-forward, not axial first. |
| Initial pair embedding | Differs slightly | Embeds `[x, x^T]` for all three slots (masked per variant), so Ddc(i,j) and Ddc(j,i) are both visible; the transposes of Re/Im Yb are redundant if Yb is symmetric but harmless. |
| Row/column summaries of Ddc | Implemented as means | Per-bus row and column means of each (masked) channel are added to every pair token. The proposal said sums; means differ only by a constant factor. They are a simple proxy for closeness, not an exact closeness or betweenness centrality. |
| Node stream with attention over buses | **Not implemented** | Left out to keep the model simple and because it could not be tested here. Only the row/column mean features are used. Adding it is the most natural next extension if the triangle model underfits. |
| Physics feature (pseudo-inverse of Im Yb) | **Not implemented** | Mentioned as optional in the proposal; it applies only to the Yb variants. |

### Caveats

- The proposal's claim that the network starts at R2 of about 0.87 is the baseline performance, not a trained
  result. Only models that beat `reference_baselines` in `summary.json` have learned anything beyond Ddc.
- Whether the triangle updates help is an empirical question; compare `tri` against `axial` under the same
  `--head`, preferably on the `nz_*` metrics.
- The gate relies on DQ being exactly zero where Ddc is exactly zero. This was verified on the supplied case30
  files only. The runner checks it on the training split and stops with an error if it fails (use `--head plain`).
