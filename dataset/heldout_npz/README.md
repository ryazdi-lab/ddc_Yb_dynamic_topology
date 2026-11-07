# External held-out datasets

Each `case30`, `case39`, and `case57` folder contains `dataset_heldout.npz` with all 10 samples from its respective `IEEE*_Ddc_Yb_Dq_10cases.mat` file.

| Case | X shape | y shape |
| --- | --- | --- |
| 30 | (10, 3, 30, 30) | (10, 1, 30, 30) |
| 39 | (10, 3, 39, 39) | (10, 1, 39, 39) |
| 57 | (10, 3, 57, 57) | (10, 1, 57, 57) |

`X` contains float32 channels `[Ddc, Re(Yb), Im(Yb)]`. `y` contains float32 standardized DQ. Normalization uses the existing training statistics from `dl_normalized_datasets/case*/normalization.json`. The statistics were copied unchanged and applied without fitting on held-out samples.

Raw float64 matrices are preserved in `X_raw` and `DQ_raw`. Additional keys `Ddc`, `DQ`, and complex128 `Yb` use the layout `(samples, buses, buses)`. The shared Yb matrix in each MAT file is repeated across its 10 samples. These files consequently represent one shared admittance topology per bus count.

The files follow the existing `ddc_yb_train_normalized_v2` schema for DQ evaluation with U-Net, Attention U-Net, Axial Transformer, and XGBoost. All four input ablations can use the same file. Supply `dataset_heldout.npz` as a separate evaluation input to a trained model of the corresponding bus count and normalization.

The source files supply DQ targets. Vmag targets are unavailable. `source_indices` are local zero-based sample indices within each new MAT file. Identify samples jointly by `dataset_id` and `source_indices` when combining sources.

Each folder includes the frozen `normalization.json` and a `manifest.json` recording source hashes, output hashes, shapes, and conversion checks.

To repeat conversion into a fresh output folder, install NumPy and SciPy and run from this folder:

```powershell
python convert_heldout_mat.py --mat "C:\Users\Admin\Downloads\IEEE57_Ddc_Yb_Dq_10cases.mat" --normalization "case57\normalization.json" --out-dir "case57_reconverted"
```
