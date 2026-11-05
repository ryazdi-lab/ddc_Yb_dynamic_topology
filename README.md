# Reactive Dependency Estimation under Varying Grid Topologies

This repository accompanies the paper *Topology-Aware Reactive Dependency Estimation from DC Load Flow Data via Deep and Non-Deep Learning*. The study estimates the AC reactive dependency matrix $D_Q$ from DC dependency $D_{dc}$ and bus admittance $Y_{bus}$ on synthetic grids with 30, 39, and 57 buses and varying topologies.

The models include convolutional networks (U-Net and Attention U-Net), an Axial Transformer, and a traditional machine learning baseline (XGBoost).

The input ablation compares four configurations:

| Configuration | Input |
| --- | --- |
| `ddc` | $D_{dc}$ |
| `ddc_yb` | $D_{dc}$ and both components of $Y_{bus}$ |
| `ddc_imag` | $D_{dc}$ and $\operatorname{Im}(Y_{bus})$ |
| `ddc_real` | $D_{dc}$ and $\operatorname{Re}(Y_{bus})$ |

Test performance is reported using $R^2$, RMSE, and MAE over three training seeds.

The **Axial Transformer with full bus admittance (`ddc_yb`)** achieved the highest mean test $R^2$ across all three grid sizes: **0.8822** for 30 buses, **0.8768** for 39 buses, and **0.8674** for 57 buses.

## Citation

If you use this code in your research, please cite the accompanying manuscript.

```bibtex
@unpublished{yazdi_topology_aware,
  author = {Yazdi, Reza and Shahraeini, Mohammad},
  title  = {Topology-Aware Reactive Dependency Estimation from {DC} Load Flow Data via Deep and Non-Deep Learning},
  note   = {Manuscript}
}
```
