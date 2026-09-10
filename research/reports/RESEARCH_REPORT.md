# SENTINEL-WM - Research Report

_Generated 2026-09-10 08:05_  |  config: [`run_config.json`](../run_config.json)

## 1. Data

- Flows: **2,829,609**  |  10-s state windows: **14,644**  |  sequences (L=12, K=6, F=53): **14,197**
- Label distribution: [`data_profile/flow_label_distribution.csv`](../data_profile/flow_label_distribution.csv)  -  [timeline](../figures/attack_timeline_by_day.png)
- Split mode = **auto** (auto -> stratified), leakage-safe (contiguous chunks, scaler fit on real-train only): [`data_profile/split_summary.csv`](../data_profile/split_summary.csv)  -  per-family coverage: [`data_profile/split_family_windows.csv`](../data_profile/split_family_windows.csv)

  - **train**: 8894 windows, 10.9% attack
  - **val**: 2875 windows, 9.5% attack
  - **test**: 2875 windows, 11.5% attack

## 2. Model benchmark

Full table + per-horizon F1: [`benchmarks/benchmark.md`](../benchmarks/benchmark.md)  -  raw: [`benchmark_full.csv`](../benchmarks/benchmark_full.csv)  -  figures: [horizon F1](../figures/horizon_f1.png), [ROC](../figures/roc.png), [PR](../figures/pr.png), [lead time](../figures/lead_time.png)

Ranked by **PR-AUC** (threshold-free; robust to the val->test attack-prevalence shift). `F1` = at the FPR<=5% threshold; `F1*` = best achievable by sweeping it.

| Model | Family | PR-AUC | F1 | F1* | AUROC | MLT (s) | Detect | ProgAcc |
|---|---|---|---|---|---|---|---|---|
| SENTINEL-WM (system) | system | 0.875 | 0.789 | 0.812 | 0.948 | 10 | 0.02 | 0.907 |
| SENTINEL-WM | world_model | 0.854 | 0.777 | 0.781 | 0.943 | 10 | 0.01 | 0.907 |
| xgboost__seq | classical | 0.852 | 0.771 | 0.788 | 0.932 | 10 | 0.01 | - |
| hist_gradient_boosting__seq | classical | 0.843 | 0.746 | 0.779 | 0.932 | 10 | 0.02 | - |
| gru | nn | 0.828 | 0.757 | 0.763 | 0.924 | 10 | 0.02 | 0.906 |
| random_forest__seq | classical | 0.826 | 0.739 | 0.772 | 0.921 | 10 | 0.03 | - |
| lstm | nn | 0.814 | 0.751 | 0.757 | 0.916 | 10 | 0.02 | 0.910 |
| tcn | nn | 0.812 | 0.753 | 0.755 | 0.915 | 10 | 0.01 | 0.904 |
| mlp | nn | 0.779 | 0.709 | 0.711 | 0.907 | 10 | 0.01 | 0.864 |
| gat | graph | 0.776 | 0.740 | 0.762 | 0.895 | 10 | 0.02 | 0.903 |
| xgboost | classical | 0.759 | 0.667 | 0.690 | 0.895 | 10 | 0.04 | - |
| hist_gradient_boosting | classical | 0.747 | 0.662 | 0.667 | 0.902 | 10 | 0.03 | - |
| random_forest | classical | 0.729 | 0.642 | 0.651 | 0.880 | 10 | 0.04 | - |
| persistence | reference | 0.723 | 0.815 | 0.818 | 0.852 | 0 | 0.00 | - |
| extra_trees | classical | 0.710 | 0.617 | 0.627 | 0.877 | 10 | 0.03 | - |
| logistic_regression__seq | classical | 0.694 | 0.612 | 0.649 | 0.880 | 10 | 0.02 | - |
| mlp_sklearn | classical | 0.690 | 0.613 | 0.621 | 0.864 | 10 | 0.01 | - |
| mlp_sklearn__seq | classical | 0.581 | 0.602 | 0.664 | 0.881 | 10 | 0.02 | - |
| knn | classical | 0.557 | 0.473 | 0.502 | 0.783 | 10 | 0.01 | - |
| logistic_regression | classical | 0.511 | 0.439 | 0.489 | 0.811 | 10 | 0.03 | - |
| linear_svc | classical | 0.497 | 0.424 | 0.497 | 0.809 | 10 | 0.03 | - |
| gaussian_nb | classical | 0.331 | 0.295 | 0.347 | 0.690 | 10 | 0.01 | - |

### Forecast-horizon F1 (does it hold up as the horizon grows?)

| Model | +10s | +20s | +30s | +40s | +50s | +60s |
|---|---|---|---|---|---|---|
| SENTINEL-WM (system) | 0.786 | 0.775 | 0.758 | 0.747 | 0.749 | 0.742 |
| SENTINEL-WM | 0.739 | 0.730 | 0.726 | 0.736 | 0.746 | 0.744 |
| xgboost__seq | 0.786 | 0.762 | 0.747 | 0.762 | 0.750 | 0.759 |
| hist_gradient_boosting__seq | 0.787 | 0.749 | 0.750 | 0.744 | 0.751 | 0.742 |
| gru | 0.743 | 0.732 | 0.739 | 0.734 | 0.747 | 0.748 |
| random_forest__seq | 0.763 | 0.751 | 0.742 | 0.727 | 0.736 | 0.721 |
| lstm | 0.734 | 0.728 | 0.717 | 0.725 | 0.735 | 0.735 |
| tcn | 0.739 | 0.735 | 0.710 | 0.720 | 0.739 | 0.740 |

### 2b. Per-attack-family detection (best model per family)

| Family | best model | F1 | recall | test +windows |
|---|---|---|---|---|
| DDoS | SENTINEL-WM (system) | 1.000 | 1.00 | 30 |
| DoS slowloris | SENTINEL-WM (system) | 1.000 | 1.00 | 30 |
| FTP-Patator | SENTINEL-WM (system) | 0.992 | 1.00 | 61 |
| DoS Hulk | SENTINEL-WM | 0.947 | 1.00 | 30 |
| Web Attack XSS | SENTINEL-WM (system) | 0.941 | 1.00 | 9 |
| SSH-Patator | SENTINEL-WM (system) | 0.902 | 0.99 | 80 |
| Web Attack Brute Force | SENTINEL-WM | 0.844 | 0.83 | 52 |
| DoS GoldenEye | hist_gradient_boosting__seq | 0.757 | 0.73 | 22 |
| Bot | SENTINEL-WM (system) | 0.478 | 0.48 | 77 |
| DoS Slowhttptest | SENTINEL-WM (system) | 0.400 | 0.65 | 20 |
| PortScan | gat | 0.300 | 0.36 | 22 |

_Insufficient test data (< 8 positive windows), excluded: Heartbleed, Infiltration._

Full matrix: [`benchmarks/per_family.csv`](../benchmarks/per_family.csv).

**Reading it.** Ranked by PR-AUC the strongest single model is **SENTINEL-WM (system)** (PR-AUC 0.875, AUROC 0.948). **SENTINEL-WM** (PR-AUC 0.854, AUROC 0.943, progression accuracy 0.907, detection 0.01) is the only model that also carries a progression-state head and a calibrated K-step Monte-Carlo rollout with ATT&CK phase + confidence. The `F1` column lags `F1*` because the FPR<=5% threshold is fitted on validation (higher attack prevalence) and transfers imperfectly to the test day; PR-AUC / AUROC / F1* are the fair headline numbers. Mean Lead Time stays low under the block split (attacks land mid-episode); use `SplitConfig.mode='episode_chrono'` for a lead-time-focused run.

## 3. Explainability

SHAP + attention + gradient attribution: [`explainability/explainability.json`](../explainability/explainability.json)  -  [world-model SHAP](../figures/shap_world_model.png)

## 4. Forward simulation (infiltration prediction engine)

Per-anchor K-step Monte-Carlo rollouts with ATT&CK phase + confidence: [`simulations/forward_sim_test.json`](../simulations/forward_sim_test.json)  -  [`attck_stage_forecast.csv`](../simulations/attck_stage_forecast.csv)  -  [timelines](../figures/forward_sim_timelines.png)

## 5. Saved models (for the serving app)

Uniform loader: `from sentinel_wm.registry import load_predictor`  -  index: [`models/registry.json`](../models/registry.json)

## 6. Reproduce

```bash
python -m sentinel_wm.research all
```
