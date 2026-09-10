# SENTINEL-WM - Research Report

_Generated 2026-09-10 10:48_  |  config: [`run_config.json`](../run_config.json)

## 1. Data

- Flows: **2,829,609**  |  10-s state windows: **14,644**  |  sequences (L=12, K=6, F=53): **14,197**
- Label distribution: [`data_profile/flow_label_distribution.csv`](../data_profile/flow_label_distribution.csv)  -  [timeline](../figures/attack_timeline_by_day.png)
- Split mode = **auto** (auto -> stratified), leakage-safe (contiguous chunks, scaler fit on real-train only): [`data_profile/split_summary.csv`](../data_profile/split_summary.csv)  -  per-family coverage: [`data_profile/split_family_windows.csv`](../data_profile/split_family_windows.csv)

  - **train**: 8489 windows, 9.0% attack
  - **val**: 2987 windows, 15.4% attack
  - **test**: 3168 windows, 11.1% attack

## 2. Model benchmark

Full table + per-horizon F1: [`benchmarks/benchmark.md`](../benchmarks/benchmark.md)  -  raw: [`benchmark_full.csv`](../benchmarks/benchmark_full.csv)  -  figures: [horizon F1](../figures/horizon_f1.png), [ROC](../figures/roc.png), [PR](../figures/pr.png), [lead time](../figures/lead_time.png)

Ranked by **PR-AUC** (threshold-free; robust to the val->test attack-prevalence shift). `F1` = at the FPR<=5% threshold; `F1*` = best achievable by sweeping it.

| Model | Family | PR-AUC | F1 | F1* | AUROC | MLT (s) | Detect | ProgAcc |
|---|---|---|---|---|---|---|---|---|
| SENTINEL-WM (system) | system | 0.990 | 0.543 | 0.955 | 1.000 | 0 | 0.00 | 0.978 |
| persistence | reference | 0.987 | 0.993 | 0.993 | 1.000 | 0 | 0.00 | - |
| lstm | nn | 0.984 | 0.776 | 0.940 | 1.000 | 0 | 0.00 | 0.991 |
| tcn | nn | 0.972 | 0.765 | 0.914 | 0.999 | 0 | 0.00 | 0.990 |
| gru | nn | 0.956 | 0.764 | 0.936 | 0.999 | 0 | 0.00 | 0.993 |
| SENTINEL-WM | world_model | 0.954 | 0.574 | 0.903 | 0.998 | 0 | 0.00 | 0.978 |
| mlp | nn | 0.904 | 0.782 | 0.841 | 0.989 | 0 | 0.00 | 0.988 |
| gat | graph | 0.599 | 0.610 | 0.621 | 0.973 | 0 | 0.00 | 0.969 |
| mlp_sklearn__seq | classical | 0.454 | 0.609 | 0.619 | 0.924 | 0 | 0.00 | - |
| xgboost__seq | classical | 0.446 | 0.452 | 0.462 | 0.811 | 0 | 0.00 | - |
| random_forest__seq | classical | 0.355 | 0.329 | 0.448 | 0.397 | 0 | 0.00 | - |
| knn | classical | 0.314 | 0.242 | 0.348 | 0.749 | 0 | 0.00 | - |
| extra_trees | classical | 0.273 | 0.110 | 0.396 | 0.368 | 0 | 0.00 | - |
| mlp_sklearn | classical | 0.269 | 0.227 | 0.337 | 0.851 | 0 | 0.00 | - |
| random_forest | classical | 0.265 | 0.171 | 0.371 | 0.455 | 10 | 0.07 | - |
| logistic_regression__seq | classical | 0.244 | 0.438 | 0.504 | 0.961 | 0 | 0.00 | - |
| linear_svc | classical | 0.194 | 0.222 | 0.311 | 0.829 | 0 | 0.00 | - |
| xgboost | classical | 0.187 | 0.031 | 0.295 | 0.226 | 10 | 0.07 | - |
| logistic_regression | classical | 0.137 | 0.213 | 0.248 | 0.823 | 0 | 0.00 | - |
| hist_gradient_boosting__seq | classical | 0.125 | 0.045 | 0.264 | 0.176 | 10 | 0.07 | - |
| gaussian_nb | classical | 0.065 | 0.137 | 0.205 | 0.499 | 0 | 0.00 | - |
| hist_gradient_boosting | classical | 0.040 | 0.039 | 0.074 | 0.172 | 10 | 0.07 | - |

### Forecast-horizon F1 (does it hold up as the horizon grows?)

| Model | +10s | +20s | +30s | +40s | +50s | +60s |
|---|---|---|---|---|---|---|
| SENTINEL-WM (system) | 0.561 | 0.568 | 0.565 | 0.557 | 0.543 | 0.547 |
| persistence | 0.994 | 0.987 | 0.980 | 0.973 | 0.966 | 0.952 |
| lstm | 0.808 | 0.843 | 0.822 | 0.789 | 0.770 | 0.753 |
| tcn | 0.777 | 0.804 | 0.793 | 0.774 | 0.780 | 0.746 |
| gru | 0.788 | 0.802 | 0.787 | 0.764 | 0.796 | 0.761 |
| SENTINEL-WM | 0.598 | 0.600 | 0.592 | 0.598 | 0.581 | 0.581 |
| mlp | 0.789 | 0.800 | 0.784 | 0.784 | 0.753 | 0.737 |
| gat | 0.577 | 0.575 | 0.571 | 0.559 | 0.573 | 0.565 |

### 2b. Per-attack-family detection (best model per family)

| Family | best model | F1 | recall | test +windows |
|---|---|---|---|---|
| DDoS | SENTINEL-WM (system) | 1.000 | 1.00 | 12 |
| SSH-Patator | tcn | 1.000 | 1.00 | 16 |
| FTP-Patator | SENTINEL-WM (system) | 0.927 | 1.00 | 40 |

_Insufficient test data (< 8 positive windows), excluded: Bot, DoS Hulk, Heartbleed, Infiltration._

Full matrix: [`benchmarks/per_family.csv`](../benchmarks/per_family.csv).

**Reading it.** Ranked by PR-AUC the strongest single model is **SENTINEL-WM (system)** (PR-AUC 0.990, AUROC 1.000). **SENTINEL-WM** (PR-AUC 0.954, AUROC 0.998, progression accuracy 0.978, detection 0.00) is the only model that also carries a progression-state head and a calibrated K-step Monte-Carlo rollout with ATT&CK phase + confidence. The `F1` column lags `F1*` because the FPR<=5% threshold is fitted on validation (higher attack prevalence) and transfers imperfectly to the test day; PR-AUC / AUROC / F1* are the fair headline numbers. Mean Lead Time stays low under the block split (attacks land mid-episode); use `SplitConfig.mode='episode_chrono'` for a lead-time-focused run.

## 4. Explainability

SHAP + attention + gradient attribution: [`explainability/explainability.json`](../explainability/explainability.json)  -  [world-model SHAP](../figures/shap_world_model.png)

## 5. Forward simulation (infiltration prediction engine)

Per-anchor K-step Monte-Carlo rollouts with ATT&CK phase + confidence: [`simulations/forward_sim_test.json`](../simulations/forward_sim_test.json)  -  [`attck_stage_forecast.csv`](../simulations/attck_stage_forecast.csv)  -  [timelines](../figures/forward_sim_timelines.png)

## 6. Saved models (for the serving app)

Uniform loader: `from sentinel_wm.registry import load_predictor`  -  index: [`models/registry.json`](../models/registry.json)

## 7. Reproduce

```bash
python -m sentinel_wm.research all
```
