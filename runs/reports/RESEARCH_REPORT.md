# SENTINEL-WM - Research Report

_Generated 2026-09-10 12:14_  |  config: [`run_config.json`](../run_config.json)

## 1. Data

- Flows: **2,829,609**  |  10-s state windows: **14,644**  |  sequences (L=12, K=6, F=53): **14,197**
- Label distribution: [`data_profile/flow_label_distribution.csv`](../data_profile/flow_label_distribution.csv)  -  [timeline](../figures/attack_timeline_by_day.png)
- Split mode = **auto** (auto -> stratified), leakage-safe (contiguous chunks, scaler fit on real-train only): [`data_profile/split_summary.csv`](../data_profile/split_summary.csv)  -  per-family coverage: [`data_profile/split_family_windows.csv`](../data_profile/split_family_windows.csv)

  - **train**: 8489 windows, 9.0% attack
  - **val**: 2987 windows, 15.4% attack
  - **test**: 3168 windows, 11.1% attack

## 2. Model benchmark

Full table + per-horizon F1: [`benchmarks/benchmark.md`](../benchmarks/benchmark.md)  -  raw: [`benchmark_full.csv`](../benchmarks/benchmark_full.csv)  -  figures: [horizon F1](../figures/horizon_f1.png), [ROC](../figures/roc.png), [PR](../figures/pr.png), [lead time](../figures/lead_time.png)

Ranked by **PR-AUC** (threshold-free; robust to the val->test attack-prevalence shift). `F1` = at the val-tuned best-F1 threshold; `F1*` = test-set ceiling. The test set has ~76 positive sequences, so the top cluster (system / lstm / gru / persistence) is within noise - trust PR-AUC / F1* / AUROC.

| Model | Family | PR-AUC | F1 | F1* | AUROC | MLT (s) | Detect | ProgAcc |
|---|---|---|---|---|---|---|---|---|
| SENTINEL-WM (system) | system | 0.992 | 0.920 | 0.968 | 1.000 | 0 | 0.00 | 0.980 |
| lstm | nn | 0.992 | 0.968 | 0.968 | 1.000 | 0 | 0.00 | 0.989 |
| persistence | reference | 0.987 | 0.993 | 0.993 | 1.000 | 0 | 0.00 | - |
| tcn | nn | 0.981 | 0.874 | 0.933 | 0.999 | 0 | 0.00 | 0.991 |
| gru | nn | 0.977 | 0.955 | 0.955 | 1.000 | 0 | 0.00 | 0.994 |
| SENTINEL-WM | world_model | 0.960 | 0.828 | 0.914 | 0.998 | 0 | 0.00 | 0.980 |
| mlp | nn | 0.849 | 0.818 | 0.827 | 0.986 | 0 | 0.00 | 0.985 |
| gat | graph | 0.599 | 0.591 | 0.621 | 0.973 | 0 | 0.00 | 0.969 |
| mlp_sklearn__seq | classical | 0.454 | 0.583 | 0.619 | 0.924 | 0 | 0.00 | - |
| xgboost__seq | classical | 0.446 | 0.427 | 0.462 | 0.811 | 0 | 0.00 | - |
| random_forest__seq | classical | 0.355 | 0.440 | 0.448 | 0.397 | 0 | 0.00 | - |
| knn | classical | 0.314 | 0.251 | 0.348 | 0.749 | 0 | 0.00 | - |
| extra_trees | classical | 0.273 | 0.265 | 0.396 | 0.368 | 0 | 0.00 | - |
| mlp_sklearn | classical | 0.269 | 0.248 | 0.337 | 0.851 | 0 | 0.00 | - |
| random_forest | classical | 0.265 | 0.266 | 0.371 | 0.455 | 0 | 0.00 | - |
| logistic_regression__seq | classical | 0.244 | 0.502 | 0.504 | 0.961 | 0 | 0.00 | - |
| linear_svc | classical | 0.192 | 0.236 | 0.308 | 0.829 | 0 | 0.00 | - |
| xgboost | classical | 0.187 | 0.028 | 0.295 | 0.226 | 10 | 0.07 | - |
| logistic_regression | classical | 0.137 | 0.246 | 0.248 | 0.823 | 0 | 0.00 | - |
| hist_gradient_boosting__seq | classical | 0.125 | 0.086 | 0.264 | 0.176 | 0 | 0.00 | - |
| gaussian_nb | classical | 0.065 | 0.112 | 0.205 | 0.499 | 0 | 0.00 | - |
| hist_gradient_boosting | classical | 0.040 | 0.054 | 0.074 | 0.172 | 10 | 0.07 | - |

### Forecast-horizon F1 (does it hold up as the horizon grows?)

| Model | +10s | +20s | +30s | +40s | +50s | +60s |
|---|---|---|---|---|---|---|
| SENTINEL-WM (system) | 0.932 | 0.961 | 0.948 | 0.935 | 0.908 | 0.889 |
| lstm | 0.962 | 0.962 | 0.948 | 0.947 | 0.917 | 0.859 |
| persistence | 0.994 | 0.987 | 0.980 | 0.973 | 0.966 | 0.952 |
| tcn | 0.879 | 0.873 | 0.871 | 0.864 | 0.850 | 0.832 |
| gru | 0.955 | 0.954 | 0.941 | 0.919 | 0.905 | 0.904 |
| SENTINEL-WM | 0.836 | 0.826 | 0.840 | 0.852 | 0.855 | 0.829 |
| mlp | 0.815 | 0.815 | 0.808 | 0.792 | 0.784 | 0.757 |
| gat | 0.537 | 0.534 | 0.544 | 0.548 | 0.540 | 0.540 |

### 2b. Per-attack-family detection (best model per family)

| Family | best model | F1 | recall | test +windows |
|---|---|---|---|---|
| DDoS | SENTINEL-WM (system) | 1.000 | 1.00 | 12 |
| SSH-Patator | lstm | 1.000 | 1.00 | 16 |
| FTP-Patator | lstm | 0.952 | 1.00 | 40 |

_Insufficient test data (< 8 positive windows), excluded: Bot, DoS Hulk, Heartbleed, Infiltration._

Full matrix: [`benchmarks/per_family.csv`](../benchmarks/per_family.csv).

**Reading it.** Ranked by PR-AUC the strongest single model is **SENTINEL-WM (system)** (PR-AUC 0.992, AUROC 1.000). **SENTINEL-WM** (PR-AUC 0.960, AUROC 0.998, progression accuracy 0.980, detection 0.00) is the only model that also carries a progression-state head and a calibrated K-step Monte-Carlo rollout with ATT&CK phase + confidence. `F1` can lag `F1*` when a model's val threshold transfers imperfectly to the test prevalence; PR-AUC / AUROC / F1* are the fair headline numbers, and with ~76 positive test sequences the top models are within noise of each other. Mean Lead Time is ~0 s for every model because the leakage-safe `stratified` split puts whole episodes on one side, so test anchors sit mid-episode - the benchmark measures now-casting, not onset forecasting. Use `SplitConfig.mode='episode_chrono'` and/or `WindowConfig.flow_augment` to measure the forecast itself. See docs/technical_reference.md 1.10.

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
