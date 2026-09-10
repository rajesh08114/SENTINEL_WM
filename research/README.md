# research/

Everything the SENTINEL-WM study produced, each number backed by a file.
Regenerate with `python -m sentinel_wm.research all`.

| folder | contents |
|---|---|
| `data_profile/` | label & timeline stats, split balance |
| `models/` | every trained model (`classical/*.pkl`, `nn/*.pt`, `world_model.pt`) + `registry.json` |
| `benchmarks/` | `benchmark_full.csv/.md/.json`, `per_horizon_f1.csv`, `leadtime.csv` |
| `figures/` | all plots (horizon F1, ROC/PR, lead time, SHAP, timelines) |
| `explainability/` | SHAP / attention / gradient attribution JSON |
| `simulations/` | K-step forward-simulation runs + ATT&CK stage forecasts |
| `reports/` | `RESEARCH_REPORT.md` |
| `logs/` | training curves per model |

Start at [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md).
