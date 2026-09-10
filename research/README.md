# research/ — SENTINEL-WM machine-learning workspace

The ML half of the project: the `sentinel_wm` package (data → state windows →
sequences → model zoo → world model → benchmark → explainability → forward
simulation), the raw-feature `extraction/` tooling, and the phase notebooks.

The **application** that serves these models lives in [`../backend/`](../backend)
(FastAPI) and [`../frontend/`](../frontend) (Next.js).

## Install

```bash
pip install -e .                  # core
pip install -e ".[benchmark]"     # + xgboost / lightgbm / shap (full zoo)
pip install -e ".[notebooks]"     # + jupyter to run notebooks/
```

Python 3.10+. CUDA GPU auto-detected, not required. No internet at any stage.
`SENTINEL_WM_ROOT` is auto-resolved to the repo root (the folder holding `data/`,
`artifacts/`, `runs/`); set it explicitly if you install the package elsewhere.

## Run the study

```bash
python -m sentinel_wm.research all           # primary (stratified) -> ../runs/
python -m sentinel_wm.research all --split family --outdir runs_zeroshot   # zero-shot
python -m sentinel_wm.research report        # fold the zero-shot section in
sentinel-wm bundle ../backend/models                 # assemble the portable model bundle
```

Details: [`RUN.md`](RUN.md), [`../docs/guide.md`](../docs/guide.md),
[`../docs/technical_reference.md`](../docs/technical_reference.md).

## Layout

| path | contents |
|---|---|
| `sentinel_wm/` | the package (24 modules) — see `../docs/guide.md` §2 |
| `sentinel_wm/bundle.py` | `sentinel-wm bundle` — copies the trained models + scaler + registry into `../backend/models/` (shipped inside the backend) |
| `extraction/` | from-scratch CICFlowMeter + packet-feature extractor (PCAP → flow CSV; needs `tshark`) |
| `notebooks/` | `00`–`05` phase notebooks (run from here; they read `../runs/…`) |

Generated outputs land in `../runs/` (benchmark tables, figures, trained models,
`registry.json`), `../artifacts/` (working parquet/npz/scaler), and `../backend/models/`
(the deploy bundle). All three are git-ignored.
