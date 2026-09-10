# app/sentinel_infer/ — vendored inference library

A **self-contained copy** of the SENTINEL-WM inference path, so `backend/` depends
on nothing in `research/` — only on a **model bundle directory**
(`SENTINEL_WM_MODEL_DIR`, produced by `sentinel-wm bundle`).

```
schema.py       LOCKED feature column lists + slim config (no paths, no I/O)
attack_stages.py  MITRE ATT&CK phase mapping        (verbatim copy)
preprocess.py   clean_flow_frame + label/day/winsor  (subset of preprocessing.py)
windows.py      build_state_windows                  (subset of state_windows.py)
net.py          SentinelWorldModel + STN + encoders + TCN/LSTM/GRU members
explain.py      per-anchor gradient×input + attention saliency
forecast.py     load_bundle() -> InferenceEngine -> forecast(flows_df) -> JSON
```

## Provenance & keeping it in sync

These files are hand-maintained copies of `research/sentinel_wm/*.py` with the
package import rewritten (`from sentinel_wm import config as C` →
`from . import schema as C`) and training-only code removed. They must stay
behaviourally equivalent to the research modules for the **feature aggregation**
(`windows._agg_windows`) and the **model architectures** (`net.*`), or a bundle
trained by `research/` will not load / will score differently.

Guards against drift:

* **`bundle.json["feature_names"]`** carries the exact ordered feature schema the
  models were trained on. `forecast.load_bundle` raises `BundleContractError` if
  it does not equal `windows.STATE_FEATURE_COLS` here.
* **`tests/test_vendor_sync.py`** (runs only when `../research` is checked out)
  diffs the LOCKED column lists in `schema.py` against
  `research/sentinel_wm/config.py` and the ATT&CK table against
  `research/sentinel_wm/attack_stages.py`.

If you change the state-window aggregation or a model architecture in
`research/`, re-vendor: copy the changed function bodies here, run
`pytest backend/`, rebuild the bundle (`sentinel-wm bundle`).
