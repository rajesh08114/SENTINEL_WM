# backend/ — SENTINEL-WM inference API

FastAPI service that loads the trained **model bundle** (`../models/`, built by
`sentinel-wm bundle`) and turns incoming traffic into per-horizon
attack-progression forecasts **with explanations and MITRE ATT&CK phase mapping**.

The MVP serves the **SENTINEL-WM world model** (self-ensemble: direct head +
K-step Monte-Carlo rollout + snapshots). It reuses the research package end to
end — `preprocessing.clean_flow_frame` → `state_windows.build_state_windows` →
persisted `RobustScaler` → `forward_sim.simulate_anchor`.

## Run

```bash
# 0. build the bundle once (in ../research):
cd ../research && python -m sentinel_wm.research all && python -m sentinel_wm.cli bundle ../models

# 1. install + run
cd ../backend
pip install -e ../research && pip install -e ".[dev]"
cp .env.example .env                       # edit SENTINEL_WM_MODEL_DIR if needed
SENTINEL_WM_MODEL_DIR=../models uvicorn app.main:app --reload
```

`GET /docs` for the interactive API. Tests need **no** trained model — they build
a throwaway random bundle: `python -m pytest -q`.

## Endpoints

| method | path | purpose |
|---|---|---|
| `GET`  | `/health` | `ok` / `degraded` (no bundle) / `error` |
| `GET`  | `/meta` | L, K, F, window size, progression states, feature names, bundle manifest |
| `GET`  | `/models` | the full trained-model registry (reference; MVP serves `SENTINEL-WM`) |
| `POST` | `/forecast/csv` | multipart `file` (flow CSV) + optional `family_hint`, `explain`. ≤ `MAX_SYNC_FLOWS` → `200` `ForecastResponse`; larger → `202 {job_id}` |
| `POST` | `/forecast/pcap` | `501` — Phase 2 (run `research/extraction/extractor.py` and upload the CSV) |
| `GET`  | `/jobs`, `/jobs/{id}`, `/jobs/{id}/result` | async job status + result |
| `WS`   | `/stream` | telemetry: stream `{"type":"flows","records":[…]}`, receive `{"type":"forecast",…}` per closed 10 s window |

### Forecast shape (per anchor)

```jsonc
{
  "alert": true, "lead_time_seconds": 40, "first_alert_k": 3, "max_attack_prob": 0.81,
  "horizon": [ {
    "k": 1, "horizon_seconds": 10, "attack_prob": 0.34, "attack_ci": [0.21, 0.49],
    "attack_std": 0.07, "progression_state": "PRE_ATTACK",
    "progression_dist": {"NORMAL": 0.6, "PRE_ATTACK": 0.3, ...},
    "attck": { "mitre_tactic": "Reconnaissance", "technique_ids": ["T1595","T1046"],
               "kill_chain_phase": "Reconnaissance", "confidence": "Medium",
               "rationale": "...", "dominant_family": "PortScan", "family_transition": null }
  }, ... 6 steps ],
  "driving_features": { "top_features": [...], "temporal_saliency_gradient": [...],
                        "temporal_saliency_attention": [...] }
}
```

## Config (`.env`, prefix `SENTINEL_`)

`SENTINEL_WM_MODEL_DIR` (bundle path) · `SENTINEL_DEVICE` (cpu/cuda) ·
`SENTINEL_MAX_SYNC_FLOWS` · `SENTINEL_MC_SAMPLES` · `SENTINEL_WORKERS` ·
`SENTINEL_JOB_DIR` · `SENTINEL_STREAM_GRACE_SECONDS` · `SENTINEL_CORS_ORIGINS`.

## Layout

```
app/
  main.py             FastAPI factory + lifespan (loads the model once)
  settings.py         pydantic-settings; exports SENTINEL_WM_MODEL_DIR
  schemas.py          response models
  inference/loader.py  Engine singleton (model + ckpt + scaler + explainer)
  inference/pipeline.py  flows -> state windows -> tensors -> simulate_anchor
  streaming/windower.py  online 10 s windowing for the telemetry WS
  jobs/store.py        SQLite job status
  jobs/worker.py       ThreadPoolExecutor (shared in-process model)
  api/                 routes_meta / routes_forecast / routes_jobs / ws_stream
tests/                 pytest (random-bundle fixture) + ws_smoke.py
```

## Not in the MVP (see the plan)

PCAP ingestion (Phase 2), the `SENTINEL-WM (system)` blend / GAT (needs per-window
host graphs), auth, Redis. Job execution is an in-process thread pool — swap for a
process pool or a queue if you add GIL-bound job kinds or horizontal scale.
