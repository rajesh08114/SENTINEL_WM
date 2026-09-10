# backend/ — SENTINEL-WM inference API

FastAPI service that loads a trained **model bundle** directory and turns
incoming traffic into per-horizon attack-progression forecasts **with
explanations and MITRE ATT&CK phase mapping**.

**Self-contained.** The inference code is vendored in
[`app/sentinel_infer/`](app/sentinel_infer) — the backend has **no dependency on
`../research`**. Its only interface is the bundle directory
(`SENTINEL_WM_MODEL_DIR`, produced by `sentinel-wm bundle` in `research/`). Copy
`backend/` + a `models/` bundle anywhere and it runs. Drift between the vendored
copy and the research source is caught by `tests/test_vendor_sync.py` and by a
`feature_names` contract check in `bundle.json`.

It serves **SENTINEL-WM (system)** — the world model (direct head + K-step MC
rollout + snapshots) whose per-horizon P(attack) is blended (validation-tuned
weight from `bundle.json`) with three decorrelated sequence nets (`tcn`, `lstm`,
`gru`) for the sharpest detection. The **forecast narrative** — `attack_prob`
with its 95 % CI, progression state, ATT&CK phase — stays the world model's; the
blended value lands in `detection_prob` and drives the alert / lead-time. Set
`SENTINEL_SERVE_MODE=world_model` to skip the members. Pipeline
(`app/sentinel_infer/forecast.py`): `clean_flow_frame` → `build_state_windows` →
persisted `RobustScaler` → `simulate_anchor` (+ member blend).

## Run

```bash
# 0. build the bundle once (in ../research) - it lands in backend/models/:
cd ../research && python -m sentinel_wm.research all && python -m sentinel_wm.cli bundle ../backend/models

# 1. install + run  (no ../research dependency; reads ./models by default)
cd ../backend
pip install -e ".[dev]"
uvicorn app.main:app --reload
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
  main.py               FastAPI factory + lifespan (loads the engine once)
  settings.py           pydantic-settings (SENTINEL_* env)
  schemas.py            response models
  sentinel_infer/       VENDORED, self-contained inference lib (see its README):
    schema · attack_stages · preprocess · windows · net · explain · forecast
  inference/loader.py   get_engine() -> sentinel_infer.forecast.load_bundle(...)
  inference/pipeline.py thin adapter: forecast(df) -> engine.forecast(df)
  streaming/windower.py online 10 s windowing for the telemetry WS
  jobs/store.py         SQLite job status
  jobs/worker.py        ThreadPoolExecutor (shared in-process engine)
  api/                  routes_meta / routes_forecast / routes_jobs / ws_stream
tests/                  pytest (random-bundle fixture) + test_vendor_sync + ws_smoke.py
```

## Not in the MVP (see the plan)

PCAP ingestion (Phase 2), the GAT member of the blend (needs per-window host
graphs as a side input), auth, Redis. The `SENTINEL-WM (system)` blend with
`tcn`/`lstm`/`gru` **is** wired in (`SENTINEL_SERVE_MODE`). Job execution is an
in-process thread pool — swap for a process pool or a queue if you add GIL-bound
job kinds or horizontal scale.
