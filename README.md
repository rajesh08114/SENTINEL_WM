# SENTINEL-WM

**S**patio-temporal **E**nemy **N**etwork **I**ntelligence with a **L**earned **World Model**

A world model for **predictive cyber defence**. Instead of labelling each network
flow benign/malicious in isolation, SENTINEL-WM aggregates traffic into
**10-second network-state windows**, learns how those states *evolve*
(`P(S_{t+1} | S_{t-L:t})`), and **rolls the world forward K = 6 steps** to
estimate the probability — and the MITRE ATT&CK phase — of attacker progression
**before the kill chain completes**, with a per-prediction explanation.

Built for the SIH problem *"AI systems that learn network behaviour, anticipate
attacker progression and support proactive cyber defence using World Models."*

---

## Repository layout

| folder | what it is |
|---|---|
| **[`research/`](research/)** | the ML workspace — the `sentinel_wm` package (preprocessing → state windows → sequences → model zoo → world model → benchmark → explainability → forward simulation), the `extraction/` PCAP tooling, and the phase notebooks. CLI + JSON artifacts. |
| **[`backend/`](backend/)** | a **self-contained** FastAPI service (inference code vendored in `app/sentinel_infer/`, no `research/` import) that loads a **model bundle** and scores incoming traffic (flow CSV upload, PCAP upload, or a live telemetry WebSocket) → per-horizon forecasts **with explanations and ATT&CK phase mapping**. |
| **[`frontend/`](frontend/)** | a Next.js UI for uploads, live monitoring, and the forecast / explanation / ATT&CK views. |
| `models/` | the portable model **bundle** the backend loads (`sentinel-wm bundle`); regenerable, git-ignored. |
| `data/` · `artifacts/` · `runs/` | raw CSVs · pipeline working files · generated benchmark study. All git-ignored. |
| `docs/` | [`guide.md`](docs/guide.md) (usage manual), [`system_architecture.md`](docs/system_architecture.md) (**SENTINEL-WM (system)** — every layer, param, forward/backward pass), [`technical_reference.md`](docs/technical_reference.md) (loss / rollout / ATT&CK / leakage-control detail), [`proposal.md`](docs/proposal.md), [`plan_validation.md`](docs/plan_validation.md). |

---

## Quickstart

### 1. Train the models (research)

```bash
cd research
pip install -e ".[benchmark]"
python -m sentinel_wm.research all          # data → zoo → world model → benchmark → report
sentinel-wm bundle ../models               # assemble the portable bundle the backend loads
```

Details + the zero-shot secondary benchmark: [`research/RUN.md`](research/RUN.md).

### 2. Serve (backend)

```bash
cd backend
pip install -e .
SENTINEL_WM_MODEL_DIR=../models uvicorn app.main:app --reload
#   POST /forecast/csv   (flow-CSV upload → forecast JSON)
#   WS   /stream         (telemetry: stream NDJSON flow records, receive forecasts)
#   GET  /meta  /models  /health
```

### 3. Or everything in containers

```bash
docker compose up --build          # backend (+ frontend); mounts ./models as a volume
```

---

## How the pieces connect

```
       research/                       models/  (the bundle = the ONLY interface)
  raw CSV/PCAP ─► sentinel_wm ─► sentinel-wm bundle ─► world_model.pt · state_scaler.pkl
                 train+benchmark                        models/nn/{tcn,lstm,gru}.pt
                                                        bundle.json (feature_names, blend_w)
                                                                  │  (mounted volume)
       backend/  (self-contained; app/sentinel_infer/ is vendored, no research import)
   CSV · PCAP · WS stream ─► clean → state windows → scaler → simulate_anchor + blend
                                                                  │
                          forecast JSON:  per-horizon P(attack)+CI · detection_prob
                                          progression state · ATT&CK phase+confidence
                                          driving features + temporal saliency
                                                                  │
                                                          frontend/  (Next.js)
```

The three folders are **decoupled**: `research/` produces a bundle; `backend/`
consumes only the bundle (its inference code is vendored in
`app/sentinel_infer/`); `frontend/` talks only HTTP/WS to `backend/`. Retraining
= re-run `sentinel-wm bundle`; no backend rebuild.

---

## Results

Regenerated every run into [`runs/benchmarks/benchmark.md`](runs/benchmarks/benchmark.md)
and [`runs/reports/RESEARCH_REPORT.md`](runs/reports/RESEARCH_REPORT.md). On the
leakage-safe `stratified` split (all 5 CIC-IDS-2017 days), ranked by **PR-AUC**:

| model | PR-AUC | F1\* | AUROC | notes |
|---|---|---|---|---|
| **SENTINEL-WM (system)** | **0.992** | 0.968 | 1.000 | #1; flattest horizon decay; + progression state, K-step MC rollout, ATT&CK phase |
| lstm | 0.992 | 0.968 | 1.000 | ties on threshold-free metrics; pure attack classifier |
| gru / tcn | 0.98 / 0.98 | 0.96 / 0.93 | 1.00 | |
| SENTINEL-WM (raw) | 0.957 | 0.909 | 0.998 | |

**Caveat:** the test set has ~76 positive sequences of easily-separable volumetric
attacks, so the top models are within noise; it measures *now-casting a sustained
attack*, not *onset forecasting* (`persistence` scores F1 0.99, lead time ~0 s).
Full trust analysis: [`docs/technical_reference.md`](docs/technical_reference.md) §1.10.

---

## License

MIT (see `research/pyproject.toml`). CIC-IDS-2017 / CTU-13 / UNSW-NB15 are the
property of their publishers and are **not** redistributed here. Research
prototype, not a production IDS.
