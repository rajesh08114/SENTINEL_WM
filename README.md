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
| **[`frontend/`](frontend/)** | a **static** SOC-analyst console (plain ES modules, no build step; Tailwind + Chart.js from CDN) — data-source wizard, processing pipeline, forecast / progression / ATT&CK dashboard, live telemetry monitor, and the architecture + model-card pages. Talks only HTTP/WS to `backend/`. |
| **[`capture-agent/`](capture-agent/)** | a **host** process (Npcap + Administrator) that enumerates local NICs Wireshark-style, sniffs a chosen interface (optional IP/CIDR filter), reassembles packets into flow rows, and streams them to `backend/` over `WS /agent` for **live network forecasting**. Not a container service. |
| `backend/models/` | the portable model **bundle** (`sentinel-wm bundle`), shipped *inside* the backend so it deploys as one unit; regenerable, git-ignored. |
| `data/` · `artifacts/` · `runs/` | raw CSVs · pipeline working files · generated benchmark study. All git-ignored. |
| `docs/` | [`guide.md`](docs/guide.md) (usage manual), [`system_architecture.md`](docs/system_architecture.md) (**SENTINEL-WM (system)** — every layer, param, forward/backward pass), [`technical_reference.md`](docs/technical_reference.md) (loss / rollout / ATT&CK / leakage-control detail), [`proposal.md`](docs/proposal.md), [`plan_validation.md`](docs/plan_validation.md). |

---

## Quickstart

### 1. Train the models (research)

```bash
cd research
pip install -e ".[benchmark]"
python -m sentinel_wm.research all          # data → zoo → world model → benchmark → report
sentinel-wm bundle ../backend/models               # assemble the portable bundle the backend loads
```

Details + the zero-shot secondary benchmark: [`research/RUN.md`](research/RUN.md).

### 2. Serve (backend)

```bash
cd backend
pip install -e .
uvicorn app.main:app --reload
#   POST /forecast/csv          flow-CSV upload → forecast JSON
#   POST /live/sessions         start a synthetic test-bed or live-capture session
#   WS   /live/sessions/{id}/stream    subscribe to a live session's forecasts
#   WS   /agent                 a capture agent connects here
#   WS   /stream                raw telemetry (NDJSON flow records)
#   GET  /meta  /models  /health  /agent/status
```

### 2b. Real-time: synthetic test-bed & live capture

```bash
# synthetic scenario streamed through the real pipeline (no setup):
make synth-demo            # or: POST /live/sessions {"source":"synthetic","scenario":"portscan"}

# live capture from a local interface (host, needs Npcap + Administrator):
cd capture-agent && pip install -e . && sentinel-capture run --backend ws://localhost:8000
# then in the console: Live → Live capture → pick an interface → Start
```

Note: the production model runs hot on out-of-distribution synthetic traffic, so
absolute P(attack) on a synthetic session is not calibrated — read the relative
ramp, the progression sequence, and the ATT&CK phase mapping.

### 3. The console (frontend)

```bash
cd frontend && python -m http.server 5173     # no build step — any static server
# open http://localhost:5173 · set the API base URL in the topbar (default :8000)
```

### 4. Or everything in containers

```bash
docker compose up --build          # backend :8000  +  console :8080; mounts ./backend/models as a volume
```

---

## How the pieces connect

```
       research/               backend/models/  (the bundle = the ONLY interface)
  raw CSV/PCAP ─► sentinel_wm ─► sentinel-wm bundle ─► world_model.pt · state_scaler.pkl
                 train+benchmark                        models/nn/{tcn,lstm,gru}.pt
                                                        bundle.json (feature_names, blend_w)
                                                                  │
       backend/  (self-contained: app/sentinel_infer/ vendored + models/ shipped in)
   CSV · PCAP · WS stream ─► clean → state windows → scaler → simulate_anchor + blend
                                                                  │
                          forecast JSON:  per-horizon P(attack)+CI · detection_prob
                                          progression state · ATT&CK phase+confidence
                                          driving features + temporal saliency
                                                                  │
                                                   frontend/  (static SPA, hash router)
```

The three folders are **decoupled**: `research/` produces a bundle; `backend/`
consumes only the bundle (its inference code is vendored in
`app/sentinel_infer/`); `frontend/` talks only HTTP/WS to `backend/` and needs no
build step. Retraining = re-run `sentinel-wm bundle`; no backend rebuild.

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
