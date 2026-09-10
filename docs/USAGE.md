# SENTINEL-WM — Usage, Testing & Demo Guide

How to run the whole system, exercise every capability, and drive a live demo —
with a concrete list of **what to observe** at each step.

- **What this is:** a world model that forecasts network‑attacker progression
  **6 windows (60 s) ahead** from CIC‑IDS‑2017‑style flow telemetry, with MITRE
  ATT&CK phase mapping and per‑feature explanations.
- **Four parts:**
  | part | what it does | where it runs |
  |---|---|---|
  | `research/` | trains + benchmarks; emits a portable **model bundle** | CLI / offline |
  | `backend/` | FastAPI service — loads the bundle, scores traffic | container or `uvicorn` |
  | `frontend/` | Next.js SOC console | static export behind nginx / `npm run dev` |
  | `capture-agent/` | sniffs a local NIC → flow rows → backend | **host** (Npcap + Admin) |

---

## 1. One‑time setup

### 1.1 Prerequisites
- Python 3.10+ and Node 20+ on `PATH`.
- The **model bundle** must exist at `backend/models/` (`world_model.pt`,
  `state_scaler.pkl`, `models/nn/*.pt`, `bundle.json`). If it isn't there:
  ```bash
  cd research && pip install -e ".[benchmark]"
  python -m sentinel_wm.research all          # full train + benchmark (long)
  python -m sentinel_wm.cli bundle ../backend/models
  ```
- Docker Desktop if you want the one‑command stack.

### 1.2 Install (native, for development / demo without Docker)
```bash
# backend
cd backend && pip install -e ".[dev]"

# frontend
cd ../frontend && npm install

# capture agent (only on the host you'll sniff from)
cd ../capture-agent && pip install -e .
```
Or from the repo root: `make install` (backend + research).

### 1.3 Demo data
Two small, **real** CIC‑IDS‑2017 slices ship in `demo/` (rebuild with
`python demo/make_demo_csv.py` if the source CSV is present):

| file | rows | window | expectation |
|---|---|---|---|
| `demo/demo_benign.csv` | ~4.5k | 45 min, quiet | **0 alerts**, P(attack) stays ≲ 0.12 |
| `demo/demo_dos_onset.csv` | ~7.2k | benign → sustained DoS | alerts fire, **~60 s lead time**, tactic = *Impact* |

---

## 2. Start it

### Option A — everything in containers (simplest)
```bash
docker compose up --build
#   backend  http://localhost:8000
#   console  http://localhost:8080
```
`docker compose down -v` to stop and drop the job volume.
> The **capture agent is not in compose** — it needs host packet‑capture
> privileges. Run it separately (§5).

### Option B — native dev (hot reload, easiest to inspect)
```bash
# terminal 1
cd backend && uvicorn app.main:app --reload            # :8000

# terminal 2
cd frontend && npm run dev                             # :3000
```
Open `http://localhost:3000`. In the **topbar API field**, confirm it points at
`http://localhost:8000` (it is auto‑detected; you can also pass `?api=` in the
URL).

### Option C — Makefile shortcuts
```bash
make backend-dev        # uvicorn --reload
make frontend-dev       # next dev
make synth-demo         # stream a synthetic scenario from a running backend (CLI)
make backend-test       # 47 pytest
make agent-test         # 8 pytest (flow meter + interfaces)
make frontend-test      # 14 Vitest
```

### First check
```bash
curl -s localhost:8000/health | python -m json.tool
```
Observe: `status: "ok"`, `bundle.loaded: true`, `bundle.serve_mode: "system"`,
`bundle.L: 12`, `bundle.K: 6`, `bundle.n_features: 53`, plus `live`, `agent`,
`uptime_s`.

---

## 3. Walkthrough A — CSV upload (the headline demo)

This is the most convincing path: **real, in‑distribution traffic**, so the
probabilities mean something.

1. Console → **Data Sources** → *CSV upload* tab.
2. Drag `demo/demo_dos_onset.csv` onto the dropzone.
3. **Observe — Column match** panel:
   - `READY` badge, `11/11 required · … Tier‑2 · … flag`.
   - *Auto‑mapped aliases* — the CIC header uses `Timestamp`, `Src IP`, `Tot Fwd
     Pkts`, … which are mapped to the canonical names. `synthTimeAxis` note: the
     time axis is derived from `Timestamp`.
   - "A Label column is present — used only for display, never as a model input."
4. **Observe — Data information**: ~7.2k rows × ~85 cols, span ~2160 s, "est.
   windows / anchors ≈ 216 / ~204", label distribution chip shows the DoS mix.
5. **Observe — Feature groups**: the 53 state features the pipeline will build,
   in 9 groups.
6. Set **family hint** = `DoS Hulk` (optional; sharpens the ATT&CK naming), leave
   *include explanations* checked, click **Run forecast →**.
7. You land on **SOC Dashboard**. **Observe:**
   - **Alert windows** tile ≈ `63 / 181` (red).
   - **Peak P(attack)** ≈ `97%`.
   - **Earliest lead time** ≈ `60 s` — the model raised the alarm a full window
     before the sustained flood.
   - **ATT&CK phases**: `Impact`.
   - **Anchor timeline** chart: a low, flat left half (the 10:20–10:35 lull) then a
     step up to ~0.85 that stays high — diamonds mark the alerting windows.
   - **Anchors table**: early rows `alert = —`, `peak P(atk)` ≈ 0.6–16 %; later
     rows `ALERT`, `first alert +1`, `lead time 60s`, `top tactic Impact`.
8. Click an **alerting** row → **Anchor detail**. **Observe:**
   - **Per‑horizon forecast** chart: solid green = system `detection_prob`, dashed
     blue = world‑model `attack_prob`, blue band = 95 % CI, red dashes = the alert
     threshold (~0.53). All six steps sit ~0.85.
   - **Horizon table** (the a11y pair): exact P per +10 s … +60 s, CI, progression
     state, tactic, `ALERT` flags.
   - **Kill‑chain progression** ribbon: `CONTINUATION → ACTIVE` across the six
     windows (colour‑coded, legend below).
   - **MITRE ATT&CK phase mapping**: six cards, tactic *Impact*, kill‑chain phase
     *Impact*, a confidence pill, `dominant_family: DoS Hulk`, a one‑line
     rationale.
   - **Driving features** (signed bars): `unique_dst_ports_per_src`,
     `payload_var_mean`, `failed_conn_ratio`, `dstport_entropy`, … — red bars push
     toward *attack*.
   - **Temporal saliency**: which of the 12 history windows the model leaned on.
9. Now repeat with `demo/demo_benign.csv`. **Observe the contrast:** ~223
   anchors, **0 alerts**, peak P(attack) ≈ 12 %, every progression cell `NORMAL`,
   tactic `None`. The model stays calm on real benign traffic — that's the point.

**Large files:** anything over `SENTINEL_MAX_SYNC_FLOWS` (20 000 flows) returns
`202 {job_id}` instead of an inline result; the console routes you to
**Pipeline** and polls `/jobs/{id}` until done, then to the dashboard.

---

## 4. Walkthrough B — Synthetic test‑bed (proves the real‑time path)

Zero setup. Streams generated scenarios through the **real** windowing + rollout
+ ATT&CK pipeline over a WebSocket.

1. Console → **Live Telemetry** → *Test‑bed (synthetic)* tab.
2. Pick a **scenario** (`portscan`, `dos_hulk`, `bruteforce`, `botnet_c2`,
   `exfil`, `benign`). Set **speed** = `30–50` (× real time — compresses the
   scenario clock so the timeline fills in seconds), leave rate/duration/seed.
3. Click **Start**. **Observe:**
   - Connection pill goes **green** ("streaming · synthetic · running").
   - `flows N` counter climbs fast.
   - After ~12 windows of compressed history (a few seconds at speed 40), the
     **Rolling P(attack)** chart starts drawing and **Incoming forecasts** cards
     appear.
   - **Latest kill‑chain progression** ribbon updates each window.
   - The **Top tactic** KPI reflects the scenario: `portscan` →
     *Reconnaissance / Discovery*, `dos_hulk` → *Impact*, `bruteforce` →
     *Credential Access*, `botnet_c2` → *Command and Control*, `exfil` →
     *Exfiltration*.
4. **Pause** freezes the view without dropping the socket; **Stop** deletes the
   session (the server also stops on its own at `duration_s`).
5. Click **open in dashboard →** to inspect any streamed anchor with the full
   detail panel (`/dashboard?session=<id>` — it re‑subscribes and accumulates).

> **Important caveat to state out loud.** The production model was trained on
> specific CIC‑IDS‑2017 statistical fingerprints and runs *hot* on
> out‑of‑distribution hand‑generated flows — a synthetic session's **absolute**
> P(attack) is not calibrated (benign and attack phases can both read ~0.99).
> What the test‑bed proves: the end‑to‑end **streaming path works** (ingest →
> 10 s windows → L=12 tensors → MC rollout → progression → ATT&CK → CI, pushed
> live), and the **ATT&CK / progression sequencing** responds to the scenario.
> For calibrated numbers, use the real‑CSV path (§3) or live capture on real
> traffic (§5).

CLI equivalent (no browser): `make synth-demo`, or
```bash
curl -s -X POST localhost:8000/live/sessions -H 'content-type: application/json' \
  -d '{"source":"synthetic","scenario":"portscan","speed":40,"duration_s":600}'
# then: python -m tests.live_smoke --api http://localhost:8000 --scenario portscan --speed 40
```

---

## 5. Walkthrough C — Live network capture ("select a network / IP like Wireshark")

Runs on the **host** whose traffic you want to watch — including a Huawei **eNSP**
topology (its cloud/bridge devices attach to local virtual adapters that show up
in the list).

### 5.1 Prerequisites
- **Windows:** install [Npcap](https://npcap.com/) (tick *WinPcap API‑compatible
  mode*). Open an **Administrator** PowerShell.
- **Linux/macOS:** run with `sudo` (or grant `cap_net_raw`).

### 5.2 Start the agent
```bash
cd capture-agent
sentinel-capture interfaces          # list NICs the way Wireshark's capture dialog does
sentinel-capture run --backend ws://localhost:8000 --name my-host
```
Leave it running. It connects to the backend and advertises its interface list.

### 5.3 Drive it from the console
1. Console → **Live Telemetry** → *Live capture* tab.
2. **Observe:** "agent **my-host** · N interfaces" and a radio list — name,
   description (e.g. *Intel(R) Wi‑Fi 6 AX201*), IPv4, up/down. (If it says *No
   capture agent connected*, the command to start one is shown.)
3. Pick an interface. Optionally set a **capture filter** (BPF) to scope it —
   this is the "select a network or IP" control:
   - `host 10.0.0.5` — one host
   - `net 192.168.56.0/24` — a subnet (e.g. an eNSP host‑only segment)
   - `tcp port 80 or tcp port 443` — web only
4. Click **Start**. **Observe:**
   - Connection pill green, `flows N` climbing as the agent assembles
     bidirectional flows from sniffed packets and streams them in.
   - Generate traffic on that interface (browse the web, `ping`, run your eNSP
     scenario). Forecasts begin once ~12 ten‑second windows exist (≈ 2 min of
     real time — capture is **not** sped up).
   - The dashboard behaves exactly as the CSV path: P(attack), progression,
     ATT&CK, CI.
5. **Stop** deletes the session and sends the agent `{"cmd":"stop"}`.

### 5.4 What the agent actually sends
Per flow, when it closes (FIN/RST) or times out (idle 15 s / active 120 s): the
11 required CICFlowMeter columns + `Source Port` + `flag_true_{fin,syn,rst,psh,
ack,urg}` + `Fwd/Bwd IAT Total` + `pkt_len_mean`. It is deliberately **not** a
full CICFlowMeter clone — only the fields the model consumes; the rest default
to 0, exactly as an uploaded CSV would.

---

## 6. A 10‑minute demo script

> Backend + console running (Option A or B). Have `demo/demo_dos_onset.csv` and
> `demo/demo_benign.csv` ready. Optionally have the capture agent running.

| # | Screen | Do | Say / point out |
|---|---|---|---|
| 1 | **Overview** | — | The problem: IDS is reactive, alerts have no trajectory, ATT&CK mapping is manual. The solution: a probabilistic **world model** that rolls network state 60 s forward. Point at the live **status tiles** (backend online, serve mode `system`, 12→6 horizon, 53 features). |
| 2 | **Architecture** | scroll | Four **decoupled** parts; the backend imports **no** research code — the model **bundle** is the only interface. Walk the "request lifecycle" and the endpoint table (`/forecast/csv`, `/live/sessions`, `WS /agent`). |
| 3 | **Model Card** | scroll | The layer table: Bi‑GRU encoder (232k params) → additive attention (103k) → **State‑Transition Network** μ/logσ (104k) → 3 heads. Forward = rollout; K‑step MC gives the CI band. 1.14 M params total with the 3 blend members. Read the **trust caveats** box. |
| 4 | **Data Sources** | upload `demo_dos_onset.csv` | The client **column‑match** report — required/aliased/Tier‑2/flag. "The backend re‑validates on upload." Set family hint `DoS Hulk`, **Run forecast**. |
| 5 | **Pipeline** (glance) | — | The five stages with live counts: ingest → clean (winsorize on **train‑day** stats) → 10 s state windows → L=12 scaled tensors → `simulate_anchor`. |
| 6 | **SOC Dashboard** | — | **63/181 alerts**, peak **97 %**, **60 s lead time**, phase **Impact**. The timeline: flat then a step‑up that holds. |
| 7 | **Anchor detail** | click an alerting row | The forecast chart (solid = system, dashed = world model, band = 95 % CI, red = threshold). The **progression ribbon** (CONTINUATION→ACTIVE). Six **ATT&CK cards** (Impact, rationale, family). **Driving features** — `unique_dst_ports_per_src`, `failed_conn_ratio`, `dstport_entropy`. |
| 8 | **Data Sources** | upload `demo_benign.csv`, run | The contrast: **0 alerts**, peak ~12 %, every cell `NORMAL`. "No false alarms on real benign traffic." |
| 9 | **Live Telemetry** | Test‑bed → `portscan`, speed 40, **Start** | Watch the timeline and cards stream in **real time**. Tactic flips to *Reconnaissance / Discovery*. State the OOD caveat — "this proves the streaming plumbing; the magnitudes here aren't calibrated." |
| 10 | **Live Telemetry** | Live capture → pick Wi‑Fi → **Start**, browse a site | (If Npcap is set up.) Real packets → flows → forecasts, same UI. "Same pipeline, live off the wire; point it at an eNSP subnet with a BPF filter." |

Fallback if no backend: every route still renders (health shows *offline*,
views show empty states) — good for a UI‑only walkthrough.

---

## 7. Screen reference — what each thing means

| element | meaning |
|---|---|
| **P(attack) / detection_prob** | system (world model **blended** with TCN+LSTM+GRU) probability that a window in the horizon is malicious. Solid line on charts. |
| **attack_prob (world model)** | the world model's own rollout probability. Dashed line. Blend adds *sharpness* only. |
| **95 % CI band** | spread over `mc_samples` reparameterised rollouts — model uncertainty. |
| **alert threshold** | best‑F1 point chosen on the **validation** split within an FPR budget (~0.53 here). `alert = detection_prob ≥ threshold` for any horizon step. |
| **first_alert_k / lead_time_seconds** | earliest horizon step that crosses the threshold, and that in seconds — the **warning time** before forecast onset. |
| **progression_state** | NORMAL · PRE_ATTACK · ONSET · ACTIVE · CONTINUATION — the kill‑chain stage per horizon step (softmax head). |
| **MITRE ATT&CK card** | `mitre_tactic`, `technique_ids`, `kill_chain_phase`, `confidence`, `dominant_family`, `rationale` — label‑free, from `attack_stages.assess_forecast`. |
| **driving_features** | signed gradient×input attribution on the newest window; red = toward attack. |
| **temporal saliency** | gradient / attention weight per history window (t‑11 … t‑0). |
| **anchor** | one scored position = the end of a 12‑window history; each anchor has a 6‑step horizon. |
| **serve_mode** | `system` (blend, needs the 3 members in the bundle) / `world_model` (WM only) / `auto`. |

---

## 8. API quick reference

```bash
B=http://localhost:8000

curl -s $B/health                      # bundle + live + agent + uptime
curl -s $B/meta   | python -m json.tool  # L/K/F, progression states, feature names, threshold
curl -s $B/models

# CSV forecast (sync if <= 20k flows, else 202 {job_id})
curl -s -X POST $B/forecast/csv -F file=@demo/demo_dos_onset.csv -F 'family_hint=DoS Hulk' -F explain=true

# async job
curl -s $B/jobs ; curl -s $B/jobs/<id> ; curl -s $B/jobs/<id>/result

# live session (synthetic)
curl -s -X POST $B/live/sessions -H 'content-type: application/json' \
  -d '{"source":"synthetic","scenario":"dos_hulk","speed":40,"duration_s":600}'
curl -s $B/live/sessions
curl -s -X DELETE $B/live/sessions/<id>

# subscribe to a live session (needs a WS client, e.g. wscat / websocat)
websocat ws://localhost:8000/live/sessions/<id>/stream

# capture agent status
curl -s $B/agent/status
```

WebSocket frames on `/live/sessions/{id}/stream`:
`{"type":"status", …}` → ring replay of `{"type":"forecast", …anchor…}` →
live `forecast` frames → `{"type":"bye"}` when the session ends. Send
`{"type":"stop"}` to end it from the client.

---

## 9. Run the tests

```bash
cd backend        && python -m pytest -q      # 47  — live spine, synth, /live, /agent, ops, CSV
cd capture-agent  && python -m pytest -q      #  8  — flow meter, interface enumeration, agent protocol
cd frontend       && npm test                 # 14  — schema/csv, api zod parsing, ws reconnect, panels
cd frontend       && npm run build && npx playwright test   # 8 — every route renders; a synthetic session streams
```
The Playwright synthetic‑session spec needs a backend on `:8000` (override with
`E2E_API=http://host:port`); it self‑skips when none is reachable.

---

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| topbar health dot **red / "offline"** | backend not reachable at the API base. Check the topbar URL / `?api=` / that `uvicorn` is up. CORS: backend `SENTINEL_CORS_ORIGINS` must allow the console origin (`*` by default). |
| `/health` **`status: "degraded"`, `model_loaded: false`** | no bundle at `backend/models/`. Build one (§1.1) or mount it: `-v ./backend/models:/app/models:ro`. |
| **port 8000 already in use** | another service. Run the backend on another port (`uvicorn app.main:app --port 8001`) and point the console at it. |
| CSV upload → **422 "missing required columns"** | not a CICFlowMeter/CIC‑IDS schema. The console's column‑match panel shows exactly which of the 11 required columns are absent. |
| CSV upload → **202 and it "hangs"** | file over 20 000 flows → async job. The Pipeline page shows the progress bar; it auto‑routes to the dashboard when done. |
| live session: **no forecasts ever** | needs ≥ 12 ten‑second windows of history. Synthetic: raise **speed**. Capture: wait ~2 min and make sure traffic is actually on the chosen interface / matches your BPF filter. |
| Live capture: **"No capture agent connected"** | the agent isn't running or can't reach the backend. Start it (§5.2); check `curl localhost:8000/agent/status`. |
| agent: **permission / "no such device"** | not elevated, or Npcap missing. Run the shell as Administrator; reinstall Npcap with WinPcap‑compatible mode. |
| synthetic session shows **P(attack) ≈ 0.99 for everything, even `benign`** | expected — the OOD calibration caveat (§4). Use real CSVs (§3) for meaningful magnitudes. |
| `docker compose up` frontend **unhealthy** briefly | nginx healthcheck `start_period`; it settles in a few seconds. |
