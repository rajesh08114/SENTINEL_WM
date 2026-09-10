# SENTINEL-WM — Live SOC Application (design)

**Date:** 2026-09-10
**Status:** approved (design), pre-implementation
**Branch:** `restructure-app`

## 1. Goal

Turn the SENTINEL-WM prototype into a production-grade SOC application with two new
real-time capabilities, and rebuild the frontend on a real framework.

1. **Frontend rebuild** — replace the vanilla ES-module SPA with **Next.js 14 (App
   Router) + TypeScript + Tailwind + shadcn/ui**, static-exported behind nginx.
2. **Synthetic test-bed** — a backend scenario generator that streams generated
   CICFlowMeter-schema flow rows through the *real* windowing + rollout pipeline so the
   live dashboard is exercised end-to-end. Selectable attack scenarios, adjustable rate
   and duration, reproducible by seed.
3. **Live local-network capture** — a separate privileged capture agent that enumerates
   local NICs (Wireshark-style, including Huawei eNSP virtual adapters), sniffs a chosen
   interface with an optional IP/CIDR (BPF) filter, assembles packets into bidirectional
   flow rows, and streams them to the backend, which forecasts in real time.
4. **Production hardening** — error/loading/empty states, input validation, tests on both
   sides, structured logging, health detail, clean Docker. Single-tenant; **no auth**
   (trusted local/LAN use).

Non-goals: authentication, multi-tenancy, persistent run history DB, PCAP file upload
(still returns 501), SSR/ISR, GPU-specific paths.

## 2. Architecture overview

```
                          ┌───────────────────────── backend (FastAPI, container) ──────────────────────────┐
 capture-agent (host,     │                                                                                 │
 admin + Npcap)           │   /agent  (WS)  ── AgentRegistry ──┐                                             │
   AsyncSniffer           │                                     │                                            │
   FlowMeter  ──flows──►  │   /live/sessions (REST + WS)  ── LiveManager ── LiveSession                      │
                          │                                       │            ├─ StreamingWindower          │
 synthetic scenario  ───► │   SyntheticSource ────────────────────┘            │     └─ simulate_anchor      │
 (in-process task)        │                                                    └─ forecast ring buffer ──► WS subscribers
                          │   /forecast/csv, /jobs, /stream, /meta, /models, /health   (unchanged)          │
                          └─────────────────────────────────────────────────────────────────────────────────┘
                                        ▲                                             ▲
                                        │ HTTP + WS (one origin)                      │
                          ┌─────────────┴───────────── frontend (Next.js static export, nginx) ─────────────┐
                          │  /  /architecture  /model  /sources  /pipeline  /dashboard  /live               │
                          └─────────────────────────────────────────────────────────────────────────────────┘
```

The capture agent is **not** in docker-compose — it needs host Npcap and Administrator
rights. It dials the backend outbound; the backend is the single API/control plane.
Synthetic and capture sources both feed one `LiveSession`; everything downstream of the
windower is the existing code path.

## 3. Backend — live session spine

New package `backend/app/live/`.

### 3.1 `session.py`

- `LiveSession`
  - fields: `id: str` (uuid4 hex), `source_kind: "synthetic" | "capture"`, `params: dict`,
    `state: "starting" | "running" | "stopping" | "stopped" | "error"`, `error: str | None`,
    `created_at`, `updated_at`, `stats: {flows_in, windows, forecasts, alerts}`.
  - owns a `StreamingWindower` (reused unchanged) and a `collections.deque(maxlen=200)`
    ring buffer of the most recent forecast dicts.
  - `subscribers: set[WebSocket]`.
  - `async def feed(self, rows: list[dict])` — push rows to the windower, call
    `poll_ready()` in a threadpool, append forecasts to the ring, fan out to subscribers
    as `{"type": "forecast", **fc}`, update stats.
  - `async def broadcast_status(self)` — send `{"type": "status", state, stats}`.
  - `async def stop(self)` — flush the windower, set state, fan out a final status, close
    subscriber sockets.
- `LiveManager`
  - `sessions: dict[str, LiveSession]`, `max_sessions` (default 4, from settings).
  - `create(source_kind, params) -> LiveSession` — raises `LiveCapacityError` past the cap.
  - `get(id)`, `list()`, `async remove(id)`.
  - `async reap_idle(max_idle_s=900)` — background task cancels sessions with no
    subscribers and no flow input past the timeout.
  - a module-level singleton `MANAGER`, created in `main.py` lifespan; reaper task
    started/stopped there.

### 3.2 `sources.py`

- `class SyntheticSource`
  - `__init__(session, scenario, rate, duration_s, seed, ip_hints)`.
  - `async def run(self)` — asyncio task. Each tick (250 ms) asks
    `synth.scenarios.emit(t, cfg)` for the rows due in that slice, calls
    `await session.feed(rows)`, sleeps to wall clock. Ends at `duration_s` (or on stop),
    then `await session.stop()`.
- `class AgentSource`
  - `__init__(session, agent_conn, iface, bpf)`.
  - registered as the consumer for one agent connection; `async def on_flows(rows)` →
    `await session.feed(rows)`. `async def start()` / `stop()` send the agent
    `{"cmd": "start", iface, bpf}` / `{"cmd": "stop"}`.
- Both are created and their tasks tracked by the `LiveSession` so `stop()` cancels them.

### 3.3 `routes_live.py`  (`APIRouter`, tag `live`)

| Method | Path | Body / behaviour |
|---|---|---|
| `POST` | `/live/sessions` | `{source:"synthetic", scenario, rate?, duration_s?, seed?, ip_hints?}` **or** `{source:"capture", iface, bpf?}`. Creates the session + source task. Returns `LiveSessionInfo`. `409` past the cap; `422` on bad params; `503` if `source:"capture"` and no agent connected. |
| `GET` | `/live/sessions` | list of `LiveSessionInfo`. |
| `GET` | `/live/sessions/{id}` | one, or `404`. |
| `DELETE` | `/live/sessions/{id}` | stop + remove. `404` if unknown. |
| `WS` | `/live/sessions/{id}/stream` | on connect: send `{"type":"status",...}` then replay the ring buffer as `{"type":"forecast",...}`; then stream live. Client may send `{"type":"stop"}`. Server closes with `{"type":"bye"}` when the session ends. |

### 3.4 Settings additions (`app/settings.py`)

`SENTINEL_LIVE_MAX_SESSIONS=4`, `SENTINEL_LIVE_IDLE_TIMEOUT_S=900`,
`SENTINEL_SYNTH_MAX_RATE=500` (flows/s ceiling), `SENTINEL_AGENT_ENABLED=true`.

## 4. Backend — synthetic test-bed

New package `backend/app/synth/`.

### 4.1 `scenarios.py`

- Constants: benign protocol mix, port pools, a small pool of internal/external IPs.
- `@dataclass ScenarioConfig` — `name`, `rate` (flows/s, clamped to `SYNTH_MAX_RATE`),
  `duration_s`, `seed`, `attacker_ip`, `victim_ip`, `phase_schedule`
  (list of `(t_start_frac, label)` marking `benign → pre_attack → onset → active →
  continuation`).
- `SCENARIOS: dict[str, Callable[[ScenarioConfig], ScenarioConfig]]` presets:
  - `benign` — baseline only.
  - `portscan` — one src → many dst ports on victim, SYN-heavy, 1–3 packet flows, ramp
    widens the port range and rate.
  - `dos_hulk` — one src → one victim:80, very high packet-rate flows, large fwd counts.
  - `bruteforce` — repeated short flows to victim:22/3389, high RST+FIN, small payloads.
  - `botnet_c2` — periodic small beacons internal → fixed external IP every ~10 s, plus
    occasional larger "task" flows.
  - `exfil` — sustained high `Total Length of Fwd Packets` internal → external over a few
    long flows.
- `def emit(t0: float, t1: float, cfg: ScenarioConfig, rng) -> list[dict]` — returns the
  flow rows whose `flow_start_epoch` falls in `[t0, t1)`. Each row has the 11 required
  columns + `flag_true_*` + `Label`/`attack_family` (display only) + a couple of Tier-2
  fields. Benign rows always present; attack rows added per the phase active at `t`.
- Determinism: all randomness from `random.Random(seed)`; `emit` is a pure function of
  its args.

### 4.2 Tests (`backend/tests/test_synth.py`)

- every preset: rows conform to `sentinel_infer` required columns; `flow_start_epoch`
  monotone within a slice; rate within ±20 % of target over a 10 s window.
- phase schedule: attack-family fraction is non-decreasing across the ramp for
  non-benign scenarios.
- seed reproducibility: same seed → identical rows.

## 5. Capture agent — `capture-agent/` (new top-level project)

`pyproject.toml` (`name = "sentinel-capture"`, deps: `scapy`, `psutil`, `websockets`,
`typer`), `sentinel_capture/` package. **Not** in docker-compose; documented as a host
process run from an Administrator shell with Npcap installed.

### 5.1 `interfaces.py`

- `list_interfaces() -> list[Interface]` — `Interface(name, description, ipv4, netmask,
  is_up, is_loopback, mac)`. Windows: `scapy.arch.windows.get_windows_if_list()` merged
  with `psutil.net_if_addrs()` / `net_if_stats()`. POSIX: `psutil` only. Virtual adapters
  (VirtualBox host-only, eNSP) appear naturally.

### 5.2 `flowmeter.py`

- `FlowKey` = (src, dst, sport, dport, proto), canonicalised so both directions map to
  one bidirectional flow with a recorded initiator (fwd) direction.
- `FlowState` accumulates: first/last ts, fwd/bwd packet counts, fwd/bwd byte totals,
  per-direction IAT samples, TCP flag counts (SYN/ACK/FIN/RST/PSH/URG), min/max packet
  lengths.
- `FlowMeter`
  - `add_packet(pkt, ts)` — update or create `FlowState`; on FIN/RST mark closing.
  - `harvest(now) -> list[dict]` — emit + drop flows that are closed, or idle >
    `idle_timeout` (15 s), or active > `active_timeout` (120 s). Row = CICFlowMeter schema:
    `flow_start_epoch, Source IP, Destination IP, Destination Port, Protocol,
    Flow Duration, Flow IAT Mean, Total Fwd Packets, Total Backward Packets,
    Total Length of Fwd Packets, Total Length of Bwd Packets`, `flag_true_*`, plus
    `Source Port`, `Fwd/Bwd IAT Total`, `pkt_len_mean`.
- Pure/synchronous; unit-tested with hand-built scapy packet lists.

### 5.3 `agent.py`

- `async def run(backend_ws_url, name)` — connect `WS {backend}/agent`, send
  `{"type":"hello", name, interfaces:[...]}`, then loop on commands:
  - `{"cmd":"start", iface, bpf}` → start `scapy.AsyncSniffer(iface=iface, filter=bpf,
    prn=meter.add_packet, store=False)`; every 1 s send
    `{"type":"flows", records: meter.harvest(now)}`.
  - `{"cmd":"stop"}` → stop sniffer, final `harvest`, `{"type":"stopped"}`.
  - `{"cmd":"interfaces"}` → re-send the list.
  - on socket drop: stop sniffer, reconnect with exponential backoff (1→30 s).

### 5.4 `cli.py`

- `sentinel-capture run --backend ws://localhost:8000 [--name host-01]`
- `sentinel-capture interfaces` — pretty table.

### 5.5 Backend side

- `app/agent/registry.py` — `AgentConnection` (ws, name, interfaces, current session id,
  `send_cmd`, `on_message`), `AgentRegistry` (`register`, `unregister`, `first()` /
  `by_name()`, `snapshot()` for the API). Module singleton created in lifespan.
- `app/api/routes_agent.py`
  - `WS /agent` — accept, register, pump messages: `flows` → the bound `AgentSource`;
    `hello`/`interfaces` → update registry; on disconnect unregister + stop any bound
    session.
  - `GET /agent/status` — `{connected: bool, agents: [{name, interfaces, session_id}]}`.
- `POST /live/sessions {source:"capture", iface, bpf}` picks `AgentRegistry.first()`,
  creates `AgentSource`, sends `start`. `503` if none connected.

### 5.6 Tests (`capture-agent/tests/`)

- `test_flowmeter.py` — SYN-scan burst → many short fwd-only flows with high SYN count;
  a full TCP conversation → one bidirectional flow with correct fwd/bwd counts and
  duration; idle-timeout eviction. Uses `scapy` packet construction, no NIC.
- `test_interfaces.py` — returns ≥ 1 interface, dataclass shape, loopback flagged.

## 6. Frontend — Next.js rebuild

Replace `frontend/`. Stack: **Next.js 14 App Router**, **TypeScript**, **Tailwind**,
**shadcn/ui** (Radix primitives), **@tanstack/react-query** (server state / polling),
**zustand** (UI + live-stream state), **recharts** (charts), **zod** +
**react-hook-form** (forms / schema), **lucide-react** (icons). `next.config.mjs` →
`output: "export"`; served by nginx. Config: `NEXT_PUBLIC_API_BASE` build-time default +
a runtime override persisted to `localStorage` and read on the client.

### 6.1 Structure

```
frontend/
  app/
    layout.tsx                 shell: topbar (API base, health, theme), sidenav
    page.tsx                   / overview
    architecture/page.tsx
    model/page.tsx
    sources/page.tsx
    pipeline/page.tsx
    dashboard/page.tsx
    live/page.tsx
  components/
    ui/                        shadcn primitives
    charts/ForecastChart.tsx   line + 95% CI band + threshold ref line
    charts/ProbTimeline.tsx    rolling P(attack)
    charts/FeatureBars.tsx  charts/Saliency.tsx
    panels/KpiTile.tsx  ProgressionRibbon.tsx  AttckPhaseTimeline.tsx  HorizonTable.tsx
    panels/AnchorTable.tsx  AnchorDetail.tsx
    live/SourcePicker.tsx  ScenarioForm.tsx  InterfacePicker.tsx  LiveControls.tsx
    sources/CsvWizard.tsx  ColumnMatchReport.tsx  DataInfo.tsx
    wizard/Stepper.tsx  common/{ErrorBoundary,EmptyState,Skeletons,ApiBaseField}.tsx
  lib/
    api.ts                     typed fetch client (zod-parsed responses)
    ws.ts                      reconnecting WS helper (live session + telemetry)
    schema.ts                  column contract + FEATURE_GROUPS (mirrors sentinel_infer)
    types.ts                   generated-ish TS types for the forecast payloads
    format.ts  store.ts (zustand)
  Dockerfile  nginx.conf  .env.example  README.md
  package.json  tsconfig.json  tailwind.config.ts  next.config.mjs
```

### 6.2 Pages

- **`/`** overview — problem/solution, "how the pieces connect", live system status
  (health, serve mode, L/K/F, agent connected, active sessions).
- **`/architecture`** — repo topology, request lifecycle, full endpoint table (incl. the
  new `/live` + `/agent` routes), deployment.
- **`/model`** — SENTINEL-WM (system) card: layer table + params, forward/backward,
  composition, trust caveats; L/K/F from `/meta`.
- **`/sources`** — CSV wizard (drop → zod-parsed header → column-match report →
  data-info → feature groups → **Run forecast**, sync or job-poll), PCAP (501 explainer),
  telemetry config snippet.
- **`/pipeline`** — ingest→clean→windows→sequences→forecast stepper with live counts
  from `result.meta` or the active live session; job progress bar.
- **`/dashboard`** — SOC view driven by either an uploaded result **or** the selected
  live session: alert KPI tiles, anchor table (select), `AnchorDetail`
  (`ForecastChart` + `HorizonTable`, `ProgressionRibbon`, `AttckPhaseTimeline`,
  `FeatureBars` + `Saliency`).
- **`/live`** — unified real-time view:
  - `SourcePicker`: **Test-bed** | **Live capture**.
  - Test-bed → `ScenarioForm` (scenario select, rate, duration, seed) → **Start**.
  - Live capture → `GET /agent/status`; if connected show `InterfacePicker`
    (name + description + IPv4, radio) + IP/CIDR filter field → **Start**; if not, show
    the one-line command to launch the agent.
  - Running: `LiveControls` (pause render / stop session), connection pill,
    `ProbTimeline`, streaming anchor cards, `ProgressionRibbon`, `AttckPhaseTimeline`.
  - Uses `ws.ts` against `/live/sessions/{id}/stream`; "Open in dashboard" deep-links to
    `/dashboard?session={id}`.

### 6.3 Quality

Error boundaries per route; `<Skeleton>` while queries load; explicit empty states
("no forecast yet", "agent not connected"); toasts for failures; all interactive
elements keyboard-reachable (shadcn/Radix); charts paired with a data table; honour
`prefers-reduced-motion`; dark-committed with a working light toggle.

### 6.4 Tests

- **Vitest + React Testing Library**: `ColumnMatchReport` (missing/aliased/tier2),
  `ProgressionRibbon` (state → colour), `api.ts` zod parsing (accept good, reject
  malformed), `ws.ts` reconnect/backoff with a mock socket.
- **Playwright** (`frontend/e2e/`): every route renders with the backend stubbed;
  starting a synthetic session shows a streamed forecast card within a timeout.

### 6.5 Docker

Multi-stage: `node:20-alpine` (`npm ci` → `next build` → static `out/`) → `nginx:alpine`
serving `out/` with SPA fallback + module MIME. Health via `wget` on `/`.

## 7. Polish & ops

- **Logging** (`app/logging.py`): stdlib `logging.config.dictConfig`; text by default,
  JSON when `SENTINEL_LOG_JSON=true`. Request-id middleware adds `X-Request-ID` and binds
  it to a `ContextVar` included in every log record.
- **`/health`** returns `{status, bundle:{loaded, serve_mode, L, K, n_features},
  live:{sessions, max}, agent:{connected, count}, uptime_s}`.
- **`/metrics`** (optional, `SENTINEL_METRICS=true`) via `prometheus-client`:
  forecast latency histogram, active sessions gauge, flows-in counter.
- **docker-compose.yml**: `backend` + `frontend`; capture-agent documented, not a
  service. `Makefile`: `synth-demo` (curl a synthetic session), `capture-agent`
  (`cd capture-agent && python -m sentinel_capture.cli run`), `frontend-dev`,
  `frontend-build`.
- Docs: root `README` (new §Live + §Capture agent), `docs/technical_reference.md`
  (live pipeline + synth scenarios + flowmeter), `capture-agent/README.md`,
  `frontend/README.md`.

## 8. Implementation phases

Each phase is independently testable and leaves the app working.

1. **Live spine + synthetic test-bed** — `app/live/`, `app/synth/`, `routes_live`,
   settings, wire into `main.py` lifespan, `test_synth.py` + `test_live_session.py`.
   Verify: `POST /live/sessions {source:synthetic,...}` then subscribe the WS → forecast
   frames stream; `DELETE` stops cleanly. (Exercisable with `curl`/`wscat` before the
   new frontend exists.)
2. **Capture agent + agent API** — `capture-agent/` project, `app/agent/`,
   `routes_agent`, `AgentSource`, flowmeter + interface tests. Verify: run the agent
   against a running backend, `GET /agent/status` shows the NIC list; a capture session
   on loopback while generating local traffic streams forecasts.
3. **Next.js frontend** — scaffold, port `/`,`/architecture`,`/model`,`/sources`,
   `/pipeline`,`/dashboard`, build the new `/live`, `lib/*`, component library, Vitest +
   Playwright, Dockerfile + nginx. Verify: `next build` static-exports; e2e smoke green;
   synthetic + capture sessions render live.
4. **Polish & ops** — logging, `/health` detail, optional `/metrics`, compose, Makefile,
   `.env.example`, docs. Verify: `docker compose up --build` serves backend :8000 +
   frontend :8080; `pytest backend/` + `pytest capture-agent/` green.

## 9. Risks / mitigations

- **Npcap / privilege friction** — agent is isolated and optional; synthetic test-bed
  proves the pipeline with zero setup; agent README covers Npcap + "Run as
  Administrator" + the eNSP host-only adapter.
- **Flowmeter fidelity** — a full CICFlowMeter reimplementation is out of scope; we emit
  the columns the model actually consumes (required + flags + a few Tier-2) and default
  the rest to 0, exactly as `normalise_upload` already tolerates.
- **Synthetic-traffic calibration** *(confirmed during Phase 1)* — the production world
  model was trained on specific CIC-IDS-2017 fingerprints and runs hot near saturation
  (see the F1 / threshold analysis). Hand-generated flows are out-of-distribution, so the
  *absolute* P(attack) of a synthetic session is not meaningful — benign and attack
  phases can both read ~0.99. The test-bed still delivers its purpose: it drives the real
  streaming → windowing → rollout → progression → ATT&CK → CI path end to end and shows
  the relative narrative. The `/live` UI and docs state this limitation plainly. Genuine
  calibration would need replaying real captures (deferred) or retraining with a
  synthetic-benign channel.
- **Static export vs shadcn** — all data is client-fetched; no server components need
  runtime, so `output: "export"` holds. If a route handler is ever needed we switch to
  `output: "standalone"` (one-line change + Node in the image).
- **WS fan-out load** — ring buffer bounded at 200; `LiveManager` caps concurrent
  sessions (default 4) and reaps idle ones.
- **Frontend rebuild scope** — porting is mechanical (logic already exists in the
  vanilla version and `lib/schema.js`); Phase 3 reuses those as the reference.
