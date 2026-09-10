# Live SOC Application — Implementation Plan

> **For agentic workers:** implement task-by-task. Steps use `- [ ]` checkboxes. Each
> task ends with an independently testable deliverable and a commit. TDD where a unit has
> logic worth pinning; smoke/integration checks where it's glue.

**Goal:** Add real-time forecasting to SENTINEL-WM — a synthetic-scenario test-bed and a
live local-network capture agent — and rebuild the frontend on Next.js, to a robust
single-tenant production bar.

**Architecture:** A backend `LiveSession` spine reuses the existing
`StreamingWindower → simulate_anchor` path; two sources feed it — an in-process synthetic
scenario generator and an outbound capture agent that assembles sniffed packets into
flow rows. The frontend is a static-exported Next.js app talking to the one FastAPI
origin.

**Tech Stack:** FastAPI · asyncio · scapy + psutil (agent) · Next.js 14 App Router ·
TypeScript · Tailwind · shadcn/ui · TanStack Query · Zustand · Recharts · zod · Vitest ·
Playwright.

**Spec:** `docs/superpowers/specs/2026-09-10-live-soc-application-design.md`

## Global Constraints

- Backend stays self-contained: **no `sentinel_wm` import**; inference code is only what's
  under `backend/app/sentinel_infer/`.
- `numpy<2.0` (bundle unpickle). Python `>=3.10`.
- Flow rows everywhere use the exact CICFlowMeter column names in
  `backend/app/sentinel_infer/forecast.py` (`REQUIRED`, `ALIASES`, `_FLAG_SRC`).
- Frontend: Next.js `output: "export"` (static); no Node server in prod. One API origin
  from `NEXT_PUBLIC_API_BASE` + a `localStorage` runtime override.
- Capture agent is a **host** process (Npcap + Administrator); never a docker-compose
  service.
- Single-tenant, **no auth**.
- Commit after every task. Branch: `restructure-app`.
- Attribution on every commit: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

## Phase 1 — Live spine + synthetic test-bed

### Task 1.1 — Settings + `app/live` package skeleton

**Files:**
- Modify: `backend/app/settings.py` — add `live_max_sessions:int=4`,
  `live_idle_timeout_s:int=900`, `synth_max_rate:int=500`, `agent_enabled:bool=True`
  (env `SENTINEL_LIVE_MAX_SESSIONS`, `SENTINEL_LIVE_IDLE_TIMEOUT_S`,
  `SENTINEL_SYNTH_MAX_RATE`, `SENTINEL_AGENT_ENABLED`).
- Create: `backend/app/live/__init__.py`
- Create: `backend/app/live/errors.py` — `LiveCapacityError`, `LiveNotFound`,
  `AgentUnavailable` (all `Exception` subclasses).

**Interfaces produced:** the four settings fields; the three exception classes.

- [x] Add settings fields with defaults + env aliases; confirm `get_settings()` still
      imports (`python -c "from app.settings import get_settings; get_settings()"`).
- [x] Create the package + `errors.py`.
- [x] Commit: `feat(live): settings + errors scaffold`.

### Task 1.2 — `synth/scenarios.py` (pure generator) + tests

**Files:**
- Create: `backend/app/synth/__init__.py`
- Create: `backend/app/synth/scenarios.py`
- Test: `backend/tests/test_synth.py`

**Interfaces produced:**
- `@dataclass ScenarioConfig(name:str, rate:float, duration_s:float, seed:int,
  attacker_ip:str, victim_ip:str, phase_schedule:list[tuple[float,str]])`
- `SCENARIOS: dict[str, ScenarioConfig]` — keys `benign portscan dos_hulk bruteforce
  botnet_c2 exfil` (each a ready default config; `rate`/`duration_s`/`seed` overridable
  by the caller).
- `def build_config(name:str, *, rate=None, duration_s=None, seed=None,
  attacker_ip=None, victim_ip=None) -> ScenarioConfig` — clamps `rate` to
  `settings.synth_max_rate`, raises `KeyError` for an unknown name.
- `def emit(t0:float, t1:float, cfg:ScenarioConfig, rng:random.Random) -> list[dict]` —
  flow rows with `flow_start_epoch ∈ [t0,t1)`, benign baseline always, attack rows per
  the phase active at `t`. Row keys: the 11 `REQUIRED` cols + `Source Port` +
  `flag_true_{fin,syn,rst,psh,ack,urg}` + `Label` + `attack_family` +
  `pkt_len_mean` + `ttl_mean`.
- `def phase_at(cfg, frac:float) -> str`.

- [x] **Test first** (`test_synth.py`): for each name in `SCENARIOS` —
      `rows = emit(0, 10, build_config(name, seed=1), Random(1))`;
      assert every row has all `REQUIRED` keys (import the list from
      `app.sentinel_infer.forecast`); assert `10*rate*0.8 <= len(rows) <= 10*rate*1.2`;
      assert `flow_start_epoch` values sorted and within `[0,10)`.
- [x] **Test**: non-benign scenarios — attack-family fraction over `emit(0, duration)`
      is non-decreasing across three equal time thirds.
- [x] **Test**: `emit` with the same seed twice → identical rows (`==`).
- [x] **Test**: `build_config("portscan", rate=99999).rate == settings.synth_max_rate`;
      `build_config("nope")` raises `KeyError`.
- [x] Run: `pytest backend/tests/test_synth.py -v` → all fail (module missing).
- [x] Implement `scenarios.py`: benign generator (random 5-tuples from internal→external
      pools, TCP/UDP mix, realistic small durations/counts), then per-scenario attack
      row builders keyed off `phase_at`. All randomness via the passed `rng`.
- [x] Run the file → green.
- [x] Commit: `feat(synth): deterministic scenario flow generator`.

### Task 1.3 — `live/session.py` (`LiveSession` + `LiveManager`) + tests

**Files:**
- Create: `backend/app/live/session.py`
- Test: `backend/tests/test_live_session.py`

**Interfaces consumed:** `StreamingWindower` from `app.streaming.windower`
(`add_flows(records)->int`, `poll_ready()->list[dict]`, `flush()->list[dict]`, `stats`).
`settings.live_max_sessions`, `settings.live_idle_timeout_s`.

**Interfaces produced:**
- `class LiveSession` — attrs `id:str`, `source_kind:str`, `params:dict`,
  `state:str`, `error:str|None`, `stats:dict`, `ring:collections.deque` (maxlen 200),
  `subscribers:set`. Methods:
  `async feed(rows:list[dict]) -> None`,
  `async subscribe(ws) -> None` (adds, sends status + ring replay),
  `unsubscribe(ws)`, `async stop() -> None`, `info() -> dict`,
  `attach_task(t:asyncio.Task)`.
- `class LiveManager` — `create(source_kind:str, params:dict) -> LiveSession`
  (raises `LiveCapacityError`), `get(id) -> LiveSession` (raises `LiveNotFound`),
  `list() -> list[dict]`, `async remove(id) -> None`, `async reap_idle() -> None`,
  `async shutdown() -> None`.
- module singleton `MANAGER = LiveManager()`.

Notes: `feed` runs `poll_ready` via `asyncio.to_thread`; fans out
`{"type":"forecast", **fc}` to subscribers, dropping any that raise; updates
`stats={flows_in,windows,forecasts,alerts}`; touches `last_activity`.

- [x] **Test** (`test_live_session.py`, `asyncio_mode=auto` already set): a fake windower
      (`add_flows` records count, `poll_ready` returns one canned forecast dict once ≥N
      rows). `s = LiveManager().create("synthetic", {})`; `await s.feed([{...}]*N)`;
      assert `s.stats["forecasts"] == 1` and `len(s.ring) == 1`.
- [x] **Test**: a dummy subscriber object with `async send_json` capturing calls;
      `await s.subscribe(sub)` → first two messages are `type=="status"` then the ring
      replay; a subsequent `feed` that produces a forecast → `sub` gets
      `type=="forecast"`.
- [x] **Test**: `create` past `live_max_sessions` raises `LiveCapacityError`;
      `get("bad")` raises `LiveNotFound`; `await remove(id)` then `get` raises.
- [x] **Test**: `await s.stop()` sets `state=="stopped"`, calls windower `flush`, sends a
      final status, and cancels an attached dummy task.
- [x] Run → fail; implement `session.py`; run → green.
- [x] Commit: `feat(live): LiveSession + LiveManager`.

### Task 1.4 — `live/sources.py` (`SyntheticSource`) + test

**Files:**
- Create: `backend/app/live/sources.py`
- Test: `backend/tests/test_live_sources.py`

**Interfaces consumed:** `LiveSession.feed`, `synth.scenarios.build_config`/`emit`.

**Interfaces produced:**
- `class SyntheticSource` — `__init__(session:LiveSession, cfg:ScenarioConfig)`;
  `async run() -> None` (250 ms ticks, wall-clock paced, `emit` the due slice,
  `await session.feed(rows)`, end at `cfg.duration_s` then `await session.stop()`);
  `request_stop() -> None`.
- `class AgentSource` — `__init__(session, agent, iface:str, bpf:str|None)`;
  `async on_flows(rows:list[dict]) -> None` → `session.feed`; `async start()` /
  `async stop()` send the agent `{"cmd":"start"/"stop", ...}`. (Agent wired in Phase 2;
  define the class now, its `start/stop` no-op if `agent is None`.)

- [x] **Test**: `cfg = build_config("portscan", rate=20, duration_s=1, seed=3)`;
      `src = SyntheticSource(fake_session, cfg)`; `await asyncio.wait_for(src.run(), 3)`;
      assert `fake_session.fed_rows > 0` and `fake_session.stopped is True`.
- [x] **Test**: `request_stop()` mid-run → `run()` returns promptly, session stopped.
- [x] Run → fail; implement; run → green.
- [x] Commit: `feat(live): SyntheticSource + AgentSource shell`.

### Task 1.5 — `api/routes_live.py` + wire into `main.py`

**Files:**
- Create: `backend/app/api/routes_live.py`
- Modify: `backend/app/main.py` — include the router; in lifespan start
  `MANAGER.reap_idle` loop task and `await MANAGER.shutdown()` on exit.
- Modify: `backend/app/schemas.py` — `LiveSessionCreate`, `LiveSessionInfo`.
- Test: `backend/tests/test_routes_live.py`

**Interfaces consumed:** `MANAGER`, `SyntheticSource`, `build_config`.

**Endpoints:** exactly §3.3 of the spec.
- `POST /live/sessions` → validate body (`source` in {synthetic,capture}); for synthetic
  build cfg (422 on bad scenario/params), `MANAGER.create`, spawn `SyntheticSource.run`
  task, `attach_task`, return `LiveSessionInfo` (201). `409` on `LiveCapacityError`.
  For capture: `503` (`AgentUnavailable`) until Phase 2.
- `GET /live/sessions`, `GET /live/sessions/{id}` (404), `DELETE /live/sessions/{id}`
  (404 / 204).
- `WS /live/sessions/{id}/stream` — `await session.subscribe(ws)`; loop receiving
  (`{"type":"stop"}` → `MANAGER.remove`); on disconnect `unsubscribe`.

- [x] **Test** (httpx ASGITransport + the `tiny_bundle`/`client` fixtures in
      `conftest.py`): `POST /live/sessions {"source":"synthetic","scenario":"portscan",
      "rate":20,"duration_s":1,"seed":1}` → 201 with an `id`; `GET /live/sessions` lists
      it; `DELETE` → 204; `GET /live/sessions/{id}` → 404.
- [x] **Test**: `POST` with `"scenario":"nope"` → 422; with `"source":"capture"` → 503.
- [x] **Test** (WS): connect `/live/sessions/{id}/stream` for a short synthetic session;
      assert the first frame is `type=="status"` and at least one `type=="forecast"`
      frame arrives within a timeout (bundle produces forecasts on random weights — assert
      on frame *shape*, not values).
- [x] Run → fail; implement router + schemas + main wiring; run → green.
- [x] Run the whole backend suite: `pytest backend/ -q` → green.
- [x] Commit: `feat(api): /live sessions (synthetic source end to end)`.

### Task 1.6 — Manual verification harness

**Files:**
- Create: `backend/tests/live_smoke.py` — a `python -m` script: start the app with
  uvicorn in-process (or hit a running one), `POST` a synthetic session, connect the WS,
  print streamed forecast summaries for ~15 s, `DELETE`.
- Modify: `Makefile` — `synth-demo:` runs it.

- [x] Implement the script; run against `uvicorn app.main:app` with the real bundle if
      present, else the tiny bundle; confirm forecast frames print.
- [x] Commit: `chore(live): synth-demo smoke script`.

---

## Phase 2 — Capture agent + agent API

### Task 2.1 — `capture-agent/` project skeleton

**Files:**
- Create: `capture-agent/pyproject.toml` (`name="sentinel-capture"`, deps `scapy>=2.5`,
  `psutil>=5.9`, `websockets>=12`, `typer>=0.9`; `dev` = `pytest`), `requirements.txt`
  (`-e .`).
- Create: `capture-agent/sentinel_capture/__init__.py` (`__version__`).
- Create: `capture-agent/README.md` — Npcap install, "Run as Administrator", eNSP
  host-only adapter note, usage.
- Create: `capture-agent/.gitignore`.

- [x] Scaffold; `pip install -e ./capture-agent` in a scratch venv or confirm metadata
      parses (`python -c "import tomllib,sys; tomllib.load(open('capture-agent/pyproject.toml','rb'))"`).
- [x] Commit: `feat(agent): capture-agent project skeleton`.

### Task 2.2 — `interfaces.py` + tests

**Files:**
- Create: `capture-agent/sentinel_capture/interfaces.py`
- Test: `capture-agent/tests/test_interfaces.py`

**Interfaces produced:**
- `@dataclass Interface(name:str, description:str, ipv4:str|None, netmask:str|None,
  mac:str|None, is_up:bool, is_loopback:bool)`
- `def list_interfaces() -> list[Interface]` — psutil `net_if_addrs()`+`net_if_stats()`;
  on Windows also merge `scapy.arch.windows.get_windows_if_list()` for friendly
  descriptions (guard the import; degrade to psutil names).
- `def to_dicts(ifaces) -> list[dict]`.

- [x] **Test**: `ifaces = list_interfaces()`; assert `len(ifaces) >= 1`, each is an
      `Interface`, at least one `is_loopback`, `to_dicts` round-trips keys.
- [x] Run → fail; implement; run → green.
- [x] Commit: `feat(agent): NIC enumeration`.

### Task 2.3 — `flowmeter.py` + tests (the core new logic)

**Files:**
- Create: `capture-agent/sentinel_capture/flowmeter.py`
- Test: `capture-agent/tests/test_flowmeter.py`

**Interfaces produced:**
- `FlowKey = tuple` (canonical bidirectional 5-tuple + a stored `fwd` orientation).
- `class FlowMeter(idle_timeout=15.0, active_timeout=120.0)`:
  - `add_packet(pkt, ts:float|None=None) -> None` — accepts a scapy packet; updates or
    creates flow state; tracks fwd/bwd packet & byte counts, per-dir IAT samples, TCP
    flag counts, first/last ts; marks closing on FIN/RST.
  - `harvest(now:float) -> list[dict]` — emit + evict flows that are closed / idle /
    over active timeout. Row = the 11 `REQUIRED` cols + `Source Port` +
    `flag_true_*` + `Fwd IAT Total` + `Bwd IAT Total` + `pkt_len_mean`.
  - `pending() -> int`.

- [x] **Test**: build a SYN scan — 20 `IP()/TCP(flags="S", dport=n)` packets, distinct
      dports, one src → `add_packet` each → `harvest(later)` → ≥ ~20 rows, each
      `flag_true_syn == 1`, `Total Backward Packets == 0`, small `Flow Duration`.
- [x] **Test**: a full TCP conversation — SYN, SYN-ACK, ACK, 2 data each way, FIN/FIN-ACK
      → exactly **one** row; `Total Fwd Packets`/`Total Backward Packets` match; row
      appears in the `harvest` right after the FIN (closed, not waiting for timeout).
- [x] **Test**: idle eviction — one packet at t=0, `harvest(10)` → 0 rows (still
      pending), `harvest(20)` → 1 row.
- [x] **Test**: UDP flow — `IP()/UDP()` both directions → one row, `Protocol == 17`.
- [x] Run → fail; implement; run → green.
- [x] Commit: `feat(agent): bidirectional flow meter`.

### Task 2.4 — `agent.py` (backend WS client) + `cli.py`

**Files:**
- Create: `capture-agent/sentinel_capture/agent.py`
- Create: `capture-agent/sentinel_capture/cli.py`
- Test: `capture-agent/tests/test_agent_protocol.py` (protocol only, mock socket — no
  real sniff)

**Interfaces produced:**
- `async def run_agent(backend_ws_url:str, name:str, *, connect=websockets.connect) -> None`
  — connect, send `{"type":"hello", name, interfaces:[...]}`, then dispatch commands:
  `start{iface,bpf}` → `AsyncSniffer(iface, filter=bpf, prn=meter.add_packet,
  store=False)` + a 1 s `harvest`→`{"type":"flows","records":[...]}` loop;
  `stop` → stop sniffer, final harvest, `{"type":"stopped"}`;
  `interfaces` → resend. Reconnect with 1→30 s backoff on drop.
- `cli.py`: `typer` app — `run(--backend, --name)`, `interfaces()` (prints a table).
  `if __name__ == "__main__"` + `[project.scripts] sentinel-capture = "sentinel_capture.cli:app"`.

- [x] **Test**: a fake async context-manager socket yielding a scripted
      `[hello-ack?, {"cmd":"interfaces"}, {"cmd":"stop"}]`; run `run_agent` with
      `connect=` the fake and `AsyncSniffer` monkey-patched to a no-op; assert the agent
      sent a `hello` with a non-empty `interfaces` list and responded to `interfaces`.
- [x] Run → fail; implement `agent.py` + `cli.py`; run → green.
- [x] `python -m sentinel_capture.cli interfaces` prints a table locally.
- [x] Commit: `feat(agent): backend WS client + CLI`.

### Task 2.5 — Backend `app/agent/` registry + `routes_agent.py`

**Files:**
- Create: `backend/app/agent/__init__.py`, `backend/app/agent/registry.py`
- Create: `backend/app/api/routes_agent.py`
- Modify: `backend/app/main.py` — include router; `REGISTRY` singleton in lifespan.
- Modify: `backend/app/live/sources.py` — `AgentSource.start/stop` now send real
  commands via the bound `AgentConnection`.
- Modify: `backend/app/api/routes_live.py` — `source:"capture"` path: pick
  `REGISTRY.first()` (503 if none), create `AgentSource`, `await src.start()`.
- Test: `backend/tests/test_routes_agent.py`

**Interfaces produced:**
- `class AgentConnection(ws, name, interfaces:list[dict])` — `async send_cmd(d:dict)`,
  `bind(session_id, on_flows)`, `unbind()`, `snapshot()->dict`.
- `class AgentRegistry` — `register(conn)`, `unregister(conn)`, `first()->AgentConnection|None`,
  `snapshot()->dict`. Singleton `REGISTRY`.
- `WS /agent` — accept, read `hello`, `REGISTRY.register`; loop: `flows` → bound
  `on_flows`; `interfaces` update; on disconnect `unregister` + stop bound session.
- `GET /agent/status` → `{"connected":bool,"agents":[snapshot,...]}`.

- [x] **Test**: connect a test WS client to `/agent`, send `hello` with 1 interface;
      `GET /agent/status` → `connected true`, agent listed.
- [x] **Test**: with that agent connected, `POST /live/sessions {"source":"capture",
      "iface":"lo"}` → 201; the agent client receives a `{"cmd":"start","iface":"lo"}`;
      pushing a `{"type":"flows","records":[...12 windows...]}` yields a forecast frame
      on the live WS. `DELETE` → agent gets `{"cmd":"stop"}`.
- [x] **Test**: no agent → `POST .../capture` → 503.
- [x] Run → fail; implement; run → `pytest backend/ -q` green.
- [x] Commit: `feat(agent): backend registry + /agent WS + capture sessions`.

### Task 2.6 — Agent integration check + docs

**Files:**
- Modify: `Makefile` — `capture-agent:` target
  (`cd capture-agent && python -m sentinel_capture.cli run --backend ws://localhost:8000`).
- Modify: `capture-agent/README.md`, root `README.md` (new §"Live capture agent").

- [x] Run a real end-to-end locally: `uvicorn app.main:app` + `make capture-agent`,
      `GET /agent/status` shows NICs; start a capture session on the loopback while
      `ping`ing localhost; confirm forecast frames stream. Record the result in the
      commit message.
- [x] Commit: `docs(agent): capture-agent usage + Makefile target`.

---

## Phase 3 — Next.js frontend

### Task 3.1 — Scaffold + tooling

**Files:**
- Move current `frontend/*` → `frontend/_legacy/` (keep as porting reference; deleted in
  Task 3.9).
- Create: `frontend/package.json`, `tsconfig.json`, `next.config.mjs`
  (`output:"export"`, `images.unoptimized:true`), `tailwind.config.ts`,
  `postcss.config.mjs`, `app/globals.css`, `.eslintrc.json`, `.gitignore`,
  `.env.example` (`NEXT_PUBLIC_API_BASE=http://localhost:8000`).
- Create: `frontend/app/layout.tsx` (shell), `frontend/app/page.tsx` (placeholder).
- Create: `frontend/components/ui/` — shadcn primitives actually used (button, card,
  input, select, tabs, table, badge, dialog, tooltip, skeleton, sonner/toast).
- Create: `frontend/lib/utils.ts` (`cn`).

- [x] `npm install`; `npm run build` (static export of the placeholder) succeeds →
      `frontend/out/index.html` exists.
- [x] `npm run lint` clean.
- [x] Commit: `feat(web): Next.js scaffold + Tailwind + shadcn primitives`.

### Task 3.2 — `lib/` core: types, api client, ws helper, schema mirror

**Files:**
- Create: `frontend/lib/types.ts` — `HorizonStep`, `AnchorForecast`, `ForecastResponse`,
  `MetaResponse`, `HealthResponse`, `JobStatus`, `LiveSessionInfo`, `AgentStatus`,
  `Interface`.
- Create: `frontend/lib/schema.ts` — `REQUIRED`, `ALIASES`, `TIER2_OPTIONAL`,
  `FLAG_FROM`, `PROGRESSION_STATES`, `FEATURE_GROUPS`, `matchColumns(headers)`
  (port from `_legacy/lib/schema.js`).
- Create: `frontend/lib/csv.ts` — `parseCSV`, `summarise` (port from `_legacy`).
- Create: `frontend/lib/api.ts` — `apiBase()` (env + `localStorage` override),
  `setApiBase`, typed `get/postForm`, zod schemas parsing each response,
  `health/meta/models/forecastCsv/jobs/job/jobResult/liveSessions/createLiveSession/
  deleteLiveSession/agentStatus`.
- Create: `frontend/lib/ws.ts` — `openLiveStream(sessionId, {onForecast,onStatus,onClose})`
  returning `{close}`, with reconnect + backoff.
- Create: `frontend/lib/store.ts` — zustand: `apiBase`, `csv`, `result`, `selectedAnchor`,
  `liveSessionId`, `liveForecasts`, `liveStatus`, setters.
- Create: `frontend/lib/format.ts` — `pct/num/sec/round`.
- Test: `frontend/lib/__tests__/schema.test.ts`, `api.test.ts`, `ws.test.ts`
  (Vitest; jsdom for ws mock).

- [x] **Test** `matchColumns`: a good header → `ok:true`; a header missing
      `Destination Port` → in `requiredMiss`; a `dst port` header → `aliased`.
- [x] **Test** `api.ts`: zod parse accepts a valid `ForecastResponse` fixture, throws on
      one with `horizon` missing.
- [x] **Test** `ws.ts`: mock `WebSocket`; simulate `onclose` → helper retries after the
      backoff; `close()` stops retries.
- [x] Add `vitest.config.ts` + `"test"` script; run → green.
- [x] Commit: `feat(web): typed api client, ws helper, schema mirror`.

### Task 3.3 — Shell: layout, topbar, sidenav, providers

**Files:**
- Modify: `frontend/app/layout.tsx` — `<QueryClientProvider>`, `<ThemeProvider>`,
  `<Toaster>`, topbar (`ApiBaseField` + health dot from `useQuery(health)` + theme
  toggle), sidenav (7 links), `<ErrorBoundary>` around `{children}`.
- Create: `frontend/components/common/ApiBaseField.tsx`, `ThemeToggle.tsx`,
  `HealthDot.tsx`, `ErrorBoundary.tsx`, `EmptyState.tsx`, `Skeletons.tsx`.
- Create: `frontend/components/nav/SideNav.tsx`, `TopBar.tsx`.

- [x] `npm run build` static-exports; manual: `npx serve out`, all 7 nav links resolve
      to a page (placeholders OK), health dot reflects backend up/down.
- [x] Commit: `feat(web): app shell — topbar, sidenav, providers`.

### Task 3.4 — Static content pages: `/`, `/architecture`, `/model`

**Files:**
- Modify: `frontend/app/page.tsx`; create `frontend/app/architecture/page.tsx`,
  `frontend/app/model/page.tsx`.
- Create: `frontend/components/panels/KpiTile.tsx`, `common/Prose.tsx`.
- Content ported + expanded from `_legacy/views/{overview,architecture,model}.js`;
  `/` and `/model` read `useQuery(meta)` for live L/K/F, serve mode, agent/session
  counts; `/architecture` endpoint table includes `/live/*` and `/agent/*`.

- [x] Build; manual check all three render with and without a backend.
- [x] Commit: `feat(web): overview, architecture, model pages`.

### Task 3.5 — Charts + shared panels

**Files:**
- Create: `frontend/components/charts/ForecastChart.tsx` (Recharts `ComposedChart`:
  `Area` CI band from `attack_ci`, `Line` `detection_prob` solid, `Line` `attack_prob`
  dashed, `ReferenceLine` at the alert threshold), `ProbTimeline.tsx`,
  `FeatureBars.tsx` (signed horizontal bars), `Saliency.tsx` (grouped bars).
- Create: `frontend/components/panels/ProgressionRibbon.tsx`, `AttckPhaseTimeline.tsx`,
  `HorizonTable.tsx`, `AnchorTable.tsx`, `AnchorDetail.tsx`.
- Test: `frontend/components/__tests__/ProgressionRibbon.test.tsx` (state→class),
  `HorizonTable.test.tsx` (6 rows, ALERT when `detection_prob ≥ threshold`).

- [x] **Tests** green (Vitest + RTL).
- [x] Storybook-free visual check: a `/dashboard` fed a fixture `ForecastResponse`
      renders chart + ribbon + ATT&CK timeline + tables.
- [x] Commit: `feat(web): forecast charts + SOC panels`.

### Task 3.6 — `/sources` + `/pipeline`

**Files:**
- Create: `frontend/app/sources/page.tsx`, `frontend/app/pipeline/page.tsx`.
- Create: `frontend/components/sources/CsvWizard.tsx`, `ColumnMatchReport.tsx`,
  `DataInfo.tsx`, `FeatureGroups.tsx`; `frontend/components/wizard/Stepper.tsx`.
- CSV path: drop → `parseCSV` → `matchColumns` report → `DataInfo` → **Run forecast**
  (`forecastCsv`; on 202 poll the job via TanStack Query `refetchInterval`) → on success
  `store.result` + route to `/dashboard`. PCAP tab = 501 explainer. Telemetry tab =
  config + agent snippet + link to `/live`.
- `/pipeline`: the 5-step stepper with live counts from `store.result?.meta` or the
  active live session; job progress bar.

- [x] Build; manual: upload `_legacy` sample or a synthetic CSV → report renders,
      missing-column case shows errors, happy path routes to dashboard (needs backend).
- [x] Commit: `feat(web): data-sources wizard + pipeline view`.

### Task 3.7 — `/dashboard`

**Files:**
- Create: `frontend/app/dashboard/page.tsx` — reads `store.result` **or**
  `?session=<id>` (subscribe via `ws.ts`, accumulate into `store.liveForecasts`).
  Alert KPI tiles, `AnchorTable` (select → `store.selectedAnchor`), `AnchorDetail`.
  Empty state when neither source present.

- [x] Build; manual with a fixture result and with `?session=` against a running
      synthetic session.
- [x] Commit: `feat(web): SOC dashboard (upload + live session)`.

### Task 3.8 — `/live` (test-bed + capture)

**Files:**
- Create: `frontend/app/live/page.tsx`.
- Create: `frontend/components/live/SourcePicker.tsx`, `ScenarioForm.tsx`
  (scenario select, rate, duration, seed; zod + RHF), `InterfacePicker.tsx`
  (radio list from `agentStatus`, IP/CIDR filter field), `LiveControls.tsx`
  (pause render toggle, stop → `deleteLiveSession`), `LiveTimeline.tsx` (wraps
  `ProbTimeline`), `IncomingCards.tsx`.
- Flow: pick source → configure → **Start** (`createLiveSession`) → subscribe
  `openLiveStream` → render timeline + cards + ribbon + ATT&CK timeline. Capture branch
  shows the agent-launch command when `agentStatus.connected === false`.
- Test: `frontend/e2e/live.spec.ts` (Playwright) — stub `/agent/status`,
  `/live/sessions*`, and the WS; starting a synthetic session shows a forecast card.

- [x] `npm run build`; `npm run test:e2e` (Playwright, `webServer` = `next start`/`serve
      out`, backend stubbed) → green.
- [x] Commit: `feat(web): live view — synthetic test-bed + capture picker`.

### Task 3.9 — Frontend Docker + cleanup

**Files:**
- Create: `frontend/Dockerfile` (node build → export → `nginx:alpine` serving `out/`),
  `frontend/nginx.conf` (SPA fallback, module MIME, gzip), update `frontend/README.md`.
- Delete: `frontend/_legacy/`.
- Modify: root `.gitignore` — `frontend/node_modules/`, `frontend/.next/`,
  `frontend/out/`.

- [x] `docker build -t sentinel-wm-frontend frontend/` succeeds; container serves `/`.
- [x] Commit: `feat(web): static Docker image; drop legacy SPA`.

---

## Phase 4 — Polish & ops

### Task 4.1 — Structured logging + request id

**Files:**
- Create: `backend/app/logging.py` — `configure_logging(json:bool)` via `dictConfig`;
  a `RequestIdMiddleware` binding `X-Request-ID` to a `ContextVar` + a logging filter
  that injects it.
- Modify: `backend/app/main.py` — call `configure_logging(settings.log_json)`, add the
  middleware.
- Modify: `backend/app/settings.py` — `log_json:bool=False` (`SENTINEL_LOG_JSON`).
- Test: `backend/tests/test_logging.py` — a request carries its `X-Request-ID` through to
  the response header; `configure_logging(True)` emits parseable JSON lines.

- [x] Tests green; `pytest backend/ -q` green.
- [x] Commit: `feat(ops): structured logging + request ids`.

### Task 4.2 — `/health` detail + optional `/metrics`

**Files:**
- Modify: `backend/app/api/routes_meta.py` — `/health` returns
  `{status, bundle:{loaded,serve_mode,L,K,n_features}, live:{sessions,max},
  agent:{connected,count}, uptime_s}`.
- Create: `backend/app/api/routes_metrics.py` — mounted only if `settings.metrics`
  (`SENTINEL_METRICS`); `prometheus_client` histogram (forecast latency, set from
  `LiveSession.feed`), gauge (active sessions), counter (flows_in).
- Modify: `backend/pyproject.toml` — `prometheus-client` in an optional `metrics` extra.
- Test: `backend/tests/test_health_detail.py`.

- [x] Tests green.
- [x] Commit: `feat(ops): health detail + optional prometheus metrics`.

### Task 4.3 — Compose, Makefile, env, docs

**Files:**
- Modify: `docker-compose.yml` — keep `backend` + `frontend`; add the new
  `SENTINEL_*` envs; comment block documenting the host capture agent.
- Modify: `Makefile` — ensure `synth-demo`, `capture-agent`, `frontend-dev`
  (`cd frontend && npm run dev`), `frontend-build`, `backend-test`,
  `agent-test` (`cd capture-agent && pytest`).
- Modify: root `README.md` — new §"Real-time: test-bed & live capture", updated
  quickstart (4 steps + agent), endpoint list.
- Modify: `docs/technical_reference.md` — live pipeline, synth scenarios table,
  flowmeter notes.
- Modify: root `.env.example` — new vars.

- [x] `docker compose config` validates; `make backend-test` + `make agent-test` green.
- [x] Commit: `docs(ops): compose, Makefile, README, technical reference`.

### Task 4.4 — Full-stack verification

- [x] `docker compose up --build` → backend :8000 healthy, frontend :8080 serves.
- [x] In the UI: start a `portscan` synthetic session on `/live` → timeline + ATT&CK
      cards populate; "Open in dashboard" shows the anchor detail.
- [x] Locally (not compose): `make capture-agent`, `/live` → Live capture → pick loopback
      → Start while pinging localhost → forecasts stream.
- [x] `pytest backend/ -q && pytest capture-agent/ -q && (cd frontend && npm test)` all
      green.
- [x] Commit: `test: full-stack live verification` (notes in the message).

---

## Self-review

**Spec coverage:** §2 arch → Tasks 1.3/1.5/2.5; §3 live spine → 1.1‑1.6; §4 synth → 1.2;
§5 capture agent → 2.1‑2.6; §6 frontend → 3.1‑3.9; §7 polish/ops → 4.1‑4.3; §8 phases →
the four phase headers; §9 risks → mitigations live in the tasks (agent optional,
flowmeter scoped to consumed columns, static export, session caps). No uncovered section.

**Placeholder scan:** no "TBD/handle edge cases/write tests for the above"; every task
names exact files, the interface signatures neighbours depend on, and concrete test
assertions.

**Type consistency:** `LiveSession`/`LiveManager`/`MANAGER`, `SyntheticSource.run`/
`request_stop`, `AgentSource.on_flows`/`start`/`stop`, `AgentConnection.send_cmd`/`bind`,
`AgentRegistry.first`/`REGISTRY`, `FlowMeter.add_packet`/`harvest`/`pending`,
`Interface` fields, `matchColumns`, `openLiveStream` — each defined once and referenced
consistently. Flow-row column names are pinned to `app.sentinel_infer.forecast` in every
producer (synth + flowmeter).
