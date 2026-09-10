# SENTINEL-WM Console (frontend)

A **Next.js 14** (App Router, TypeScript, Tailwind, shadcn-style primitives)
SOC-analyst console for the SENTINEL-WM inference backend. Builds to a **static
export** (`output: "export"`) served by nginx — no Node server in production.

## Run it

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000  (set the API base in the topbar)
```

Static build + preview:

```bash
npm run build          # -> out/
npx serve out -l 3100
```

Docker (nginx on :8080):

```bash
docker build -t sentinel-wm-frontend frontend/
docker run -p 8080:8080 sentinel-wm-frontend
```

Whole stack: `docker compose up --build` (console on `http://localhost:8080`).

## Pointing at a backend

Resolution order for the API base URL:

1. `?api=https://host:port` query param
2. `localStorage["sentinel.apiBase"]` (topbar field)
3. `NEXT_PUBLIC_API_BASE` (build-time)
4. `${location.protocol}//${location.hostname}:8000`

The backend must allow the console origin in `SENTINEL_CORS_ORIGINS`.

## Layout

```
app/
  layout.tsx              shell: topbar (API base, health, theme), sidenav, providers
  page.tsx                / overview — problem, solution, live system status
  architecture/page.tsx   repo topology, request lifecycles, endpoint table
  model/page.tsx          SENTINEL-WM (system) card — layers, params, forward/backward
  sources/page.tsx        CSV column-match wizard · PCAP (501) · telemetry snippet
  pipeline/page.tsx       ingest→clean→windows→sequences→forecast, live counts
  dashboard/             SOC dashboard — from an upload OR ?session=<id> (live)
  live/page.tsx           synthetic test-bed + live-capture interface picker
components/
  ui/                     button, card, tabs, inputs, badges (shadcn-style)
  charts/                 ForecastChart (CI band), ProbTimeline, FeatureBars, Saliency
  panels/                 KpiTile, ProgressionRibbon, AttckPhaseTimeline, HorizonTable,
                          AnchorTable, AnchorDetail
  live/                   ScenarioForm, InterfacePicker
  common/                 ApiBaseField, HealthDot, ThemeToggle, ErrorBoundary, EmptyState
lib/
  api.ts   zod-parsed typed client + pollJob
  ws.ts    reconnecting live-stream subscription
  schema.ts / csv.ts   column contract mirror + parser
  store.ts (zustand)   types.ts   format.ts
```

## Tests

```bash
npm test               # Vitest — schema/csv, api zod parsing, ws reconnect, panels
npm run build && npx playwright test    # routes render; a synthetic session streams
```

The Playwright synthetic-session test needs a backend on `:8000` (override with
`E2E_API`); it self-skips when none is reachable.

## Design

Dark-committed (SOC context); a light token set drives the topbar toggle.
Matrix-green brand `#2ee66b`, alert red `#ff4d4d`, blue `#5aa9ff` for the
world-model rollout series. Fira Code / Fira Sans. Every chart is paired with a
data table; motion respects `prefers-reduced-motion`.
