# frontend/ — SENTINEL-WM UI  (Phase 4, not built yet)

A Next.js (app router) + Tailwind UI over the [`../backend`](../backend) API.

## Planned screens

| screen | what |
|---|---|
| **Upload** | CSV / PCAP dropzone → `POST /forecast/csv` (or a job) → progress → results table |
| **Live** | telemetry monitor over `WS /stream`; a rolling P(attack) timeline, alerts as windows close |
| **Anchor detail** | per-horizon P(attack) + 95% CI line chart, progression-state ribbon (NORMAL→…→CONTINUATION), ATT&CK panel (tactic / technique IDs / kill-chain phase / confidence / rationale), driving-features bar + temporal-saliency sparkline |

## Planned layout

```
app/            routes (upload, live, anchor/[id])
components/      charts, ATT&CK panel, timeline, dropzone
lib/api.ts      typed client for /meta /forecast/csv /jobs /stream
lib/useForecastStream.ts   WebSocket hook
Dockerfile      multi-stage next build
```

Config: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). Add the
`frontend` service to `../docker-compose.yml` when ready.
