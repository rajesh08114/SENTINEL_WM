import { Panel } from "@/components/ui/card";

const ENDPOINTS: [string, string, string][] = [
  ["GET", "/health", "liveness + bundle / live / agent detail"],
  ["GET", "/meta", "L / K / feature_dim, window seconds, progression states, serve mode, alert threshold, bundle manifest"],
  ["GET", "/models", "registry of models in the served bundle"],
  ["POST", "/forecast/csv", "multipart flow CSV → ForecastResponse (sync ≤ max_sync_flows, else 202 {job_id})"],
  ["POST", "/forecast/pcap", "501 — deferred to Phase 2"],
  ["GET", "/jobs · /jobs/{id} · /jobs/{id}/result", "async forecast jobs"],
  ["POST", "/live/sessions", "start a synthetic test-bed or live-capture session"],
  ["GET/DELETE", "/live/sessions[/{id}]", "list / inspect / stop a live session"],
  ["WS", "/live/sessions/{id}/stream", "subscribe: status + ring replay, then live forecasts"],
  ["WS", "/agent", "a capture agent connects, advertises NICs, streams flows"],
  ["GET", "/agent/status", "connected agents + their interface lists"],
  ["WS", "/stream", "raw telemetry (NDJSON flow records)"],
];

export default function ArchitecturePage() {
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-xl font-semibold">Interaction architecture</h1>
      <p className="max-w-3xl text-sm text-muted">
        Four decoupled parts. <b className="text-ink">research/</b> trains and benchmarks,
        emitting a portable model bundle. <b className="text-ink">backend/</b> serves that
        bundle through a stateless FastAPI service (inference code vendored in{" "}
        <code>app/sentinel_infer/</code> — no research import). <b className="text-ink">frontend/</b>{" "}
        is this static Next.js console. <b className="text-ink">capture-agent/</b> is a host
        process that feeds live packets in.
      </p>

      <Panel title="Repository topology">
        <pre className="mono overflow-x-auto text-xs leading-relaxed text-muted sm:text-[13px]">
{`SIH/
├── research/            ML package (sentinel_wm) — CLI + JSON artifacts
├── backend/             FastAPI inference service
│   ├── app/sentinel_infer/   vendored, self-contained inference lib
│   ├── app/live/             LiveSession + LiveManager + sources
│   ├── app/synth/            synthetic scenario generator
│   ├── app/agent/            capture-agent registry
│   └── app/api/              forecast · jobs · meta · live · agent · ws_stream
├── capture-agent/       host process: sniff NIC → flows → WS /agent
├── frontend/            this console (Next.js static export → nginx)
└── runs/                research benchmark output`}
        </pre>
      </Panel>

      <Panel title="Request lifecycle — CSV forecast">
        <ol className="flex list-decimal flex-col gap-2 pl-5 text-sm text-muted">
          <li>
            <b className="text-ink">Client match.</b> The browser parses the CSV header and
            reports required / aliased / Tier-2 / flag columns before upload.
          </li>
          <li>
            <b className="text-ink">Upload.</b> <code>POST /forecast/csv</code>.{" "}
            <code>normalise_upload</code> canonicalizes aliases, derives{" "}
            <code>flow_start_epoch</code>; a 422 lists any missing required column.
          </li>
          <li>
            <b className="text-ink">Windowing.</b> <code>clean_flow_frame</code> →{" "}
            <code>build_state_windows</code> (10 s aggregation, 53 features).
          </li>
          <li>
            <b className="text-ink">Tensors.</b> Slide L = 12 with a contiguity guard, scale
            with the bundle&apos;s <code>state_scaler.pkl</code>.
          </li>
          <li>
            <b className="text-ink">Rollout.</b> Per anchor: world-model K-step MC rollout,
            optional system blend, <code>assess_forecast</code> for ATT&amp;CK, gradient /
            attention explainer.
          </li>
          <li>
            <b className="text-ink">Response.</b> Sync JSON under the sync ceiling; otherwise
            a job id to poll.
          </li>
        </ol>
      </Panel>

      <Panel title="Real-time lifecycle — synthetic & capture">
        <ol className="flex list-decimal flex-col gap-2 pl-5 text-sm text-muted">
          <li>
            <code>POST /live/sessions</code> creates a <code>LiveSession</code> (own{" "}
            <code>StreamingWindower</code> + a 200-entry forecast ring).
          </li>
          <li>
            <b className="text-ink">Synthetic:</b> an asyncio task paces{" "}
            <code>scenarios.emit()</code> against the wall clock (× a speed multiplier) and
            feeds rows in.
          </li>
          <li>
            <b className="text-ink">Capture:</b> the bound agent gets <code>{"{cmd:start}"}</code>,
            sniffs the chosen NIC, and streams assembled flow rows over <code>WS /agent</code>.
          </li>
          <li>
            Each closed window → <code>simulate_anchor</code> → fan-out to every WS
            subscriber of <code>/live/sessions/{"{id}"}/stream</code>.
          </li>
        </ol>
      </Panel>

      <Panel title="HTTP + WebSocket surface">
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Method</th>
                <th>Path</th>
                <th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              {ENDPOINTS.map(([m, p, d]) => (
                <tr key={p}>
                  <td className="mono text-brand">{m}</td>
                  <td className="mono">{p}</td>
                  <td className="text-muted">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Decoupling contract">
          <ul className="flex list-disc flex-col gap-2 pl-4 text-sm text-muted">
            <li>
              <code>backend/app/sentinel_infer/</code> is a vendored copy — no{" "}
              <code>sentinel_wm</code> import anywhere in the backend.
            </li>
            <li>
              <code>load_bundle</code> asserts{" "}
              <code>bundle.json[&quot;feature_names&quot;] == STATE_FEATURE_COLS</code> and
              raises <code>BundleContractError</code> on drift.
            </li>
            <li>
              <code>tests/test_vendor_sync.py</code> diffs the vendored schema / features /
              ATT&amp;CK table against <code>research/</code> when present.
            </li>
          </ul>
        </Panel>
        <Panel title="Deployment">
          <ul className="flex list-disc flex-col gap-2 pl-4 text-sm text-muted">
            <li>Backend image bakes <code>backend/models/</code>; a read-only volume overrides it for retrains.</li>
            <li>Stateless service — the only durable artifact is the SQLite job store.</li>
            <li>Frontend ships as static files behind nginx; the topbar field points it at any backend origin.</li>
            <li>The capture agent runs on the host (Npcap + Administrator), never a container.</li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}
