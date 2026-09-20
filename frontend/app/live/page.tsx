"use client";
import * as React from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Pause, Play, Square, Activity, ExternalLink } from "lucide-react";
import { Panel } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { KpiTile } from "@/components/panels/KpiTile";
import { ProbTimeline } from "@/components/charts/ProbTimeline";
import { ProgressionRibbon } from "@/components/panels/ProgressionRibbon";
import { ScenarioForm } from "@/components/live/ScenarioForm";
import { InterfacePicker } from "@/components/live/InterfacePicker";
import { CyberSpinner, RadarScanner, LivePulseBadge } from "@/components/ui/loading";
import { api } from "@/lib/api";
import { useStore, type LiveSourceMode } from "@/lib/store";
import { pct } from "@/lib/format";
import { cn } from "@/lib/utils";

export default function LivePage() {
  const mode = useStore((s) => s.liveMode);
  const setMode = useStore((s) => s.setLiveMode);
  const form = useStore((s) => s.scenarioForm);
  const setForm = useStore((s) => s.setScenarioForm);
  const iface = useStore((s) => s.captureIface);
  const bpf = useStore((s) => s.captureBpf);
  const setCaptureParams = useStore((s) => s.setCaptureParams);

  const sessionId = useStore((s) => s.liveSessionId);
  const status = useStore((s) => s.liveStatus);
  const connected = useStore((s) => s.liveConnected);
  const paused = useStore((s) => s.livePaused);
  const forecasts = useStore((s) => s.liveForecasts);

  const startLiveSession = useStore((s) => s.startLiveSession);
  const stopLiveSession = useStore((s) => s.stopLiveSession);
  const toggleLivePause = useStore((s) => s.toggleLivePause);

  const [starting, setStarting] = React.useState(false);

  const start = async () => {
    setStarting(true);
    try {
      const payload =
        mode === "synthetic"
          ? { source: "synthetic", ...form }
          : { source: "capture", iface, bpf: bpf || null };
      if (mode === "capture" && !iface) {
        toast.error("Pick an interface first");
        setStarting(false);
        return;
      }
      const s = await api.createLiveSession(payload);
      startLiveSession(s);
      toast.success(`Live session ${s.id.slice(0, 8)} active`, {
        description: `Streaming ${mode === "synthetic" ? `scenario: ${form.scenario}` : `interface: ${iface}`}`,
      });
    } catch (e: any) {
      toast.error(
        e?.status === 503
          ? "No capture agent connected"
          : "Could not start: " + (e?.message || e)
      );
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    await stopLiveSession();
    toast.info("Live telemetry session terminated");
  };

  const running = !!sessionId;
  const latest = forecasts[forecasts.length - 1];
  const points = forecasts.map((f, i) => ({
    i,
    p: f.max_detection_prob ?? f.max_attack_prob ?? f.horizon?.[0]?.attack_prob ?? 0,
  }));
  const nAlerts = forecasts.filter((f) => f.alert).length;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink flex items-center gap-2.5">
            Live Telemetry
            {running && <LivePulseBadge connected={connected} label={connected ? "STREAMING" : "CONNECTING"} />}
          </h1>
          <p className="mt-1 text-xs text-muted">
            Continuous 10-second state window aggregation and rolling K=6 predictive horizon simulation.
          </p>
        </div>

        {running && (
          <div className="flex items-center gap-2">
            <Link
              href={`/dashboard?session=${sessionId}`}
              className="inline-flex items-center gap-1.5 rounded-md border border-brand/40 bg-brand/10 px-3 py-1.5 text-xs font-medium text-brand hover:bg-brand/20 transition-colors"
            >
              Open in SOC Dashboard <ExternalLink size={12} />
            </Link>
          </div>
        )}
      </div>

      {!running && (
        <Panel title="Source Configuration">
          <div className="flex gap-1 border-b border-line">
            {(["synthetic", "capture"] as LiveSourceMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors",
                  mode === m
                    ? "border-brand text-brand"
                    : "border-transparent text-muted hover:text-ink"
                )}
              >
                {m === "synthetic" ? "Test-bed (Synthetic Scenario)" : "Live Capture (Local NIC)"}
              </button>
            ))}
          </div>
          <div className="pt-4">
            {mode === "synthetic" ? (
              <>
                <ScenarioForm value={form} onChange={setForm} />
                <p className="mt-3 text-[11px] text-warn">
                  Synthetic traffic is out-of-distribution for the production model —
                  absolute P(attack) is not calibrated. Watch the relative ramp, the
                  progression sequence, and the ATT&amp;CK mapping.
                </p>
              </>
            ) : (
              <InterfacePicker
                iface={iface}
                bpf={bpf}
                onIface={(val) => setCaptureParams(val, bpf)}
                onBpf={(val) => setCaptureParams(iface, val)}
              />
            )}
          </div>
          <div className="mt-4 flex items-center gap-3">
            <Button className="w-max" onClick={start} disabled={starting}>
              {starting ? (
                <>
                  <CyberSpinner size={14} /> Initializing stream…
                </>
              ) : (
                <>
                  <Play size={14} /> Start {mode === "synthetic" ? "Scenario Test-Bed" : "Live Capture"}
                </>
              )}
            </Button>
          </div>
        </Panel>
      )}

      {running && (
        <>
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-surface px-4 py-3 shadow-sm">
            <span className="flex items-center gap-2 text-sm">
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  connected ? "bg-ok" : "bg-warn animate-pulse"
                )}
              />
              <span className="text-muted">
                {connected ? "streaming" : "connecting"} · {status?.source_kind || "live"} ·{" "}
                {status?.state || "active"}
              </span>
            </span>
            <span className="mono text-xs text-muted">
              flows: <strong className="text-ink">{status?.stats.flows_in ?? 0}</strong> · windows:{" "}
              <strong className="text-ink">{status?.stats.windows ?? 0}</strong> · forecasts:{" "}
              <strong className="text-brand">{forecasts.length}</strong>
            </span>
            <span className="flex-1" />
            <Button size="sm" variant="ghost" onClick={toggleLivePause}>
              {paused ? <Play size={13} /> : <Pause size={13} />}
              {paused ? "resume stream" : "pause stream"}
            </Button>
            <Button size="sm" variant="danger" onClick={stop}>
              <Square size={13} /> stop session
            </Button>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <KpiTile
              label="Latest P(attack)"
              value={latest ? pct(latest.max_detection_prob ?? latest.max_attack_prob) : "—"}
              tone={latest?.alert ? "danger" : "ok"}
            />
            <KpiTile label="Alert windows" value={`${nAlerts} / ${forecasts.length}`} />
            <KpiTile
              label="Top tactic"
              value={latest?.horizon?.[0]?.attck?.mitre_tactic || "—"}
            />
          </div>

          <Panel title="Rolling P(attack) Timeline">
            {points.length > 0 ? (
              <ProbTimeline
                points={points}
                threshold={
                  status?.params && (status.params as any).alert_threshold
                    ? Number((status.params as any).alert_threshold)
                    : 0.5
                }
                height={240}
              />
            ) : (
              <div className="py-2">
                <RadarScanner
                  flowsReceived={status?.stats.flows_in ?? 0}
                  windowsCount={status?.stats.windows ?? 0}
                  targetWindows={12}
                  statusText={connected ? "Streaming telemetry frames" : "Connecting to telemetry feed"}
                />
              </div>
            )}
          </Panel>

          {latest?.horizon && (
            <Panel title="Latest kill-chain progression">
              <ProgressionRibbon horizon={latest.horizon} />
            </Panel>
          )}

          <Panel title="Incoming forecasts">
            <div className="flex gap-3 overflow-x-auto pb-2">
              {forecasts
                .slice(-8)
                .reverse()
                .map((f, i) => (
                  <div
                    key={i}
                    className="flex min-w-[220px] flex-col gap-2 rounded-lg border border-line bg-surface p-3 transition-transform hover:-translate-y-0.5 shadow-sm"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="mono text-muted">
                        win {String((f.meta?.stream_window as number) ?? "?")}
                      </span>
                      <span
                        className={cn(
                          "font-semibold",
                          f.alert ? "text-danger font-mono" : "text-muted"
                        )}
                      >
                        {f.alert ? `ALERT +${f.first_alert_k}` : "clear"}
                      </span>
                    </div>
                    <div className="kpi text-xl font-bold">
                      {pct(f.max_detection_prob ?? f.max_attack_prob)}
                    </div>
                    <div className="text-xs text-muted flex items-center justify-between">
                      <span>{f.horizon?.[0]?.attck?.mitre_tactic || "—"}</span>
                      <span className="mono text-[10px] text-muted/80">
                        {f.horizon?.[0]?.progression_state || ""}
                      </span>
                    </div>
                  </div>
                ))}
              {!forecasts.length && (
                <div className="w-full py-4 text-center">
                  <span className="mono text-xs text-muted flex items-center justify-center gap-2">
                    <CyberSpinner size={14} /> Telemetry frames streaming. Waiting for 12th window to emit anchor...
                  </span>
                </div>
              )}
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
