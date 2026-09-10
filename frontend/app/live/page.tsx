"use client";
import * as React from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Pause, Play, Square } from "lucide-react";
import { Panel } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { KpiTile } from "@/components/panels/KpiTile";
import { ProbTimeline } from "@/components/charts/ProbTimeline";
import { ProgressionRibbon } from "@/components/panels/ProgressionRibbon";
import { ScenarioForm, type ScenarioValues } from "@/components/live/ScenarioForm";
import { InterfacePicker } from "@/components/live/InterfacePicker";
import { api } from "@/lib/api";
import { openLiveStream, type LiveStream } from "@/lib/ws";
import { pct } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AnchorForecast, LiveSessionInfo } from "@/lib/types";

type Mode = "synthetic" | "capture";

export default function LivePage() {
  const [mode, setMode] = React.useState<Mode>("synthetic");
  const [form, setForm] = React.useState<ScenarioValues>({
    scenario: "portscan",
    rate: 25,
    duration_s: 600,
    speed: 30,
    seed: 1,
  });
  const [iface, setIface] = React.useState("");
  const [bpf, setBpf] = React.useState("");

  const [sessionId, setSessionId] = React.useState<string | null>(null);
  const [status, setStatus] = React.useState<LiveSessionInfo | null>(null);
  const [connected, setConnected] = React.useState(false);
  const [paused, setPaused] = React.useState(false);
  const [forecasts, setForecasts] = React.useState<AnchorForecast[]>([]);
  const [starting, setStarting] = React.useState(false);
  const streamRef = React.useRef<LiveStream | null>(null);
  const pausedRef = React.useRef(false);
  pausedRef.current = paused;

  React.useEffect(() => () => streamRef.current?.close(), []);

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
      setSessionId(s.id);
      setStatus(s);
      setForecasts([]);
      streamRef.current = openLiveStream(s.id, {
        onOpen: () => setConnected(true),
        onClose: () => setConnected(false),
        onStatus: (st) => setStatus(st),
        onError: (d) => toast.error("stream: " + d),
        onForecast: (f) => {
          if (pausedRef.current) return;
          setForecasts((prev) => [...prev.slice(-199), f]);
        },
      });
      toast.success(`Live session ${s.id.slice(0, 8)} started`);
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
    streamRef.current?.close();
    streamRef.current = null;
    if (sessionId) await api.deleteLiveSession(sessionId).catch(() => {});
    setSessionId(null);
    setStatus(null);
    setConnected(false);
    toast.info("Session stopped");
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
      <h1 className="text-xl font-semibold">Live telemetry</h1>

      {!running && (
        <Panel title="Source">
          <div className="flex gap-1 border-b border-line">
            {(["synthetic", "capture"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "-mb-px border-b-2 px-3 py-2 text-sm",
                  mode === m
                    ? "border-brand text-ink"
                    : "border-transparent text-muted hover:text-ink"
                )}
              >
                {m === "synthetic" ? "Test-bed (synthetic)" : "Live capture"}
              </button>
            ))}
          </div>
          <div className="pt-3">
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
                onIface={setIface}
                onBpf={setBpf}
              />
            )}
          </div>
          <Button className="w-max" onClick={start} disabled={starting}>
            <Play size={14} /> {starting ? "starting…" : "Start"}
          </Button>
        </Panel>
      )}

      {running && (
        <>
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-surface px-4 py-3">
            <span className="flex items-center gap-2 text-sm">
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  connected ? "bg-ok" : "bg-warn animate-pulse"
                )}
              />
              <span className="text-muted">
                {connected ? "streaming" : "connecting"} · {status?.source_kind} ·{" "}
                {status?.state}
              </span>
            </span>
            <span className="mono text-xs text-muted">
              flows {status?.stats.flows_in ?? 0} · forecasts {forecasts.length}
            </span>
            <span className="flex-1" />
            <Button size="sm" variant="ghost" onClick={() => setPaused((p) => !p)}>
              {paused ? <Play size={13} /> : <Pause size={13} />}
              {paused ? "resume" : "pause"}
            </Button>
            <Button size="sm" variant="danger" onClick={stop}>
              <Square size={13} /> stop
            </Button>
            <Link
              href={`/dashboard?session=${sessionId}`}
              className="text-xs text-brand underline decoration-dotted"
            >
              open in dashboard →
            </Link>
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

          <Panel title="Rolling P(attack)">
            {points.length ? (
              <ProbTimeline
                points={points}
                threshold={status?.params && (status.params as any).alert_threshold ? Number((status.params as any).alert_threshold) : 0.5}
                height={240}
              />
            ) : (
              <p className="text-sm text-muted">
                Waiting for the first forecast (~12 windows of history)…
              </p>
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
                    className="flex min-w-[220px] flex-col gap-2 rounded-lg border border-line bg-surface p-3"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="mono text-muted">
                        win {String((f.meta?.stream_window as number) ?? "?")}
                      </span>
                      <span
                        className={cn(
                          "font-semibold",
                          f.alert ? "text-danger" : "text-muted"
                        )}
                      >
                        {f.alert ? `ALERT +${f.first_alert_k}` : "clear"}
                      </span>
                    </div>
                    <div className="kpi text-xl">
                      {pct(f.max_detection_prob ?? f.max_attack_prob)}
                    </div>
                    <div className="text-xs text-muted">
                      {f.horizon?.[0]?.attck?.mitre_tactic || "—"}
                    </div>
                  </div>
                ))}
              {!forecasts.length && (
                <p className="text-xs text-muted">Nothing yet.</p>
              )}
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
