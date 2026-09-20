"use client";
import * as React from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Radio, FileText, UploadCloud, Play, CheckCircle2 } from "lucide-react";
import { Panel } from "@/components/ui/card";
import { KpiTile } from "@/components/panels/KpiTile";
import { AnchorTable } from "@/components/panels/AnchorTable";
import { AnchorDetail } from "@/components/panels/AnchorDetail";
import { EmptyState } from "@/components/common/EmptyState";
import { RadarScanner, LivePulseBadge } from "@/components/ui/loading";
import { api } from "@/lib/api";
import { useStore, type ActiveSource } from "@/lib/store";
import { num, pct, round } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AnchorForecast } from "@/lib/types";

export function DashboardView() {
  const params = useSearchParams();
  const sessionParam = params.get("session");

  const result = useStore((s) => s.result);
  const selected = useStore((s) => s.selectedAnchor);
  const selectAnchor = useStore((s) => s.selectAnchor);

  const activeSource = useStore((s) => s.activeSource);
  const setActiveSource = useStore((s) => s.setActiveSource);

  const liveSessionId = useStore((s) => s.liveSessionId);
  const liveStatus = useStore((s) => s.liveStatus);
  const liveForecasts = useStore((s) => s.liveForecasts);
  const liveConnected = useStore((s) => s.liveConnected);
  const attachLiveStream = useStore((s) => s.attachLiveStream);

  const { data: meta } = useQuery({ queryKey: ["meta"], queryFn: api.meta });

  // When arriving with ?session=, sync and attach to session
  React.useEffect(() => {
    if (sessionParam) {
      setActiveSource("live");
      if (liveSessionId !== sessionParam) {
        attachLiveStream(sessionParam);
      }
    }
  }, [sessionParam, liveSessionId, attachLiveStream, setActiveSource]);

  const hasLive = !!liveSessionId || liveForecasts.length > 0;
  const hasUpload = !!result && result.anchors.length > 0;

  // Determine current display mode
  let currentMode: ActiveSource = activeSource;
  if (!hasLive && hasUpload) {
    currentMode = "upload";
  } else if (hasLive && !hasUpload) {
    currentMode = "live";
  }

  const liveMode = currentMode === "live" && hasLive;
  const anchors: AnchorForecast[] = liveMode ? liveForecasts : result?.anchors ?? [];

  const threshold =
    (liveMode ? undefined : result?.summary.alert_threshold) ??
    meta?.alert_threshold ??
    0.5;

  // If nothing is loaded anywhere
  if (!hasLive && !hasUpload) {
    return (
      <div className="flex flex-col gap-5">
        <h1 className="text-xl font-semibold text-ink">SOC Dashboard</h1>
        <EmptyState
          icon="▣"
          title="No forecast data active."
          body="Start a live telemetry test-bed session or upload a network flow CSV / PCAP to generate predictive kill-chain forecasts."
        />
        <div className="mx-auto flex flex-wrap items-center gap-4 pt-2">
          <Link
            href="/live"
            className="inline-flex items-center gap-2 rounded-lg border border-brand/40 bg-brand/10 px-4 py-2 text-sm font-medium text-brand hover:bg-brand/20 transition-colors"
          >
            <Play size={15} /> Start Live Test-Bed
          </Link>
          <Link
            href="/sources"
            className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-4 py-2 text-sm font-medium text-ink hover:bg-elevated transition-colors"
          >
            <UploadCloud size={15} /> Upload Flow CSV / PCAP
          </Link>
        </div>
      </div>
    );
  }

  // If live mode is active but still buffering first 12 windows
  if (liveMode && anchors.length === 0) {
    return (
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-semibold text-ink flex items-center gap-2.5">
            SOC Dashboard
            <LivePulseBadge connected={liveConnected} label={liveConnected ? "STREAMING" : "CONNECTING"} />
          </h1>
          {hasUpload && (
            <button
              onClick={() => setActiveSource("upload")}
              className="text-xs text-brand underline decoration-dotted hover:text-brand/80"
            >
              Switch to uploaded CSV forecast →
            </button>
          )}
        </div>

        <RadarScanner
          title="Telemetry buffer warming up..."
          subtitle="The model aggregates 10-second flow state windows. Once 12 consecutive history windows accumulate (120 s), the first anchor forecast will emit."
          flowsReceived={liveStatus?.stats.flows_in ?? 0}
          windowsCount={liveStatus?.stats.windows ?? 0}
          targetWindows={12}
          statusText={liveConnected ? "Streaming live network telemetry" : "Connecting to session..."}
        />
      </div>
    );
  }

  const sel = Math.min(Math.max(0, selected), Math.max(0, anchors.length - 1));
  const a = anchors[sel];
  const nAlerts = anchors.filter((x) => x.alert).length;
  const peak = anchors.length
    ? Math.max(...anchors.map((x) => x.max_detection_prob ?? x.max_attack_prob ?? 0))
    : 0;
  const leads = anchors
    .map((x) => x.lead_time_seconds)
    .filter((v): v is number => v != null && v >= 0);
  const phases = liveMode ? [] : result?.summary.phases ?? [];

  return (
    <div className="flex flex-col gap-6">
      {/* Source Switcher Header if multiple sources are available */}
      {hasLive && hasUpload && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-line bg-surface p-2.5 shadow-sm">
          <div className="flex items-center gap-2 text-xs font-medium text-muted">
            <span>Forecast Source:</span>
            <div className="flex rounded-md border border-line bg-bg p-0.5">
              <button
                onClick={() => setActiveSource("live")}
                className={cn(
                  "flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors",
                  liveMode
                    ? "bg-brand/15 text-brand shadow-sm"
                    : "text-muted hover:text-ink"
                )}
              >
                <Radio size={12} />
                Live Telemetry ({liveForecasts.length} frames)
                {liveSessionId && <span className="h-1.5 w-1.5 rounded-full bg-ok animate-pulse" />}
              </button>
              <button
                onClick={() => setActiveSource("upload")}
                className={cn(
                  "flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors",
                  !liveMode
                    ? "bg-brand/15 text-brand shadow-sm"
                    : "text-muted hover:text-ink"
                )}
              >
                <FileText size={12} />
                Uploaded Dataset ({result?.anchors.length} anchors)
              </button>
            </div>
          </div>
          <div className="text-[11px] font-mono text-muted">
            Viewing: <strong className="text-ink">{liveMode ? `Session ${liveSessionId?.slice(0, 8)}` : result?.meta.family_hint || result?.meta.model || "Uploaded CSV"}</strong>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold text-ink flex items-center gap-2.5">
            SOC Dashboard
            {liveMode ? (
              <LivePulseBadge connected={liveConnected} label="LIVE STREAM" />
            ) : (
              <span className="inline-flex items-center gap-1 rounded-full border border-line bg-elevated px-2 py-0.5 font-mono text-[11px] text-muted">
                <CheckCircle2 size={11} className="text-ok" /> DATASET
              </span>
            )}
          </h1>
          <p className="text-xs text-muted mt-1">
            {liveMode
              ? `Real-time forecast horizon for active session ${liveSessionId?.slice(0, 8) || "live"}`
              : `Predictive attack-progression evaluation for ${result?.meta.family_hint || result?.meta.model || "uploaded traffic"}`}
          </p>
        </div>

        <div className="mono text-xs text-muted">
          {liveMode
            ? `${anchors.length} streaming anchors · threshold ${round(threshold, 3)}`
            : `${result?.meta.model || "SENTINEL-WM"} · ${result?.meta.n_anchors} anchors · threshold ${round(threshold, 3)}`}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          label="Alert windows"
          value={`${nAlerts} / ${anchors.length}`}
          sub={nAlerts ? "attacker progression forecast" : "all clear"}
          tone={nAlerts ? "danger" : "ok"}
        />
        <KpiTile label="Peak P(attack)" value={pct(peak)} sub="across all horizons" />
        <KpiTile
          label="Earliest lead time"
          value={leads.length ? `${num(Math.min(...leads))} s` : "—"}
          sub="to forecast onset"
        />
        <KpiTile
          label="ATT&CK phases"
          value={String(
            phases.length ||
              new Set(anchors.map((x) => x.horizon[0]?.attck?.mitre_tactic)).size
          )}
          sub={
            (phases.slice(0, 3) as any[])
              .map((p) => (typeof p === "string" ? p : p.mitre_tactic))
              .join(", ") ||
            anchors[0]?.horizon[0]?.attck?.mitre_tactic ||
            "none"
          }
        />
      </div>

      <Panel title={liveMode ? "Streaming Anchors (10s State Windows)" : "Dataset Anchors"}>
        <AnchorTable
          anchors={anchors}
          threshold={threshold}
          selected={sel}
          onSelect={selectAnchor}
          fallbackModel={result?.meta.model}
        />
      </Panel>

      {a && <AnchorDetail anchor={a} threshold={threshold} />}
    </div>
  );
}
