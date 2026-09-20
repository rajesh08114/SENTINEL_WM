"use client";
import * as React from "react";
import { Activity, ShieldAlert, Cpu, Network, Radio } from "lucide-react";
import { cn } from "@/lib/utils";

/** Glowing cyber spinner with outer ring, rotating accent, and central pulse */
export function CyberSpinner({
  size = 24,
  className,
  label,
}: {
  size?: number;
  className?: string;
  label?: string;
}) {
  return (
    <div className={cn("inline-flex items-center gap-2.5", className)}>
      <div
        className="relative flex items-center justify-center shrink-0"
        style={{ width: size, height: size }}
      >
        <div className="absolute inset-0 rounded-full border-2 border-brand/20" />
        <div
          className="absolute inset-0 rounded-full border-2 border-transparent border-t-brand animate-spin"
          style={{ animationDuration: "0.85s" }}
        />
        <div className="h-1.5 w-1.5 rounded-full bg-brand animate-ping" />
      </div>
      {label && <span className="text-xs text-muted font-mono">{label}</span>}
    </div>
  );
}

/** High-tech radar scanner for live telemetry warming and buffer accumulation */
export function RadarScanner({
  title = "Awaiting initial telemetry history...",
  subtitle = "The World Model requires ~12 consecutive 10-second windows (120 s history) to anchor state transitions.",
  flowsReceived = 0,
  windowsCount = 0,
  targetWindows = 12,
  statusText = "Ingesting telemetry frames",
}: {
  title?: string;
  subtitle?: string;
  flowsReceived?: number;
  windowsCount?: number;
  targetWindows?: number;
  statusText?: string;
}) {
  const progressPct = Math.min(100, Math.round((windowsCount / targetWindows) * 100));

  return (
    <div className="relative overflow-hidden rounded-xl border border-brand/30 bg-gradient-to-b from-surface to-elevated p-8 text-center shadow-lg">
      {/* Background cyber grid effect */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#00ffcc08_1px,transparent_1px),linear-gradient(to_bottom,#00ffcc08_1px,transparent_1px)] bg-[size:24px_24px]" />

      <div className="relative z-10 flex flex-col items-center justify-center">
        {/* Animated Radar Circle */}
        <div className="relative mb-5 flex h-36 w-36 items-center justify-center rounded-full border border-brand/40 bg-brand/5 shadow-[0_0_30px_rgba(0,255,204,0.15)]">
          {/* Concentric rings */}
          <div className="absolute h-24 w-24 rounded-full border border-brand/30" />
          <div className="absolute h-12 w-12 rounded-full border border-brand/25" />
          
          {/* Crosshairs */}
          <div className="absolute h-full w-[1px] bg-brand/20" />
          <div className="absolute h-[1px] w-full bg-brand/20" />

          {/* Sweeping Radar Beam */}
          <div
            className="absolute inset-0 origin-center rounded-full bg-[conic-gradient(from_0deg,transparent_0_300deg,rgba(0,255,204,0.35)_360deg)] animate-spin"
            style={{ animationDuration: "2.5s" }}
          />

          {/* Center core pulse */}
          <div className="relative z-10 flex h-7 w-7 items-center justify-center rounded-full bg-brand/20 border border-brand text-brand shadow-[0_0_12px_rgba(0,255,204,0.6)]">
            <Radio size={14} className="animate-pulse" />
          </div>

          {/* Simulated scanning blips */}
          <div className="absolute top-7 right-8 h-1.5 w-1.5 rounded-full bg-brand animate-ping" />
          <div
            className="absolute bottom-9 left-7 h-2 w-2 rounded-full bg-warn/80 animate-ping"
            style={{ animationDelay: "1.2s", animationDuration: "2s" }}
          />
        </div>

        <h3 className="text-base font-semibold text-ink flex items-center gap-2">
          <Activity size={16} className="text-brand animate-pulse" />
          {title}
        </h3>
        <p className="mt-1.5 max-w-md text-xs text-muted leading-relaxed">
          {subtitle}
        </p>

        {/* Dynamic telemetry buffer indicator */}
        <div className="mt-6 w-full max-w-sm rounded-lg border border-line bg-surface/80 p-3.5 backdrop-blur">
          <div className="flex items-center justify-between text-xs mb-2">
            <span className="flex items-center gap-1.5 font-mono text-muted">
              <Network size={12} className="text-brand" />
              {statusText}
            </span>
            <span className="font-mono font-semibold text-brand">
              {windowsCount} / {targetWindows} windows ({progressPct}%)
            </span>
          </div>

          {/* Progress bar */}
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-line">
            <div
              className="h-full bg-gradient-to-r from-brand/60 to-brand transition-all duration-500"
              style={{ width: `${Math.max(5, progressPct)}%` }}
            />
          </div>

          <div className="mt-2.5 flex items-center justify-between font-mono text-[11px] text-muted">
            <span>Flows ingested: <strong className="text-ink">{flowsReceived.toLocaleString()}</strong></span>
            <span>Window span: 10s · History: L=12</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Multi-stage cyber progress indicator for forecast generation */
export function ForecastProgress({
  stage,
  isJob = false,
}: {
  stage?: string;
  isJob?: boolean;
}) {
  const steps = [
    { label: "Validating & Ingesting Flows", desc: "Sanitizing 53 network state dimensions" },
    { label: "Temporal Window Aggregation", desc: "10-second non-overlapping state windows" },
    { label: "World Model Rollout", desc: "Latent transition simulation K=6 (+10s..+60s)" },
    { label: "ATT&CK Progression Mapping", desc: "Temporal saliency and tactic correlation" },
  ];

  return (
    <div className="rounded-xl border border-brand/40 bg-surface/95 p-6 shadow-2xl backdrop-blur">
      <div className="flex items-center gap-3 border-b border-line pb-4">
        <div className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-brand/10 text-brand border border-brand/30">
          <Cpu size={20} className="animate-pulse" />
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-brand" />
          </span>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-ink flex items-center gap-2">
            Predictive Model Inference in Progress
            {isJob && (
              <span className="rounded bg-brand/15 px-1.5 py-0.5 text-[10px] font-mono text-brand">
                Async Job
              </span>
            )}
          </h4>
          <p className="text-xs text-muted">
            Rolling network state forward across multiple horizons...
          </p>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {steps.map((s, idx) => (
          <div key={idx} className="flex items-start gap-3 text-xs">
            <div className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-brand/40 bg-brand/10 text-[10px] font-mono text-brand">
              {idx + 1}
            </div>
            <div className="flex-1">
              <div className="font-medium text-ink">{s.label}</div>
              <div className="text-[11px] text-muted">{s.desc}</div>
            </div>
            <div className="h-2 w-2 rounded-full bg-brand animate-ping" />
          </div>
        ))}
      </div>

      <div className="mt-5 flex items-center justify-between border-t border-line pt-3 text-[11px] font-mono text-muted">
        <span className="flex items-center gap-1.5">
          <CyberSpinner size={12} />
          {stage || "Running SENTINEL-WM blend..."}
        </span>
        <span>Please do not close this tab</span>
      </div>
    </div>
  );
}

/** Animated pulse dot for live telemetry status */
export function LivePulseBadge({
  connected = true,
  label = "LIVE STREAM",
}: {
  connected?: boolean;
  label?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-[11px] font-medium border transition-colors",
        connected
          ? "border-ok/30 bg-ok/10 text-ok"
          : "border-warn/30 bg-warn/10 text-warn"
      )}
    >
      <span className="relative flex h-2 w-2">
        {connected && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-ok opacity-75" />
        )}
        <span
          className={cn(
            "relative inline-flex rounded-full h-2 w-2",
            connected ? "bg-ok" : "bg-warn"
          )}
        />
      </span>
      {label}
    </span>
  );
}
