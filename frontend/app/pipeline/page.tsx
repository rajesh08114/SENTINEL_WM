"use client";
import Link from "next/link";
import { Check } from "lucide-react";
import { Panel } from "@/components/ui/card";
import { KpiTile } from "@/components/panels/KpiTile";
import { useStore } from "@/lib/store";
import { num, round } from "@/lib/format";
import { cn } from "@/lib/utils";

const STEPS: [string, string, string][] = [
  ["Ingest", "normalise_upload", "Canonicalize CICFlowMeter aliases, derive flow_start_epoch, inject benign labels. 422 if a required column is absent."],
  ["Clean", "clean_flow_frame", "Coerce numerics, winsorize with train-day percentile stats (no test-time fitting), drop unusable rows."],
  ["State windows", "build_state_windows", "Aggregate flows into 10 s windows → 53 features each; derive progression + ATT&CK stage labels (display only)."],
  ["Sequences", "windows_to_tensors", "Slide L=12 with a window-index contiguity guard; scale with the bundle's state_scaler.pkl; log1p the Δt channel."],
  ["Forecast", "simulate_anchor", "Per anchor: world-model K-step MC rollout + system blend + assess_forecast (ATT&CK) + gradient/attention explainer."],
];

export default function PipelinePage() {
  const csv = useStore((s) => s.csv);
  const result = useStore((s) => s.result);
  const done = result ? 5 : csv ? 1 : 0;

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-xl font-semibold">Processing pipeline</h1>
      {!csv && !result && (
        <p className="text-sm text-muted">
          No data selected yet — pick a source on{" "}
          <Link href="/sources" className="text-brand underline decoration-dotted">
            Data Sources
          </Link>
          .
        </p>
      )}

      <ol className="flex flex-col gap-3">
        {STEPS.map(([name, fn, desc], i) => (
          <li
            key={name}
            className={cn(
              "flex gap-4 rounded-lg border border-line bg-surface p-4",
              i < done ? "" : "opacity-60"
            )}
          >
            <span
              className={cn(
                "mono grid h-7 w-7 shrink-0 place-items-center rounded-full border text-sm",
                i < done
                  ? "border-brand bg-brand text-black"
                  : i === done
                    ? "border-brand text-brand"
                    : "border-line text-muted"
              )}
            >
              {i < done ? <Check size={13} /> : i + 1}
            </span>
            <div className="flex min-w-0 flex-col gap-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold">{name}</span>
                <code className="text-xs text-muted">{fn}</code>
              </div>
              <p className="text-xs leading-snug text-muted">{desc}</p>
              <StepDetail i={i} />
            </div>
          </li>
        ))}
      </ol>

      {result && (
        <Panel title="Result summary">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <KpiTile label="Flows" value={num(result.meta.n_flows)} />
            <KpiTile label="Windows" value={num(result.meta.n_windows)} />
            <KpiTile
              label="Anchors"
              value={num(result.meta.n_anchors)}
              sub={`L=${result.meta.history_windows} · K=${result.meta.horizon_steps}`}
            />
            <KpiTile
              label="Alert windows"
              value={num(result.summary.n_alerts)}
              sub={`threshold ${round(result.summary.alert_threshold, 3)}`}
              tone={result.summary.n_alerts ? "danger" : "ok"}
            />
          </div>
          <Link
            href="/dashboard"
            className="w-max text-sm text-brand underline decoration-dotted"
          >
            Open the SOC dashboard →
          </Link>
        </Panel>
      )}
    </div>
  );
}

function StepDetail({ i }: { i: number }) {
  const csv = useStore((s) => s.csv);
  const result = useStore((s) => s.result);
  if (result) {
    const m = result.meta;
    const map: Record<number, string> = {
      0: `${num(m.n_flows)} flows accepted · model ${m.model || "system"}${m.family_hint ? " · hint " + m.family_hint : ""}`,
      1: `winsorized on train-day stats · feature dim ${m.feature_dim}`,
      2: `${num(m.n_windows)} state windows · ${m.window_seconds}s each`,
      3: `${num(m.n_anchors)} anchors · scaled ${m.history_windows}×${m.feature_dim} tensors`,
      4: `${num(m.n_anchors)} forecasts · ${m.horizon_steps} horizon steps each`,
    };
    return <div className="mono mt-0.5 text-xs text-brand">{map[i]}</div>;
  }
  if (csv && i <= 2) {
    const s = csv.summary;
    const map: Record<number, string> = {
      0: `${num(s.rowCount)} rows parsed · ${csv.match.requiredMiss.length ? csv.match.requiredMiss.length + " required missing" : "contract OK"}`,
      1: `${csv.match.aliased.length} aliases mapped · ${csv.match.tier2Hit.length} Tier-2 present`,
      2:
        s.estWindows != null
          ? `~${s.estWindows} windows expected (${num(Math.round(s.spanSeconds || 0))}s span)`
          : "time span unknown",
    };
    return <div className="mono mt-0.5 text-xs text-muted">{map[i]}</div>;
  }
  return null;
}
