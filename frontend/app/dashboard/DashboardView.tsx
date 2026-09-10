"use client";
import * as React from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Panel } from "@/components/ui/card";
import { KpiTile } from "@/components/panels/KpiTile";
import { AnchorTable } from "@/components/panels/AnchorTable";
import { AnchorDetail } from "@/components/panels/AnchorDetail";
import { EmptyState } from "@/components/common/EmptyState";
import { api } from "@/lib/api";
import { openLiveStream } from "@/lib/ws";
import { useStore } from "@/lib/store";
import { num, pct, round } from "@/lib/format";
import type { AnchorForecast } from "@/lib/types";

export function DashboardView() {
  const params = useSearchParams();
  const sessionParam = params.get("session");

  const result = useStore((s) => s.result);
  const selected = useStore((s) => s.selectedAnchor);
  const selectAnchor = useStore((s) => s.selectAnchor);
  const liveForecasts = useStore((s) => s.liveForecasts);
  const pushLiveForecast = useStore((s) => s.pushLiveForecast);
  const resetLive = useStore((s) => s.resetLive);

  const { data: meta } = useQuery({ queryKey: ["meta"], queryFn: api.meta });

  // when arriving with ?session=, subscribe and accumulate
  React.useEffect(() => {
    if (!sessionParam) return;
    resetLive();
    const stream = openLiveStream(sessionParam, {
      onForecast: (f) => pushLiveForecast(f),
    });
    return () => stream.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionParam]);

  const liveMode = !!sessionParam;
  const anchors: AnchorForecast[] = liveMode ? liveForecasts : result?.anchors ?? [];
  const threshold =
    (liveMode ? undefined : result?.summary.alert_threshold) ??
    meta?.alert_threshold ??
    0.5;

  if (!anchors.length) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-xl font-semibold">SOC dashboard</h1>
        {liveMode ? (
          <EmptyState
            icon="◉"
            title="Waiting for the first forecast…"
            body="This live session needs ~12 windows of history before the model emits an anchor."
          />
        ) : (
          <EmptyState
            icon="▣"
            title="No forecast loaded."
            body="Upload a flow CSV, or run a session on the Live page."
            cta={{ href: "/sources", label: "Upload a flow CSV →" }}
          />
        )}
      </div>
    );
  }

  const sel = Math.min(selected, anchors.length - 1);
  const a = anchors[sel];
  const nAlerts = anchors.filter((x) => x.alert).length;
  const peak = Math.max(...anchors.map((x) => x.max_detection_prob ?? x.max_attack_prob));
  const leads = anchors
    .map((x) => x.lead_time_seconds)
    .filter((v): v is number => v != null && v >= 0);
  const phases = liveMode ? [] : result?.summary.phases ?? [];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <h1 className="text-xl font-semibold">
          SOC dashboard {liveMode && <span className="text-brand">· live</span>}
        </h1>
        <div className="mono text-xs text-muted">
          {liveMode
            ? `session ${sessionParam?.slice(0, 8)} · ${anchors.length} forecasts`
            : `${result?.meta.model || "system"} · ${result?.meta.n_anchors} anchors`}{" "}
          · threshold {round(threshold, 3)}
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
          value={String(phases.length || new Set(anchors.map((x) => x.horizon[0]?.attck?.mitre_tactic)).size)}
          sub={
            (phases.slice(0, 3) as any[])
              .map((p) => (typeof p === "string" ? p : p.mitre_tactic))
              .join(", ") || anchors[0]?.horizon[0]?.attck?.mitre_tactic || "none"
          }
        />
      </div>

      <Panel title="Anchors">
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
