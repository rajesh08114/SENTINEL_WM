"use client";
import { Panel } from "@/components/ui/card";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { FeatureBars } from "@/components/charts/FeatureBars";
import { Saliency } from "@/components/charts/Saliency";
import { HorizonTable } from "./HorizonTable";
import { ProgressionRibbon } from "./ProgressionRibbon";
import { AttckPhaseTimeline } from "./AttckPhaseTimeline";
import type { AnchorForecast } from "@/lib/types";

export function AnchorDetail({
  anchor,
  threshold,
}: {
  anchor: AnchorForecast;
  threshold: number;
}) {
  const df = anchor.driving_features;
  const win = (anchor.meta?.window_index as number | undefined) ?? "";
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between rounded-lg border border-line bg-surface px-4 py-3 text-sm">
        <span>Anchor detail — window {String(win)}</span>
        <span
          className={
            "font-mono text-xs " + (anchor.alert ? "text-danger" : "text-muted")
          }
        >
          {anchor.alert
            ? `first alert +${anchor.first_alert_k} · lead ${anchor.lead_time_seconds}s`
            : "no alert in horizon"}
        </span>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Per-horizon forecast">
          <ForecastChart anchor={anchor} threshold={threshold} />
          <p className="text-[11px] text-muted">
            Solid = system detection_prob · dashed = world-model attack_prob · band =
            95% CI · red dashes = alert threshold.
          </p>
        </Panel>
        <Panel title="Horizon table">
          <HorizonTable anchor={anchor} threshold={threshold} />
        </Panel>
      </div>

      <Panel title="Kill-chain progression (next 6 windows)">
        <ProgressionRibbon horizon={anchor.horizon} />
      </Panel>

      <Panel title="MITRE ATT&CK phase mapping">
        <AttckPhaseTimeline horizon={anchor.horizon} />
      </Panel>

      {df && (df.top_features?.length || df.temporal_saliency_gradient) ? (
        <div className="grid gap-4 lg:grid-cols-2">
          {df.top_features?.length ? (
            <Panel title="Driving features">
              <FeatureBars df={df} />
              <p className="text-[11px] text-muted">
                Signed gradient×input contribution — red pushes toward attack, green
                toward benign.
              </p>
            </Panel>
          ) : null}
          {df.temporal_saliency_gradient ? (
            <Panel title="Temporal saliency">
              <Saliency
                gradient={df.temporal_saliency_gradient}
                attention={df.temporal_saliency_attention}
              />
              <p className="text-[11px] text-muted">
                Which of the L history windows the model leaned on.
              </p>
            </Panel>
          ) : null}
        </div>
      ) : (
        <p className="text-xs text-muted">
          Explanations were not requested for this run.
        </p>
      )}
    </section>
  );
}
