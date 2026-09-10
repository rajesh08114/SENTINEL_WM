"use client";
import { pct } from "@/lib/format";
import type { AnchorForecast } from "@/lib/types";
import { cn } from "@/lib/utils";

export function AnchorTable({
  anchors,
  threshold,
  selected,
  onSelect,
  fallbackModel,
}: {
  anchors: AnchorForecast[];
  threshold: number;
  selected: number;
  onSelect: (i: number) => void;
  fallbackModel?: string;
}) {
  return (
    <div className="max-h-[340px] overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            {["#", "window", "alert", "first alert", "lead time", "peak P(atk)", "model", "top tactic"].map(
              (h) => (
                <th key={h}>{h}</th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {anchors.map((x, i) => (
            <tr
              key={i}
              aria-selected={i === selected}
              className="cursor-pointer"
              onClick={() => onSelect(i)}
            >
              <td className="mono">{i}</td>
              <td className="mono text-muted">
                {String(
                  (x.meta?.window_index as number | undefined) ??
                    (x.meta?.stream_window as number | undefined) ??
                    "—"
                )}
              </td>
              <td className={cn(x.alert ? "font-semibold text-danger" : "text-muted")}>
                {x.alert ? "ALERT" : "—"}
              </td>
              <td className="mono">
                {x.first_alert_k != null ? `+${x.first_alert_k}` : "—"}
              </td>
              <td className="mono">
                {x.lead_time_seconds != null ? `${x.lead_time_seconds}s` : "—"}
              </td>
              <td className="mono">{pct(x.max_detection_prob ?? x.max_attack_prob)}</td>
              <td className="text-xs text-muted">
                {x.detection_model || fallbackModel || "system"}
              </td>
              <td className="text-xs">
                {x.horizon?.[0]?.attck?.mitre_tactic || "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
