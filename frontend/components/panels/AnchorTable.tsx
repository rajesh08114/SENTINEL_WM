"use client";
import * as React from "react";
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
            {["#", "window", "alert", "rules", "first alert", "lead time", "peak P(atk)", "top tactic"].map(
              (h) => (
                <th key={h}>{h}</th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {anchors.map((x, i) => {
            const currentWin =
              (x.meta?.window_index as number | undefined) ??
              (x.meta?.stream_window as number | undefined);
            const prevWin =
              i > 0
                ? ((anchors[i - 1].meta?.window_index as number | undefined) ??
                  (anchors[i - 1].meta?.stream_window as number | undefined))
                : undefined;
            const hasGap =
              currentWin != null && prevWin != null && currentWin - prevWin > 1;

            return (
              <React.Fragment key={i}>
                {hasGap && (
                  <tr className="bg-elevated/40 text-muted">
                    <td
                      colSpan={8}
                      className="py-1 px-3 text-center text-[11px] font-mono text-muted tracking-wider border-y border-line/60"
                    >
                      ⚡ Session boundary / temporal gap (+{(currentWin - prevWin) * 10}s · {currentWin - prevWin - 1} unmonitored windows)
                    </td>
                  </tr>
                )}
                <tr
                  aria-selected={i === selected}
                  className="cursor-pointer transition-colors"
                  onClick={() => onSelect(i)}
                >
                  <td className="mono">{i}</td>
                  <td className="mono text-muted">
                    {String(currentWin ?? "—")}
                  </td>
                  <td className={cn(x.alert ? "font-semibold text-danger" : "text-muted")}>
                    {x.alert ? "ALERT" : "—"}
                  </td>
                  <td className="text-xs">
                    {x.matched_rules && x.matched_rules.length > 0 ? (
                      <span
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 font-mono text-[10px] border border-amber-500/30"
                        title={x.matched_rules.map((r) => `${r.rule_id}: ${r.rule_name} (${r.severity})`).join("\n")}
                      >
                        {x.matched_rules.length} match
                      </span>
                    ) : (
                      <span className="text-muted text-[11px]">—</span>
                    )}
                  </td>
                  <td className="mono">
                    {x.first_alert_k != null ? `+${x.first_alert_k}` : "—"}
                  </td>
                  <td className="mono">
                    {x.lead_time_seconds != null ? `${x.lead_time_seconds}s` : "—"}
                  </td>
                  <td className="mono">{pct(x.max_detection_prob ?? x.max_attack_prob)}</td>
                  <td className="text-xs">
                    {x.horizon?.[0]?.attck?.mitre_tactic || "—"}
                  </td>
                </tr>
              </React.Fragment>
            );
          })}
        </tbody>

      </table>
    </div>
  );
}
