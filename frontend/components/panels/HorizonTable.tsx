import { pct } from "@/lib/format";
import type { AnchorForecast } from "@/lib/types";
import { cn } from "@/lib/utils";

export function HorizonTable({
  anchor,
  threshold,
}: {
  anchor: AnchorForecast;
  threshold: number;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            {["horizon", "P(atk) sys", "P(atk) WM", "95% CI", "progression", "ATT&CK tactic", "flag"].map(
              (h) => (
                <th key={h}>{h}</th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {anchor.horizon.map((h, i) => {
            const p = h.detection_prob ?? h.attack_prob;
            const alert = p >= threshold;
            return (
              <tr key={i}>
                <td className="mono">+{h.horizon_seconds}s</td>
                <td className="mono">{pct(p)}</td>
                <td className="mono">{pct(h.attack_prob)}</td>
                <td className="mono text-muted">
                  {pct(h.attack_ci[0])} – {pct(h.attack_ci[1])}
                </td>
                <td>{(h.progression_state || "NORMAL").replace("_", " ")}</td>
                <td>
                  {h.attck?.mitre_tactic || "—"}
                  {(h.attck as any)?.family_inferred && (
                    <span className="ml-1 text-[10px] text-warn">(inferred)</span>
                  )}
                </td>
                <td className={cn(alert ? "font-semibold text-danger" : "text-muted")}>
                  {alert ? "ALERT" : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
