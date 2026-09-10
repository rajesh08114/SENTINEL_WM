"use client";
import { SCOREBOARD, HORIZON_F1 } from "@/lib/research-data";
import { cn } from "@/lib/utils";

const num = (n: number | null, d = 3) => (n == null ? "—" : n.toFixed(d));
const kparams = (n: number | null) => (n == null ? "—" : `${(n / 1000).toFixed(0)}k`);

const tierClass: Record<string, string> = {
  system: "text-brand font-semibold",
  member: "text-info",
  reference: "text-muted italic",
  other: "text-ink",
};

export function Scoreboard() {
  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            <th>Model</th>
            <th title="best-F1 threshold, tuned on validation">F1</th>
            <th title="best achievable by sweeping the threshold on test (ceiling)">F1*</th>
            <th title="threshold-free">PR-AUC</th>
            <th title="threshold-free">AUROC</th>
            <th title="progression-state accuracy">ProgAcc</th>
            <th>Params</th>
            <th title="inference ms / anchor">ms</th>
          </tr>
        </thead>
        <tbody>
          {SCOREBOARD.map((r) => (
            <tr key={r.model}>
              <td className={cn("mono text-xs", tierClass[r.tier])}>{r.model}</td>
              <td className="mono">{num(r.f1)}</td>
              <td className="mono">{num(r.f1star)}</td>
              <td className="mono">{num(r.prauc)}</td>
              <td className="mono">{num(r.auroc)}</td>
              <td className="mono text-muted">{num(r.progAcc, 3)}</td>
              <td className="mono text-muted">{kparams(r.params)}</td>
              <td className="mono text-muted">{r.inferMs.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function HorizonF1Table() {
  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            <th>Model</th>
            {["+10s", "+20s", "+30s", "+40s", "+50s", "+60s"].map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {HORIZON_F1.map((r) => (
            <tr key={r.model}>
              <td className="mono text-xs">{r.model}</td>
              {r.f1.map((v, i) => (
                <td key={i} className="mono">
                  {v.toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
