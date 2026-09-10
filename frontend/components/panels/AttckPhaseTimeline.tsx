import { ConfBadge } from "@/components/ui/primitives";
import { Card } from "@/components/ui/card";
import type { HorizonStep } from "@/lib/types";

export function AttckPhaseTimeline({ horizon }: { horizon: HorizonStep[] }) {
  return (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {horizon.map((h, i) => {
        const a = h.attck || ({} as HorizonStep["attck"]);
        return (
          <Card key={i} className="flex flex-col gap-1.5 p-3">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs text-muted">
                +{h.horizon_seconds}s
              </span>
              <ConfBadge c={a.confidence} />
            </div>
            <div className="flex items-center gap-1.5 text-sm font-semibold">
              {a.mitre_tactic || "—"}
              {(a as any).family_inferred && (
                <span className="rounded border border-warn/40 bg-warn/10 px-1 py-[1px] text-[9px] font-normal uppercase text-warn">
                  inferred
                </span>
              )}
            </div>
            <div className="text-xs text-muted">{a.kill_chain_phase || ""}</div>
            {(a as any).possible_family && (
              <div className="text-[11px] text-muted">
                possible family:{" "}
                <span className="text-ink">{(a as any).possible_family}</span>{" "}
                <span className="opacity-70">
                  ({(a as any).possible_family_confidence || "low"} conf · advisory)
                </span>
              </div>
            )}
            {a.technique_ids?.length ? (
              <div className="flex flex-wrap gap-1">
                {a.technique_ids.map((t) => (
                  <span
                    key={t}
                    className="rounded-full border border-line px-2 py-[2px] font-mono text-[11px]"
                  >
                    {t}
                  </span>
                ))}
              </div>
            ) : null}
            {a.dominant_family && (
              <div className="text-xs">
                <span className="text-muted">family: </span>
                {a.dominant_family}
                {a.family_transition ? (
                  <span className="text-muted"> ({a.family_transition})</span>
                ) : null}
              </div>
            )}
            {a.rationale && (
              <p className="text-xs leading-snug text-muted">{a.rationale}</p>
            )}
          </Card>
        );
      })}
    </div>
  );
}
