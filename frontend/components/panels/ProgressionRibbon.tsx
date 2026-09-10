import { PROGRESSION_STATES } from "@/lib/schema";
import { pct } from "@/lib/format";
import type { HorizonStep } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ProgressionRibbon({ horizon }: { horizon: HorizonStep[] }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex gap-1">
        {horizon.map((h, i) => {
          const st = h.progression_state || "NORMAL";
          const top = Object.entries(h.progression_dist || {}).sort(
            (a, b) => b[1] - a[1]
          )[0];
          return (
            <div
              key={i}
              className={cn("flex-1 rounded-sm px-1 py-2 text-center", `st-${st}`)}
              title={`+${h.horizon_seconds}s · ${st} · ${top ? pct(top[1]) : ""}`}
            >
              <div className="font-mono text-[10px] opacity-80">
                +{h.horizon_seconds}s
              </div>
              <div className="text-[11px] font-semibold leading-tight">
                {st.replace("_", " ")}
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted">
        {PROGRESSION_STATES.map((s) => (
          <span key={s} className="inline-flex items-center gap-1">
            <i className={cn("inline-block h-2.5 w-2.5 rounded-sm", `dot-${s}`)} />
            {s.replace("_", " ")}
          </span>
        ))}
      </div>
    </div>
  );
}
