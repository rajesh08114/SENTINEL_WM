"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { KpiTile } from "@/components/panels/KpiTile";
import { round } from "@/lib/format";

export function SystemStatus() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000 });
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const agent = useQuery({ queryKey: ["agent"], queryFn: api.agentStatus, refetchInterval: 10_000 });
  const sessions = useQuery({ queryKey: ["live"], queryFn: api.listLiveSessions, refetchInterval: 10_000 });

  const online = !!health.data && !health.isError;
  const m = meta.data;
  const running = (sessions.data || []).filter((s) => s.state !== "stopped").length;

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <KpiTile
        label="Backend"
        value={online ? "ONLINE" : "OFFLINE"}
        sub={online ? (health.data?.model_loaded ? "bundle loaded" : "no bundle") : "start the API service"}
        tone={online ? "ok" : "danger"}
      />
      <KpiTile
        label="Serve mode"
        value={(m?.serve_mode || "—").toUpperCase()}
        sub={m?.blend_members?.length ? `blend: ${m.blend_members.join(" + ")}` : "world model + members"}
      />
      <KpiTile
        label="Horizon"
        value={m ? `${m.history_windows}→${m.horizon_steps}` : "12→6"}
        sub={m ? `${m.window_seconds}s windows · ${m.feature_dim} features` : "10s windows"}
      />
      <KpiTile
        label="Live"
        value={`${running} session${running === 1 ? "" : "s"}`}
        sub={
          agent.data?.connected
            ? `capture agent: ${agent.data.count} connected`
            : m?.blend_weight != null
              ? `blend weight ${round(m.blend_weight, 2)}`
              : "no capture agent"
        }
      />
    </div>
  );
}
