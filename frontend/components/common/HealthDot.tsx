"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export function HealthDot() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });
  const state = isError
    ? "offline"
    : isLoading
      ? "connecting"
      : data?.model_loaded
        ? "ok"
        : "degraded";
  const map = {
    ok: ["bg-ok", "online"],
    degraded: ["bg-warn", "no bundle"],
    connecting: ["bg-warn animate-pulse", "connecting"],
    offline: ["bg-danger", "offline"],
  } as const;
  const [dot, label] = map[state];
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted">
      <span className={cn("h-2 w-2 rounded-full", dot)} />
      <span>{label}</span>
    </span>
  );
}
