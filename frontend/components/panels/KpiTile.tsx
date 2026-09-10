import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

export function KpiTile({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "ok" | "danger" | "warn" | "muted";
}) {
  const toneCls = {
    ok: "text-ok",
    danger: "text-danger",
    warn: "text-warn",
    muted: "text-muted",
  };
  return (
    <Card className="flex flex-col gap-1 p-3">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className={cn("kpi text-2xl", tone && toneCls[tone])}>{value}</div>
      {sub != null && <div className="text-xs text-muted">{sub}</div>}
    </Card>
  );
}
