import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...p }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-9 w-full rounded-md border border-line bg-elevated px-3 text-sm text-ink placeholder:text-muted focus-visible:outline-none focus-visible:border-brand",
      className
    )}
    {...p}
  />
));
Input.displayName = "Input";

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...p }, ref) => (
  <select
    ref={ref}
    className={cn(
      "h-9 w-full rounded-md border border-line bg-elevated px-2 text-sm text-ink focus-visible:outline-none focus-visible:border-brand",
      className
    )}
    {...p}
  >
    {children}
  </select>
));
Select.displayName = "Select";

export function Label({ className, ...p }: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return (
    <label className={cn("text-xs font-medium text-muted", className)} {...p} />
  );
}

export function Badge({
  className,
  tone = "default",
  ...p
}: React.HTMLAttributes<HTMLSpanElement> & {
  tone?: "default" | "ok" | "warn" | "danger" | "info";
}) {
  const tones = {
    default: "border-line text-muted",
    ok: "border-ok/40 text-ok bg-ok/10",
    warn: "border-warn/40 text-warn bg-warn/10",
    danger: "border-danger/40 text-danger bg-danger/10",
    info: "border-info/40 text-info bg-info/10",
  } as const;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-[2px] font-mono text-[11px]",
        tones[tone],
        className
      )}
      {...p}
    />
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded bg-elevated", className)} />;
}

export function ConfBadge({ c }: { c?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-[2px] font-mono text-[11px]",
        `conf-${c || "Low"}`
      )}
    >
      {c || "—"}
    </span>
  );
}
