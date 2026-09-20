"use client";
import Link from "next/link";
import { ApiBaseField } from "@/components/common/ApiBaseField";
import { HealthDot } from "@/components/common/HealthDot";
import { ThemeToggle } from "@/components/common/ThemeToggle";
import { useStore } from "@/lib/store";

export function TopBar() {
  const liveSessionId = useStore((s) => s.liveSessionId);
  const liveConnected = useStore((s) => s.liveConnected);
  const liveForecastsCount = useStore((s) => s.liveForecasts.length);

  return (
    <header className="flex items-center gap-4 border-b border-line bg-surface px-4 py-2.5 sm:px-6">
      <Link href="/" className="font-mono font-semibold tracking-tight text-ink">
        <span className="text-brand">SENTINEL</span>-WM{" "}
        <span className="text-muted">Console</span>
      </Link>

      {liveSessionId && (
        <Link
          href="/live"
          className="hidden sm:inline-flex items-center gap-2 rounded-full border border-ok/30 bg-ok/10 px-2.5 py-1 text-xs font-mono font-medium text-ok hover:bg-ok/20 transition-all"
        >
          <span className="relative flex h-2 w-2">
            {liveConnected && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-ok opacity-75" />
            )}
            <span className="relative inline-flex rounded-full h-2 w-2 bg-ok" />
          </span>
          <span>LIVE: {liveForecastsCount} frames</span>
        </Link>
      )}

      <span className="flex-1" />
      <ApiBaseField />
      <HealthDot />
      <ThemeToggle />
    </header>
  );
}
