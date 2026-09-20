"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity, Boxes, Cpu, FlaskConical, GaugeCircle, LayoutGrid, Radio, Workflow,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useStore } from "@/lib/store";

export const NAV = [
  { href: "/", label: "Overview", icon: LayoutGrid },
  { href: "/research", label: "Research", icon: FlaskConical },
  { href: "/sources", label: "Data Sources", icon: Boxes },
  { href: "/pipeline", label: "Pipeline", icon: Workflow },
  { href: "/dashboard", label: "SOC Dashboard", icon: GaugeCircle },
  { href: "/live", label: "Live Telemetry", icon: Radio },
  { href: "/architecture", label: "Architecture", icon: Activity },
  { href: "/model", label: "Model Card", icon: Cpu },
];

export function SideNav() {
  const path = usePathname();
  const liveSessionId = useStore((s) => s.liveSessionId);
  const liveConnected = useStore((s) => s.liveConnected);

  return (
    <nav
      aria-label="Primary"
      className="flex shrink-0 gap-1 overflow-x-auto border-b border-line bg-surface p-2 md:w-[232px] md:flex-col md:overflow-visible md:border-b-0 md:border-r md:p-3"
    >
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        const isLiveItem = href === "/live" || href === "/dashboard";
        const hasActiveLiveStream = isLiveItem && !!liveSessionId;

        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex items-center gap-2 whitespace-nowrap rounded-md px-2.5 py-2 text-sm text-muted transition-colors hover:bg-elevated hover:text-ink",
              active &&
                "bg-brand/10 text-ink shadow-[inset_2px_0_0_var(--brand)]"
            )}
          >
            <Icon size={15} className={active ? "text-brand" : undefined} />
            <span className="hidden md:inline flex-1">{label}</span>
            {hasActiveLiveStream && (
              <span className="relative hidden md:flex h-2 w-2 ml-auto">
                {liveConnected && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-ok opacity-75" />
                )}
                <span className="relative inline-flex rounded-full h-2 w-2 bg-ok" />
              </span>
            )}
          </Link>
        );
      })}
    </nav>
  );
}
