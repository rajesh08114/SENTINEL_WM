import Link from "next/link";
import { ApiBaseField } from "@/components/common/ApiBaseField";
import { HealthDot } from "@/components/common/HealthDot";
import { ThemeToggle } from "@/components/common/ThemeToggle";

export function TopBar() {
  return (
    <header className="flex items-center gap-4 border-b border-line bg-surface px-4 py-2.5 sm:px-6">
      <Link href="/" className="font-mono font-semibold tracking-tight text-ink">
        <span className="text-brand">SENTINEL</span>-WM{" "}
        <span className="text-muted">Console</span>
      </Link>
      <span className="flex-1" />
      <ApiBaseField />
      <HealthDot />
      <ThemeToggle />
    </header>
  );
}
