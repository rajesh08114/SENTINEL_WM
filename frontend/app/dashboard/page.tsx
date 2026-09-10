import { Suspense } from "react";
import { DashboardView } from "./DashboardView";

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="text-sm text-muted">loading…</div>}>
      <DashboardView />
    </Suspense>
  );
}
