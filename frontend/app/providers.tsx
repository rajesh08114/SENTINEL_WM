"use client";
import * as React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { useStore } from "@/lib/store";

function GlobalLiveManager() {
  const reconnectLiveIfActive = useStore((s) => s.reconnectLiveIfActive);

  React.useEffect(() => {
    // Attempt re-syncing to active live session if browser was reloaded or reopened
    reconnectLiveIfActive();
  }, [reconnectLiveIfActive]);

  return null;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [qc] = React.useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 10_000 },
        },
      })
  );

  return (
    <QueryClientProvider client={qc}>
      <GlobalLiveManager />
      {children}
      <Toaster
        theme="dark"
        position="bottom-right"
        duration={5000}
        toastOptions={{
          duration: 5000,
          style: { fontFamily: "Fira Sans, sans-serif" },
        }}
      />
    </QueryClientProvider>
  );
}
