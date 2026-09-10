"use client";
import * as React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";

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
      {children}
      <Toaster
        theme="dark"
        position="bottom-right"
        toastOptions={{ style: { fontFamily: "Fira Sans, sans-serif" } }}
      />
    </QueryClientProvider>
  );
}
