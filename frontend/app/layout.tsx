import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { TopBar } from "@/components/nav/TopBar";
import { SideNav } from "@/components/nav/SideNav";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";

export const metadata: Metadata = {
  title: "SENTINEL-WM Console",
  description:
    "SOC analyst console for SENTINEL-WM — network attack-progression forecasting with MITRE ATT&CK phase mapping.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <body className="min-h-screen bg-bg text-ink antialiased">
        <Providers>
          <div className="flex min-h-screen flex-col">
            <TopBar />
            <div className="flex flex-1 flex-col md:flex-row">
              <SideNav />
              <main className="min-w-0 flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-4 py-6 sm:px-6">
                  <ErrorBoundary>{children}</ErrorBoundary>
                </div>
              </main>
            </div>
            <footer className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-line px-4 py-3 text-xs text-muted sm:px-6">
              <span className="font-mono">SENTINEL-WM Console</span>
              <span>Research prototype · CIC-IDS-2017</span>
            </footer>
          </div>
        </Providers>
      </body>
    </html>
  );
}
