import { defineConfig, devices } from "@playwright/test";

// e2e runs against the static export served by `next start` (or `serve out`).
// The backend is stubbed via route interception inside the specs.
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: true,
  retries: 0,
  use: { baseURL: "http://localhost:3100", trace: "on-first-retry" },
  webServer: {
    command: "npx next start -p 3100",
    url: "http://localhost:3100",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
