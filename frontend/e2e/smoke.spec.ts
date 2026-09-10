import { test, expect } from "@playwright/test";

// All routes render without a backend (health shows offline, views show empty
// states). The synthetic-session test needs a backend on :8000 and is skipped
// when it isn't reachable.

const ROUTES = ["/", "/research", "/sources", "/pipeline", "/dashboard", "/live", "/architecture", "/model"];

for (const path of ROUTES) {
  test(`route ${path} renders`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    await page.goto(path);
    await expect(page.locator("header").first()).toBeVisible();
    await expect(page.locator("nav[aria-label='Primary']")).toBeVisible();
    expect(errors, errors.join("\n")).toHaveLength(0);
  });
}

test("synthetic live session streams a forecast", async ({ page, request }) => {
  const base = process.env.E2E_API || "http://localhost:8000";
  let up = false;
  try {
    const r = await request.get(`${base}/health`, { timeout: 2000 });
    up = r.ok();
  } catch {
    up = false;
  }
  test.skip(!up, `backend not reachable at ${base}`);

  await page.addInitScript(
    ([b]) => window.localStorage.setItem("sentinel.apiBase", b),
    [base]
  );
  await page.goto("/live");

  await page.getByRole("combobox").selectOption("dos_hulk");
  await page.getByLabel("speed (× real time)").fill("45");
  await page.getByRole("button", { name: /start/i }).click();

  // a forecast card / KPI shows a percentage within the model's warm-up window
  await expect(page.getByText("Incoming forecasts")).toBeVisible();
  await expect(page.locator(".kpi", { hasText: "%" }).first()).toBeVisible({
    timeout: 45_000,
  });

  await page.getByRole("button", { name: /stop/i }).click();
});
