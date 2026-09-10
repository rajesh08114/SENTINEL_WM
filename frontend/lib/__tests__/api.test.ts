import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api";

const anchor = (over: Record<string, unknown> = {}) => ({
  alert: false,
  first_alert_k: null,
  lead_time_seconds: null,
  max_attack_prob: 0.2,
  horizon: Array.from({ length: 6 }, (_, i) => ({
    k: i + 1,
    horizon_seconds: (i + 1) * 10,
    attack_prob: 0.1,
    attack_ci: [0.05, 0.2],
    progression_state: "NORMAL",
    progression_dist: { NORMAL: 1 },
    attck: { mitre_tactic: "None", kill_chain_phase: "n/a" },
  })),
  ...over,
});

const goodForecast = {
  meta: { n_flows: 10, n_anchors: 1 },
  summary: { n_alerts: 0, alert_threshold: 0.5 },
  anchors: [anchor()],
};

function mockFetch(body: unknown, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status < 400,
    status,
    statusText: "",
    json: async () => body,
  });
}

afterEach(() => vi.restoreAllMocks());

describe("api zod parsing", () => {
  it("accepts a well-formed forecast job result", async () => {
    vi.stubGlobal("fetch", mockFetch(goodForecast));
    const r = await api.jobResult("j1");
    expect(r.anchors[0].horizon).toHaveLength(6);
    expect(r.summary.alert_threshold).toBe(0.5);
  });

  it("rejects a forecast whose horizon is the wrong length", async () => {
    const bad = { ...goodForecast, anchors: [anchor({ horizon: [] })] };
    vi.stubGlobal("fetch", mockFetch(bad));
    await expect(api.jobResult("j1")).rejects.toBeTruthy();
  });

  it("surfaces backend error detail as ApiError", async () => {
    vi.stubGlobal("fetch", mockFetch({ detail: "missing required columns" }, 422));
    await expect(api.meta()).rejects.toBeInstanceOf(ApiError);
  });
});
