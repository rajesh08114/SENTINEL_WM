import { describe, expect, it, beforeEach } from "vitest";
import { useStore } from "../store";
import type { AnchorForecast, ForecastResponse } from "../types";

describe("useStore state management & persistence", () => {
  beforeEach(() => {
    useStore.getState().resetLive();
    useStore.getState().clearResult();
    useStore.setState({ activeSource: "live" });
  });

  it("initializes with expected default values", () => {
    const s = useStore.getState();
    expect(s.liveSessionId).toBeNull();
    expect(s.liveForecasts).toHaveLength(0);
    expect(s.result).toBeNull();
    expect(s.livePaused).toBe(false);
  });

  it("pushes and caps live forecasts properly", () => {
    const mockForecast = {
      alert: true,
      max_attack_prob: 0.88,
      horizon: [],
    } as unknown as AnchorForecast;

    useStore.getState().pushLiveForecast(mockForecast);
    expect(useStore.getState().liveForecasts).toHaveLength(1);
    expect(useStore.getState().liveForecasts[0].alert).toBe(true);

    // If paused, pushing should be ignored
    useStore.getState().toggleLivePause();
    expect(useStore.getState().livePaused).toBe(true);

    useStore.getState().pushLiveForecast({ ...mockForecast, alert: false });
    expect(useStore.getState().liveForecasts).toHaveLength(1); // unchanged
  });

  it("setting CSV does not wipe out existing forecast result", () => {
    const mockResult = {
      summary: { n_alerts: 2, alert_threshold: 0.5 },
      anchors: [],
      meta: { model: "SENTINEL-WM" },
    } as unknown as ForecastResponse;

    useStore.getState().setResult(mockResult);
    expect(useStore.getState().result).toBe(mockResult);
    expect(useStore.getState().activeSource).toBe("upload");

    // Setting a CSV file inspection should NOT wipe out the result
    useStore.getState().setCsv({
      name: "test.csv",
      header: ["a", "b"],
      rows: [],
      summary: {} as any,
      match: {} as any,
    });

    expect(useStore.getState().result).toBe(mockResult);
  });

  it("allows switching active source between live and upload", () => {
    useStore.getState().setActiveSource("upload");
    expect(useStore.getState().activeSource).toBe("upload");

    useStore.getState().setActiveSource("live");
    expect(useStore.getState().activeSource).toBe("live");
  });
});
