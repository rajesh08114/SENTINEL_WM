import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProgressionRibbon } from "../panels/ProgressionRibbon";
import { HorizonTable } from "../panels/HorizonTable";
import type { AnchorForecast, HorizonStep } from "@/lib/types";

const step = (k: number, over: Partial<HorizonStep> = {}): HorizonStep => ({
  k,
  horizon_seconds: k * 10,
  attack_prob: 0.2,
  attack_ci: [0.1, 0.3],
  detection_prob: 0.2,
  progression_state: "NORMAL",
  progression_dist: { NORMAL: 1 },
  attck: {
    mitre_tactic: "None",
    technique_ids: [],
    kill_chain_phase: "n/a",
    confidence: "Low",
  },
  ...over,
});

const horizon = [
  step(1, { progression_state: "PRE_ATTACK" }),
  step(2, { progression_state: "ONSET" }),
  step(3, { progression_state: "ACTIVE", detection_prob: 0.9 }),
  step(4),
  step(5),
  step(6),
];

describe("ProgressionRibbon", () => {
  it("renders one cell per horizon step with the state class", () => {
    const { container } = render(<ProgressionRibbon horizon={horizon} />);
    expect(container.querySelector(".st-PRE_ATTACK")).toBeTruthy();
    expect(container.querySelector(".st-ACTIVE")).toBeTruthy();
    expect(screen.getAllByText(/\+\d+s/).length).toBeGreaterThanOrEqual(6);
  });
});

describe("HorizonTable", () => {
  it("flags rows at or above the threshold", () => {
    const anchor = { horizon } as AnchorForecast;
    render(<HorizonTable anchor={anchor} threshold={0.5} />);
    const alerts = screen.getAllByText("ALERT");
    expect(alerts).toHaveLength(1); // only the 0.9 step
  });
});
