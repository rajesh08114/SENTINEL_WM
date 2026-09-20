import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  CyberSpinner,
  RadarScanner,
  ForecastProgress,
  LivePulseBadge,
} from "../ui/loading";

describe("Loading animations", () => {
  it("renders CyberSpinner with label", () => {
    render(<CyberSpinner label="Testing model..." />);
    expect(screen.getByText("Testing model...")).toBeInTheDocument();
  });

  it("renders RadarScanner with telemetry buffer progress", () => {
    render(
      <RadarScanner
        flowsReceived={1500}
        windowsCount={4}
        targetWindows={12}
        statusText="Streaming frames"
      />
    );
    expect(screen.getByText(/Awaiting initial telemetry history/i)).toBeInTheDocument();
    expect(screen.getByText("4 / 12 windows (33%)")).toBeInTheDocument();
    expect(screen.getByText("1,500")).toBeInTheDocument();
  });

  it("renders ForecastProgress with stage description", () => {
    render(<ForecastProgress stage="World Model Rollout K=6" />);
    expect(screen.getByText(/Predictive Model Inference in Progress/i)).toBeInTheDocument();
    expect(screen.getByText("World Model Rollout K=6")).toBeInTheDocument();
  });

  it("renders LivePulseBadge with proper connection tone", () => {
    const { rerender } = render(<LivePulseBadge connected={true} label="LIVE" />);
    expect(screen.getByText("LIVE")).toBeInTheDocument();

    rerender(<LivePulseBadge connected={false} label="RECONNECTING" />);
    expect(screen.getByText("RECONNECTING")).toBeInTheDocument();
  });
});
