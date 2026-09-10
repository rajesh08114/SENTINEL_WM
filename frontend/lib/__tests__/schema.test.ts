import { describe, expect, it } from "vitest";
import { matchColumns, REQUIRED } from "../schema";
import { parseCSV, summarise } from "../csv";

const GOOD = [
  "flow_start_epoch", "Source IP", "Destination IP", "Destination Port",
  "Protocol", "Flow Duration", "Flow IAT Mean", "Total Fwd Packets",
  "Total Backward Packets", "Total Length of Fwd Packets",
  "Total Length of Bwd Packets", "SYN Flag Count", "Label",
];

describe("matchColumns", () => {
  it("accepts a complete header", () => {
    const m = matchColumns(GOOD);
    expect(m.ok).toBe(true);
    expect(m.requiredMiss).toHaveLength(0);
    expect(m.flagHit.some((f) => f.startsWith("flag_true_syn"))).toBe(true);
    expect(m.labelled).toBe(true);
  });

  it("flags a missing required column", () => {
    const m = matchColumns(GOOD.filter((c) => c !== "Destination Port"));
    expect(m.ok).toBe(false);
    expect(m.requiredMiss).toContain("Destination Port");
  });

  it("maps CICFlowMeter lower-case aliases", () => {
    const hdr = ["timestamp", "src ip", "dst ip", "dst port", "protocol",
      "flow duration", "flow iat mean", "tot fwd pkts", "tot bwd pkts",
      "totlen fwd pkts", "totlen bwd pkts"];
    const m = matchColumns(hdr);
    expect(m.aliased.map((a) => a.to)).toContain("Destination Port");
    // timestamp satisfies the flow_start_epoch requirement
    expect(m.requiredMiss).not.toContain("flow_start_epoch");
    expect(m.synthTimeAxis).toBe(true);
  });
});

describe("parseCSV + summarise", () => {
  it("parses quoted fields and summarises the time span", () => {
    const csv =
      'flow_start_epoch,Protocol,Label\n' +
      '1000,6,"BENIGN"\n' +
      '1030,17,PortScan\n';
    const { header, rows } = parseCSV(csv);
    expect(header).toEqual(["flow_start_epoch", "Protocol", "Label"]);
    expect(rows).toHaveLength(2);
    const s = summarise(header, rows);
    expect(s.rowCount).toBe(2);
    expect(s.spanSeconds).toBe(30);
    expect(s.estWindows).toBe(3);
    expect(s.protocols.map(([k]) => k).sort()).toEqual(["TCP", "UDP"]);
  });
});

it("REQUIRED is the 11-column contract", () => {
  expect(REQUIRED).toHaveLength(11);
});
