"use client";
import * as React from "react";
import { Input, Label, Select } from "@/components/ui/primitives";

export interface ScenarioValues {
  scenario: string;
  rate: number;
  duration_s: number;
  speed: number;
  seed: number;
}

const SCENARIOS = [
  ["portscan", "Port scan — SYN fan-out to many ports"],
  ["dos_hulk", "DoS (Hulk) — high-rate flood to one victim"],
  ["bruteforce", "Brute force — repeated auth attempts (SSH/RDP)"],
  ["botnet_c2", "Botnet C2 — periodic beacons to a fixed external IP"],
  ["exfil", "Exfiltration — sustained large outbound transfer"],
  ["benign", "Benign only — baseline, no attack ramp"],
];

export function ScenarioForm({
  value,
  onChange,
}: {
  value: ScenarioValues;
  onChange: (v: ScenarioValues) => void;
}) {
  const set = (patch: Partial<ScenarioValues>) => onChange({ ...value, ...patch });
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Label className="sm:col-span-2 flex flex-col gap-1">
        scenario
        <Select
          value={value.scenario}
          onChange={(e) => set({ scenario: e.target.value })}
        >
          {SCENARIOS.map(([v, label]) => (
            <option key={v} value={v}>
              {label}
            </option>
          ))}
        </Select>
      </Label>
      <Label className="flex flex-col gap-1">
        benign rate (flows/s)
        <Input
          type="number"
          min={1}
          max={500}
          value={value.rate}
          onChange={(e) => set({ rate: Number(e.target.value) })}
        />
      </Label>
      <Label className="flex flex-col gap-1">
        duration (scenario seconds)
        <Input
          type="number"
          min={30}
          value={value.duration_s}
          onChange={(e) => set({ duration_s: Number(e.target.value) })}
        />
      </Label>
      <Label className="flex flex-col gap-1">
        speed (× real time)
        <Input
          type="number"
          min={1}
          max={120}
          value={value.speed}
          onChange={(e) => set({ speed: Number(e.target.value) })}
        />
      </Label>
      <Label className="flex flex-col gap-1">
        seed
        <Input
          type="number"
          value={value.seed}
          onChange={(e) => set({ seed: Number(e.target.value) })}
        />
      </Label>
      <p className="sm:col-span-2 text-[11px] text-muted">
        A speed multiplier compresses scenario time so the timeline fills in seconds. At
        speed 30, the first forecast lands in ~5 s.
      </p>
    </div>
  );
}
