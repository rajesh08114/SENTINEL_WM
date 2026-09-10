"use client";
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { AnchorForecast } from "@/lib/types";

const css = (v: string) =>
  typeof window === "undefined"
    ? ""
    : getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export function ForecastChart({
  anchor,
  threshold,
}: {
  anchor: AnchorForecast;
  threshold: number;
}) {
  const brand = css("--brand") || "#2ee66b";
  const info = css("--info") || "#5aa9ff";
  const danger = css("--danger") || "#ff4d4d";
  const grid = css("--border") || "#1b2620";
  const muted = css("--muted") || "#8ca196";

  const data = anchor.horizon.map((h) => ({
    t: `+${h.horizon_seconds}s`,
    det: h.detection_prob ?? h.attack_prob,
    wm: h.attack_prob,
    ci: h.attack_ci as [number, number],
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={grid} strokeDasharray="2 4" />
        <XAxis
          dataKey="t"
          stroke={grid}
          tick={{ fill: muted, fontSize: 10, fontFamily: "Fira Code" }}
        />
        <YAxis
          domain={[0, 1]}
          stroke={grid}
          tick={{ fill: muted, fontSize: 10, fontFamily: "Fira Code" }}
        />
        <Tooltip
          contentStyle={{
            background: css("--surface") || "#0b120e",
            border: `1px solid ${grid}`,
            fontFamily: "Fira Code",
            fontSize: 12,
          }}
          formatter={(v: unknown, name) => {
            if (name === "ci" && Array.isArray(v))
              return [`${(v[0] * 100).toFixed(1)}–${(v[1] * 100).toFixed(1)}%`, "95% CI"];
            return [`${(Number(v) * 100).toFixed(1)}%`, name === "det" ? "detection" : "world model"];
          }}
        />
        <Area
          type="monotone"
          dataKey="ci"
          stroke="none"
          fill={info}
          fillOpacity={0.14}
          isAnimationActive={false}
          name="ci"
        />
        <Line
          type="monotone"
          dataKey="wm"
          stroke={info}
          strokeWidth={1.5}
          strokeDasharray="5 4"
          dot={{ r: 2 }}
          isAnimationActive={false}
          name="wm"
        />
        <Line
          type="monotone"
          dataKey="det"
          stroke={brand}
          strokeWidth={2}
          dot={{ r: 3 }}
          isAnimationActive={false}
          name="det"
        />
        <ReferenceLine
          y={threshold}
          stroke={danger}
          strokeDasharray="3 3"
          label={{
            value: `alert ≥ ${threshold.toFixed(2)}`,
            position: "insideTopLeft",
            fill: danger,
            fontSize: 10,
          }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
