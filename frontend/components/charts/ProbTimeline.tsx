"use client";
import {
  Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

const css = (v: string) =>
  typeof window === "undefined"
    ? ""
    : getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export function ProbTimeline({
  points,
  threshold,
  height = 220,
}: {
  points: { i: number | string; p: number }[];
  threshold: number;
  height?: number;
}) {
  const brand = css("--brand") || "#2ee66b";
  const danger = css("--danger") || "#ff4d4d";
  const grid = css("--border") || "#1b2620";
  const muted = css("--muted") || "#8ca196";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={grid} strokeDasharray="2 4" />
        <XAxis dataKey="i" stroke={grid} tick={{ fill: muted, fontSize: 10 }} />
        <YAxis domain={[0, 1]} stroke={grid} tick={{ fill: muted, fontSize: 10 }} />
        <Tooltip
          contentStyle={{
            background: css("--surface") || "#0b120e",
            border: `1px solid ${grid}`,
            fontFamily: "Fira Code",
            fontSize: 12,
          }}
          formatter={(v: unknown) => [`${(Number(v) * 100).toFixed(1)}%`, "max P(attack)"]}
        />
        <Area
          type="monotone"
          dataKey="p"
          stroke={brand}
          fill={brand}
          fillOpacity={0.12}
          strokeWidth={2}
          isAnimationActive={false}
          dot={false}
        />
        <ReferenceLine y={threshold} stroke={danger} strokeDasharray="3 3" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
