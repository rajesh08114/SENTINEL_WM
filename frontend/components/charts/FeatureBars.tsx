"use client";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { DrivingFeatures } from "@/lib/types";

const css = (v: string) =>
  typeof window === "undefined"
    ? ""
    : getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export function FeatureBars({ df }: { df: DrivingFeatures }) {
  const brand = css("--brand") || "#2ee66b";
  const danger = css("--danger") || "#ff4d4d";
  const grid = css("--border") || "#1b2620";
  const muted = css("--muted") || "#8ca196";
  const data = (df.top_features || []).slice(0, 12).map((t) => ({
    feature: t.feature,
    v: t.importance * (t.signed || 1),
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 12, bottom: 0, left: 8 }}
      >
        <CartesianGrid stroke={grid} strokeDasharray="2 4" horizontal={false} />
        <XAxis type="number" stroke={grid} tick={{ fill: muted, fontSize: 10 }} />
        <YAxis
          type="category"
          dataKey="feature"
          width={140}
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
          formatter={(v: unknown) => [Number(v).toFixed(4), "signed contribution"]}
        />
        <Bar dataKey="v" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.v >= 0 ? danger : brand} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
