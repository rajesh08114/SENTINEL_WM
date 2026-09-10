"use client";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const css = (v: string) =>
  typeof window === "undefined"
    ? ""
    : getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export function Saliency({
  gradient,
  attention,
}: {
  gradient?: number[];
  attention?: number[];
}) {
  const brand = css("--brand") || "#2ee66b";
  const info = css("--info") || "#5aa9ff";
  const grid = css("--border") || "#1b2620";
  const muted = css("--muted") || "#8ca196";
  const n = (gradient || attention || []).length;
  const data = Array.from({ length: n }, (_, i) => ({
    t: `t-${n - 1 - i}`,
    grad: gradient?.[i],
    attn: attention?.[i],
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={grid} strokeDasharray="2 4" />
        <XAxis dataKey="t" stroke={grid} tick={{ fill: muted, fontSize: 10 }} />
        <YAxis stroke={grid} tick={{ fill: muted, fontSize: 10 }} />
        <Tooltip
          contentStyle={{
            background: css("--surface") || "#0b120e",
            border: `1px solid ${grid}`,
            fontFamily: "Fira Code",
            fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: muted }} />
        {gradient && (
          <Bar dataKey="grad" name="gradient·input" fill={brand} isAnimationActive={false} />
        )}
        {attention && (
          <Bar dataKey="attn" name="attention" fill={info} isAnimationActive={false} />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}
