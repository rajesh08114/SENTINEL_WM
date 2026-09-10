export const pct = (x: number | null | undefined, d = 1) =>
  x == null || Number.isNaN(x) ? "—" : (100 * x).toFixed(d) + "%";

// deterministic thousands separator — Intl.NumberFormat() picks up the runtime
// locale, which differs between the static-export build (Node) and the browser
// and causes React hydration mismatches.
export const thousands = (x: number | null | undefined) =>
  x == null ? "—" : String(Math.trunc(x)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

export const num = thousands;

export const sec = (x: number | null | undefined) => (x == null ? "—" : `${x} s`);

export const round = (x: number | null | undefined, d = 3) =>
  x == null ? "—" : Number(x).toFixed(d);

export const ago = (epoch: number) => {
  const s = Math.max(0, Math.round(Date.now() / 1000 - epoch));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
};
