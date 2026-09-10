import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Fira Sans"', "system-ui", "sans-serif"],
        mono: ['"Fira Code"', "ui-monospace", "monospace"],
      },
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        elevated: "var(--elevated)",
        line: "var(--border)",
        ink: "var(--fg)",
        muted: "var(--muted)",
        brand: "var(--brand)",
        "brand-dim": "var(--brand-dim)",
        danger: "var(--danger)",
        warn: "var(--warn)",
        info: "var(--info)",
        ok: "var(--ok)",
      },
      borderColor: { DEFAULT: "var(--border)" },
      keyframes: {
        pulse2: { "70%": { boxShadow: "0 0 0 8px transparent" }, "100%": { boxShadow: "0 0 0 0 transparent" } },
      },
      animation: { pulse2: "pulse2 2s infinite" },
    },
  },
  plugins: [],
};
export default config;
