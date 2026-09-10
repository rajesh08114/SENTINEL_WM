"use client";
import * as React from "react";
import { Moon, Sun } from "lucide-react";

export function ThemeToggle() {
  const [theme, setTheme] = React.useState<"dark" | "light">("dark");
  React.useEffect(() => {
    let t: "dark" | "light" = "dark";
    try {
      const s = localStorage.getItem("sentinel.theme");
      if (s === "light" || s === "dark") t = s;
    } catch {
      /* ignore */
    }
    setTheme(t);
    document.documentElement.setAttribute("data-theme", t);
  }, []);
  const flip = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("sentinel.theme", next);
    } catch {
      /* ignore */
    }
  };
  return (
    <button
      onClick={flip}
      aria-label="Toggle colour theme"
      className="rounded border border-line p-1.5 text-muted hover:text-ink hover:border-brand"
    >
      {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
    </button>
  );
}
