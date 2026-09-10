"use client";
import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiBase, setApiBase } from "@/lib/api";

export function ApiBaseField() {
  const qc = useQueryClient();
  const [val, setVal] = React.useState("");
  React.useEffect(() => setVal(apiBase()), []);
  const commit = () => {
    const v = val.trim();
    if (v) {
      setApiBase(v);
      qc.invalidateQueries();
    }
  };
  return (
    <label className="flex items-center gap-2 text-xs text-muted">
      API
      <input
        value={val}
        onChange={(e) => setVal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && commit()}
        spellCheck={false}
        aria-label="Backend base URL"
        className="w-[210px] rounded border border-line bg-elevated px-2 py-1 font-mono text-xs text-ink focus-visible:outline-none focus-visible:border-brand"
      />
    </label>
  );
}
