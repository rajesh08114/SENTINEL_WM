"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { Input, Label } from "@/components/ui/primitives";
import { Button } from "@/components/ui/button";
import type { NetInterface } from "@/lib/types";

export function InterfacePicker({
  iface,
  bpf,
  onIface,
  onBpf,
}: {
  iface: string;
  bpf: string;
  onIface: (v: string) => void;
  onBpf: (v: string) => void;
}) {
  const q = useQuery({
    queryKey: ["agent"],
    queryFn: api.agentStatus,
    refetchInterval: 8000,
  });
  const agent = q.data?.agents?.[0];
  const ifaces = (agent?.interfaces || []) as unknown as NetInterface[];

  if (!q.data?.connected) {
    return (
      <div className="rounded-lg border border-warn/40 bg-warn/10 p-4 text-sm text-warn">
        <p className="font-semibold">No capture agent connected.</p>
        <p className="mt-1 text-muted">
          Run it on the host you want to monitor (needs Npcap + Administrator):
        </p>
        <pre className="mono mt-2 overflow-x-auto rounded bg-elevated p-2 text-xs text-ink">
cd capture-agent{"\n"}pip install -e .{"\n"}sentinel-capture run --backend ws://localhost:8000
        </pre>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted">
          agent <b className="text-ink">{agent?.name}</b> · {ifaces.length} interfaces
        </span>
        <Button size="sm" variant="ghost" onClick={() => q.refetch()}>
          <RefreshCw size={12} /> refresh
        </Button>
      </div>
      <div className="max-h-64 overflow-y-auto rounded border border-line">
        {ifaces.map((i) => (
          <label
            key={i.name}
            className="flex cursor-pointer items-center gap-3 border-b border-line px-3 py-2 text-sm last:border-b-0 hover:bg-elevated"
          >
            <input
              type="radio"
              name="iface"
              checked={iface === i.name}
              onChange={() => onIface(i.name)}
            />
            <span className="flex-1">
              <span className="text-ink">{i.name}</span>
              <span className="ml-2 text-xs text-muted">{i.description}</span>
            </span>
            <span className="mono text-xs text-muted">{i.ipv4 || "—"}</span>
            <span
              className={
                "mono text-[10px] " + (i.is_up ? "text-ok" : "text-muted")
              }
            >
              {i.is_up ? "up" : "down"}
            </span>
          </label>
        ))}
      </div>
      <div className="flex flex-col gap-2 rounded-md border border-line/60 bg-elevated/40 p-3">
        <div className="flex items-center justify-between">
          <Label className="text-xs font-semibold uppercase tracking-wider text-muted">
            Wireshark / BPF Capture Filter (Optional)
          </Label>
          <span className="text-[11px] text-muted">applied directly on NIC driver</span>
        </div>
        <Input
          value={bpf}
          onChange={(e) => onBpf(e.target.value)}
          placeholder="e.g. not broadcast and not multicast, port 80, host 192.168.1.10"
          className="font-mono text-xs"
        />
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          <span className="text-[11px] text-muted mr-1">Presets:</span>
          {[
            { label: "All Traffic", val: "" },
            { label: "Exclude Noise", val: "not broadcast and not multicast" },
            { label: "HTTP/HTTPS", val: "tcp port 80 or tcp port 443" },
            { label: "DNS", val: "port 53" },
            { label: "Admin Ports", val: "tcp port 22 or tcp port 3389" },
          ].map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => onBpf(p.val)}
              className={`rounded border px-2 py-0.5 text-[11px] transition-colors ${
                bpf === p.val
                  ? "border-brand bg-brand/10 text-brand"
                  : "border-line bg-surface hover:bg-elevated text-muted hover:text-ink"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

