"use client";
import * as React from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { FileUp } from "lucide-react";
import { Panel } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import { num } from "@/lib/format";

export function PcapUpload() {
  const router = useRouter();
  const setResult = useStore((s) => s.setResult);
  const setCsv = useStore((s) => s.setCsv);
  const [file, setFile] = React.useState<File | null>(null);
  const [drag, setDrag] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [familyHint, setFamilyHint] = React.useState("");
  const [explain, setExplain] = React.useState(true);
  const [bpfFilter, setBpfFilter] = React.useState("");
  const ref = React.useRef<HTMLInputElement>(null);

  const BPF_PRESETS = [
    { label: "All Traffic", bpf: "" },
    { label: "HTTP/HTTPS", bpf: "tcp port 80 or tcp port 443" },
    { label: "Exclude Broadcast/Multicast", bpf: "not broadcast and not multicast" },
    { label: "DNS", bpf: "port 53" },
    { label: "Admin (SSH/RDP)", bpf: "tcp port 22 or tcp port 3389" },
  ];

  const run = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const result = await api.forecastPcap(file, {
        familyHint: familyHint.trim() || undefined,
        explain,
        bpfFilter: bpfFilter.trim() || undefined,
      });
      setCsv(null);
      setResult(result);
      toast[result.summary.n_alerts ? "warning" : "success"](
        `${result.meta.n_flows} flows reassembled · ${result.summary.n_alerts} alert window(s)`
      );
      router.push("/dashboard");
    } catch (e: any) {
      toast.error(
        e?.status === 422
          ? "Could not read the capture: " + (e.body?.detail || e.message)
          : e?.status === 413
            ? e.body?.detail || "pcap too large"
            : "Forecast failed: " + e.message
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Wireshark / PCAP Capture Ingestion">
        <p className="text-sm text-muted">
          Offline packet ingestion reads standard Wireshark (<code>.pcap</code>, <code>.pcapng</code>)
          traces, reassembles bidirectional TCP/UDP flows, constructs 10-second state windows, and rolls the
          World Model forward $K=6$ steps.
        </p>
        <div
          role="button"
          tabIndex={0}
          onClick={() => ref.current?.click()}
          onKeyDown={(e) =>
            (e.key === "Enter" || e.key === " ") && (e.preventDefault(), ref.current?.click())
          }
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
          }}
          className={"dropzone cursor-pointer p-8 text-center" + (drag ? " drag" : "")}
        >
          <FileUp className="mx-auto mb-1 text-brand" />
          <div className="text-sm font-medium">
            {file ? file.name : "Drop a Wireshark capture (.pcap / .pcapng) here, or click to browse"}
          </div>
          <div className="mt-1 text-xs text-muted">
            {file
              ? `${num(Math.round(file.size / 1024))} KB · ready for packet reassembly`
              : "≤ 60 MB · for large captures, slice with tshark / editcap or use BPF filter"}
          </div>
          <input
            ref={ref}
            type="file"
            accept=".pcap,.pcapng,.cap,application/vnd.tcpdump.pcap"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
          />
        </div>

        {/* BPF Filter Section */}
        <div className="flex flex-col gap-2 rounded-md border border-line/60 bg-elevated/40 p-3">
          <div className="flex items-center justify-between">
            <Label className="text-xs font-semibold uppercase tracking-wider text-muted">
              Wireshark BPF Capture Filter (Optional)
            </Label>
            <span className="text-[11px] text-muted">Berkeley Packet Filter syntax</span>
          </div>
          <Input
            value={bpfFilter}
            onChange={(e) => setBpfFilter(e.target.value)}
            placeholder="e.g. tcp port 80 or ip host 192.168.1.10"
            className="font-mono text-xs"
          />
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="text-[11px] text-muted mr-1">Presets:</span>
            {BPF_PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setBpfFilter(p.bpf)}
                className={`rounded border px-2 py-0.5 text-[11px] transition-colors ${
                  bpfFilter === p.bpf
                    ? "border-brand bg-brand/10 text-brand"
                    : "border-line bg-surface hover:bg-elevated text-muted hover:text-ink"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Label className="flex items-center gap-2">
            family hint
            <Input
              value={familyHint}
              onChange={(e) => setFamilyHint(e.target.value)}
              placeholder="optional, e.g. DoS Hulk"
              className="w-40"
            />
          </Label>
          <label className="flex items-center gap-2 text-xs text-muted">
            <input
              type="checkbox"
              checked={explain}
              onChange={(e) => setExplain(e.target.checked)}
            />
            include explanations
          </label>
          <Button
            className="ml-auto"
            variant="outline"
            disabled={!file || busy}
            onClick={run}
          >
            {busy ? "Parsing packets…" : "Run forecast →"}
          </Button>
        </div>
      </Panel>

      <Panel title="Wireshark Export & Preprocessing Guidelines">
        <div className="flex flex-col gap-2 text-xs text-muted leading-relaxed">
          <p>
            <strong className="text-ink">1. Exporting from Wireshark:</strong> In Wireshark, use <code>File → Save As…</code> and select
            <code> Wireshark/tcpdump/… pcap (*.pcap)</code> or <code>pcapng (*.pcapng)</code>.
          </p>
          <p>
            <strong className="text-ink">2. Large Traces (&gt; 60 MB):</strong> Slice into targeted time intervals using standard Wireshark utilities:
            <br />
            <code className="bg-elevated px-1 py-0.5 rounded text-ink">
              editcap -A &quot;2024-05-10 09:00:00&quot; -B &quot;2024-05-10 09:20:00&quot; full.pcap sliced.pcap
            </code>
          </p>
          <p>
            <strong className="text-ink">3. Extracting Flows Directly:</strong> You can also convert PCAPs to flow CSVs with CICFlowMeter:
            <br />
            <code className="bg-elevated px-1 py-0.5 rounded text-ink">
              cicflowmeter -f capture.pcap -c flows.csv
            </code>
          </p>
        </div>
      </Panel>
    </div>
  );
}

