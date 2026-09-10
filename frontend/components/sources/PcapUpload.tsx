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
  const ref = React.useRef<HTMLInputElement>(null);

  const run = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const result = await api.forecastPcap(file, { familyHint, explain });
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
      <Panel title="PCAP / PCAPNG capture">
        <p className="text-sm text-muted">
          The server reads the capture offline (no live-capture privileges), reassembles
          bidirectional TCP/UDP flows, and runs the same pipeline as a flow CSV. Other
          packet types are ignored; a capture with fewer than ~13 ten-second windows
          produces no anchors.
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
          <div className="text-sm">
            {file ? file.name : "Drop a .pcap / .pcapng here, or click to browse"}
          </div>
          <div className="mt-1 text-xs text-muted">
            {file
              ? `${num(Math.round(file.size / 1024))} KB · uploaded to the backend on run`
              : "≤ 60 MB · slice larger captures with editcap / tshark -c"}
          </div>
          <input
            ref={ref}
            type="file"
            accept=".pcap,.pcapng,.cap,application/vnd.tcpdump.pcap"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
          />
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
            {busy ? "parsing…" : "Run forecast →"}
          </Button>
        </div>
      </Panel>

      <Panel title="Converting a large capture instead">
        <p className="text-sm text-muted">
          For very large or filtered captures, extract flows first and use the CSV tab:{" "}
          <code>cicflowmeter -f capture.pcap -c flows.csv</code>, or slice with{" "}
          <code>editcap -A &quot;2024-01-01 10:00:00&quot; -B &quot;2024-01-01 10:20:00&quot; in.pcap out.pcap</code>.
        </p>
      </Panel>
    </div>
  );
}
