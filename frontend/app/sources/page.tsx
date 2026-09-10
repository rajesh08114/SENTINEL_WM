"use client";
import * as React from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Panel } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Stepper } from "@/components/wizard/Stepper";
import { CsvWizard } from "@/components/sources/CsvWizard";
import { useStore } from "@/lib/store";
import { apiBase, wsBase } from "@/lib/api";

export default function SourcesPage() {
  const csv = useStore((s) => s.csv);
  return (
    <div className="flex flex-col gap-5">
      <h1 className="text-xl font-semibold">Data sources</h1>
      <Stepper
        steps={["Select source", "Match & inspect", "Forecast", "Dashboard"]}
        active={csv ? 2 : 1}
      />
      <Tabs defaultValue="csv">
        <TabsList>
          <TabsTrigger value="csv">CSV upload</TabsTrigger>
          <TabsTrigger value="pcap">PCAP</TabsTrigger>
          <TabsTrigger value="telemetry">Telemetry</TabsTrigger>
        </TabsList>
        <TabsContent value="csv" className="mt-4">
          <CsvWizard />
        </TabsContent>
        <TabsContent value="pcap" className="mt-4">
          <PcapPanel />
        </TabsContent>
        <TabsContent value="telemetry" className="mt-4">
          <TelemetryPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function PcapPanel() {
  return (
    <div className="flex flex-col gap-4">
      <Panel title="PCAP ingestion — Phase 2">
        <p className="text-sm text-muted">
          PCAP upload is scoped for Phase 2. The endpoint{" "}
          <code>POST /forecast/pcap</code> exists and deliberately returns{" "}
          <code>501 Not Implemented</code> today.
        </p>
        <p className="text-sm text-muted">
          For real packets now, use <b className="text-ink">Live capture</b> on the Live
          page — the host agent turns sniffed packets into the same flow schema this
          pipeline consumes.
        </p>
        <Button
          variant="subtle"
          className="w-max"
          onClick={() =>
            fetch(apiBase() + "/forecast/pcap", { method: "POST" })
              .then((r) => r.json().catch(() => ({})))
              .then((b) => toast.info(`Server: ${b.detail || "501 not implemented"}`))
              .catch(() => toast.error("unreachable"))
          }
        >
          Ping the endpoint
        </Button>
      </Panel>
      <Panel title="Interim workaround">
        <p className="text-sm text-muted">
          Convert offline with CICFlowMeter and upload the CSV:{" "}
          <code>cicflowmeter -f capture.pcap -c flows.csv</code>.
        </p>
      </Panel>
    </div>
  );
}

function TelemetryPanel() {
  const snippet = `import json, websocket
ws = websocket.create_connection("${wsBase()}/stream")
ws.send(json.dumps({"type": "hello", "source": "sensor-01"}))
for batch in flow_batches():                 # list[dict], CICFlowMeter columns
    ws.send(json.dumps({"type": "flows", "records": batch}))
    print(ws.recv())                          # {"type":"ack"|"forecast", ...}
ws.send(json.dumps({"type": "close"}))`;
  return (
    <div className="flex flex-col gap-4">
      <Panel title="Raw telemetry WebSocket">
        <p className="text-sm text-muted">
          <code>{wsBase()}/stream</code> — the server buffers incoming flows into 10 s
          windows and pushes a forecast each time a window closes with ≥ 12 windows of
          history.
        </p>
        <pre className="mono overflow-x-auto text-xs leading-relaxed">{snippet}</pre>
        <Button
          variant="subtle"
          className="w-max"
          onClick={() => navigator.clipboard.writeText(snippet).then(() => toast.success("Copied"))}
        >
          Copy
        </Button>
      </Panel>
      <Panel title="Prefer the managed path?">
        <p className="text-sm text-muted">
          The{" "}
          <Link href="/live" className="text-brand underline decoration-dotted">
            Live Telemetry
          </Link>{" "}
          page runs a synthetic test-bed or a live-capture session for you, with a rolling
          P(attack) timeline and streaming anchor cards.
        </p>
      </Panel>
    </div>
  );
}
