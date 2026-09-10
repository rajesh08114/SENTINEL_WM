"use client";
import * as React from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Panel } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Stepper } from "@/components/wizard/Stepper";
import { CsvWizard } from "@/components/sources/CsvWizard";
import { PcapUpload } from "@/components/sources/PcapUpload";
import { useStore } from "@/lib/store";
import { wsBase } from "@/lib/api";

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
          <PcapUpload />
        </TabsContent>
        <TabsContent value="telemetry" className="mt-4">
          <TelemetryPanel />
        </TabsContent>
      </Tabs>
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
