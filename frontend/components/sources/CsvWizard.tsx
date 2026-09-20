"use client";
import * as React from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { UploadCloud } from "lucide-react";
import { Panel } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge, Input, Label } from "@/components/ui/primitives";
import { parseCSV, summarise } from "@/lib/csv";
import { matchColumns, REQUIRED, FEATURE_GROUPS } from "@/lib/schema";
import { api, pollJob } from "@/lib/api";
import { useStore, type CsvState } from "@/lib/store";
import { num } from "@/lib/format";
import { ForecastProgress, CyberSpinner } from "@/components/ui/loading";

export function CsvWizard() {
  const router = useRouter();
  const csv = useStore((s) => s.csv);
  const setCsv = useStore((s) => s.setCsv);
  const setResult = useStore((s) => s.setResult);
  const [drag, setDrag] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [familyHint, setFamilyHint] = React.useState("");
  const [explain, setExplain] = React.useState(true);
  const fileRef = React.useRef<HTMLInputElement>(null);

  const [stage, setStage] = React.useState("Ingesting and preparing flow records...");

  const handle = async (file: File) => {
    try {
      const text = await file.text();
      const { header, rows } = parseCSV(text, { maxRows: 200_000 });
      if (!header.length) return toast.error("Could not read a header row");
      const match = matchColumns(header);
      const summary = summarise(header, rows);
      const next: CsvState = { file, name: file.name, header, rows, summary, match };
      setCsv(next);
      toast[match.ok ? "success" : "warning"](
        match.ok
          ? "CSV matched the contract"
          : `${match.requiredMiss.length} required column(s) missing`
      );
    } catch (e) {
      toast.error("Parse failed: " + (e as Error).message);
    }
  };

  const run = async () => {
    if (!csv) return;
    if (!csv.file) {
      toast.error("File session expired. Please re-select the CSV file.");
      fileRef.current?.click();
      return;
    }
    setBusy(true);
    setStage("Uploading flow records to backend...");
    try {
      const stageTimer = setTimeout(() => {
        setStage("Aggregating 10s state windows and rolling forward World Model...");
      }, 1500);

      const out = await api.forecastCsv(csv.file, { familyHint, explain });
      clearTimeout(stageTimer);

      if (out.job) {
        setStage("Large dataset detected — processing asynchronous worker job...");
        toast.info(`Large file — job ${out.job_id} queued`);
        router.push("/pipeline");
        const res = await pollJob(out.job_id);
        setResult(res);
        toast.success("Job complete: predictive forecast ready");
        router.push("/dashboard");
      } else {
        setStage("Synthesizing ATT&CK tactics & saliency explanations...");
        setResult(out.result);
        toast[out.result.summary.n_alerts ? "warning" : "success"](
          `${out.result.summary.n_alerts} alert window(s) detected`,
          {
            description: `Peak attack prob: ${Math.round(
              Math.max(
                ...out.result.anchors.map(
                  (a) => a.max_detection_prob ?? a.max_attack_prob
                )
              ) * 100
            )}%`,
          }
        );
        router.push("/dashboard");
      }
    } catch (e: any) {
      toast.error(
        e?.status === 422
          ? "Schema rejected by backend: " + (e.body?.detail || "")
          : "Forecast failed: " + e.message
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div
        role="button"
        tabIndex={0}
        onClick={() => fileRef.current?.click()}
        onKeyDown={(e) =>
          (e.key === "Enter" || e.key === " ") &&
          (e.preventDefault(), fileRef.current?.click())
        }
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files[0];
          if (f) handle(f);
        }}
        className={"dropzone cursor-pointer p-8 text-center" + (drag ? " drag" : "")}
      >
        <UploadCloud className="mx-auto mb-1 text-brand" />
        <div className="text-sm">Drop a CICFlowMeter CSV here, or click to browse</div>
        <div className="mt-1 text-xs text-muted">
          parsed in-browser · nothing uploaded until you run the forecast
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])}
        />
      </div>

      {csv && (
        <>
          <Panel title="Column match">
            <div className="flex items-center gap-2 text-sm">
              <Badge tone={csv.match.ok ? "ok" : "warn"}>
                {csv.match.ok ? "READY" : "INCOMPLETE"}
              </Badge>
              <span className="text-muted">
                {csv.match.requiredHit.length}/{REQUIRED.length} required ·{" "}
                {csv.match.tier2Hit.length} Tier-2 · {csv.match.flagHit.length} flag
              </span>
            </div>
            {csv.match.requiredMiss.length > 0 && (
              <div>
                <div className="mb-1 text-xs text-danger">Missing required columns</div>
                <div className="flex flex-wrap gap-1">
                  {csv.match.requiredMiss.map((c) => (
                    <Badge key={c}>{c}</Badge>
                  ))}
                </div>
              </div>
            )}
            {csv.match.aliased.length > 0 && (
              <div>
                <div className="mb-1 text-xs text-muted">Auto-mapped aliases</div>
                <div className="flex flex-wrap gap-1">
                  {csv.match.aliased.map((a) => (
                    <Badge key={a.from}>
                      {a.from} → {a.to}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {csv.match.synthTimeAxis && (
              <p className="text-xs text-warn">
                No flow_start_epoch — a synthetic time axis will be derived from Timestamp
                order.
              </p>
            )}
            {csv.match.flagHit.length > 0 && (
              <div>
                <div className="mb-1 text-xs text-muted">TCP flag columns</div>
                <div className="flex flex-wrap gap-1">
                  {csv.match.flagHit.map((c) => (
                    <Badge key={c}>{c}</Badge>
                  ))}
                </div>
              </div>
            )}
            {csv.match.labelled && (
              <p className="text-xs text-muted">
                A Label column is present — used only for display, never as a model input.
              </p>
            )}
          </Panel>

          <Panel title="Data information">
            <div className="grid gap-3 sm:grid-cols-3">
              <div>
                <div className="text-[11px] uppercase text-muted">rows × cols</div>
                <div className="kpi text-lg">
                  {num(csv.summary.rowCount)} × {csv.summary.colCount}
                </div>
              </div>
              <div>
                <div className="text-[11px] uppercase text-muted">time span</div>
                <div className="kpi text-lg">
                  {csv.summary.spanSeconds != null
                    ? `${num(Math.round(csv.summary.spanSeconds))} s`
                    : "—"}
                </div>
                <div className="text-xs text-muted">axis: {csv.summary.timeAxis}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase text-muted">est. windows / anchors</div>
                <div className="kpi text-lg">
                  {csv.summary.estWindows != null
                    ? `${csv.summary.estWindows} / ~${Math.max(0, csv.summary.estWindows - 12)}`
                    : "—"}
                </div>
                <div className="text-xs text-muted">10 s windows, L=12</div>
              </div>
            </div>
            {csv.summary.families.length > 0 && (
              <div>
                <div className="mb-1 text-xs text-muted">Label distribution (display only)</div>
                <div className="flex flex-wrap gap-1">
                  {csv.summary.families.slice(0, 12).map(([k, n]) => (
                    <Badge key={k}>
                      {k} {num(n)}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {csv.summary.protocols.length > 0 && (
              <div>
                <div className="mb-1 text-xs text-muted">Protocols</div>
                <div className="flex flex-wrap gap-1">
                  {csv.summary.protocols.slice(0, 8).map(([k, n]) => (
                    <Badge key={k}>
                      {k} {num(n)}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </Panel>

          <Panel title="Feature groups this pipeline will build (53)">
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(FEATURE_GROUPS).map(([g, feats]) => (
                <div key={g} className="rounded border border-line p-2.5">
                  <div className="text-xs font-semibold">{g}</div>
                  <div className="mono mt-0.5 text-[11px] leading-relaxed text-muted">
                    {feats.join(", ")}
                  </div>
                </div>
              ))}
            </div>
          </Panel>

          <div className="flex flex-wrap items-center gap-3">
            <Label className="flex items-center gap-2">
              family hint
              <Input
                value={familyHint}
                onChange={(e) => setFamilyHint(e.target.value)}
                placeholder="optional, e.g. PortScan"
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
              disabled={!csv.match.ok || busy}
              onClick={run}
            >
              {busy ? (
                <>
                  <CyberSpinner size={13} /> {stage}
                </>
              ) : (
                "Run forecast →"
              )}
            </Button>
          </div>

          {busy && (
            <div className="mt-2">
              <ForecastProgress stage={stage} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
