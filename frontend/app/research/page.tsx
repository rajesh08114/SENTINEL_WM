"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/common/PageHeader";
import { Panel } from "@/components/ui/card";
import { KpiTile } from "@/components/panels/KpiTile";
import { Scoreboard, HorizonF1Table } from "@/components/research/Scoreboard";
import { thousands } from "@/lib/format";
import {
  LEAKAGE_CONTROLS, PARAM_BUDGET, SPLIT_NOTE, TRUST_POINTS,
} from "@/lib/research-data";

export default function ResearchPage() {
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.models });
  const m = (models?.metrics as Record<string, number> | undefined) ?? undefined;

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Research"
        title="Why SENTINEL-WM — the world model, the blend, and the benchmark"
        lead="SENTINEL-WM forecasts network-attacker progression 6 windows (60 s) ahead from
          CIC-IDS-2017 flow telemetry. This page is the case for the design: what problem
          it solves, why a probabilistic world model rather than a classifier, why the
          system blend, and how it scores on a leakage-safe split — with the caveats."
      />

      {/* headline numbers (live from the served bundle) */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <KpiTile label="PR-AUC (system)" value={m ? m.pr_auc.toFixed(3) : "0.992"} sub="threshold-free · ranked #1" tone="ok" />
        <KpiTile label="F1* (ceiling)" value={m ? m.f1_best.toFixed(3) : "0.968"} sub="best threshold on test" />
        <KpiTile label="F1 (val-tuned)" value={m ? m.f1.toFixed(3) : "0.920"} sub="operating point" />
        <KpiTile label="AUROC" value={m ? m.auroc.toFixed(3) : "1.000"} sub="ranking quality" />
      </div>

      {/* 1. problem */}
      <Panel title="1 · The problem">
        <div className="grid gap-6 lg:grid-cols-3">
          <Problem
            h="Detection is reactive"
            b="Signature and anomaly IDS fire once an attack is already underway. The analyst inherits an incident, not a warning — there is no time to act before impact."
          />
          <Problem
            h="Alerts have no trajectory"
            b="A SYN-flood spike tells you what is happening now, not whether it precedes lateral movement, credential access, or exfiltration. Triage is guesswork."
          />
          <Problem
            h="Context is manual"
            b="Turning raw flow features into a kill-chain phase and a MITRE ATT&CK tactic is analyst tribal knowledge, re-derived under pressure every time."
          />
        </div>
      </Panel>

      {/* 2. why a world model */}
      <Panel title="2 · Why a world model, not a classifier">
        <div className="flex flex-col gap-3 text-sm text-muted">
          <p>
            A classifier answers <em>“is this window malicious?”</em>. A world model answers
            <em> “given the last 12 windows of network state, what does the next 6 look
            like?”</em> — and attack probability, kill-chain stage and ATT&CK phase fall out
            of that rolled-forward state.
          </p>
          <ul className="flex list-disc flex-col gap-2 pl-5">
            <li>
              <b className="text-ink">Probabilistic state transition.</b> A Bi-GRU encoder
              compresses the history; a State-Transition Network emits{" "}
              <code>(μ, logσ)</code> for the next 53-dim state and samples it
              (reparameterised). Uncertainty is first-class.
            </li>
            <li>
              <b className="text-ink">K-step rollout with Monte-Carlo.</b> Feed the predicted
              state back as the newest window and repeat 6×; sampling ε many times gives the
              95% confidence band you see on every forecast chart.
            </li>
            <li>
              <b className="text-ink">One model, three outputs.</b> Heads on the sampled
              state give P(attack), a 5-way progression distribution
              (NORMAL→CONTINUATION), and a next-state reconstruction that acts as a
              self-supervised anchor during training.
            </li>
            <li>
              <b className="text-ink">Explainable by construction.</b> Gradient×input and the
              encoder’s attention weights expose which features and which history windows
              drove the forecast.
            </li>
          </ul>
        </div>
      </Panel>

      {/* 3. why the blend */}
      <Panel title="3 · Why the system blend">
        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <div className="flex flex-col gap-3 text-sm text-muted">
            <p>
              The world model is strong at <b className="text-ink">progression, rollout and
              ATT&CK</b> but its raw P(attack) is a little soft (F1 0.83). Three small,
              architecturally-decorrelated members — a dilated <b className="text-ink">TCN</b>,
              an <b className="text-ink">LSTM</b> and a <b className="text-ink">GRU</b> — each
              predict P(attack) only. A validation-tuned scalar blend
              (<code>w · WM + (1−w) · mean(members)</code>, w ≈ 0.5) sharpens the detection
              probability without touching the world model’s narrative.
            </p>
            <p>
              Result: F1 <b className="text-ink">0.83 → 0.92</b>, F1* 0.91 → 0.97, and the
              system moves to <b className="text-brand">#1 by PR-AUC</b>. The blend weight is
              <em> selected</em> on validation, never learned end-to-end.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2 self-start">
            <KpiTile label="WM-only F1" value="0.828" tone="muted" />
            <KpiTile label="System F1" value="0.920" tone="ok" />
            <KpiTile label="WM-only F1*" value="0.914" tone="muted" />
            <KpiTile label="System F1*" value="0.968" tone="ok" />
          </div>
        </div>
      </Panel>

      {/* 4. leakage controls */}
      <Panel title="4 · Data & leakage controls">
        <p className="text-sm text-muted">
          Every number below is on the leakage-safe <code>stratified</code> split. The
          controls that make it honest:
        </p>
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th className="w-56">Control</th>
                <th>What it does</th>
              </tr>
            </thead>
            <tbody>
              {LEAKAGE_CONTROLS.map((c) => (
                <tr key={c.control}>
                  <td className="text-xs font-medium">{c.control}</td>
                  <td className="text-xs text-muted">{c.what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* 5. benchmark */}
      <Panel title="5 · Benchmark — leakage-safe split, ranked by PR-AUC">
        <p className="text-xs text-muted">{SPLIT_NOTE}</p>
        <Scoreboard />
        <div className="mt-4">
          <div className="mb-1 text-xs font-semibold text-muted">
            Forecast-horizon F1 — does it hold up as the horizon grows?
          </div>
          <HorizonF1Table />
        </div>
      </Panel>

      {/* 6. trust */}
      <Panel title="6 · How far to trust these numbers">
        <ul className="flex list-disc flex-col gap-2 pl-5 text-sm text-muted">
          {TRUST_POINTS.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </Panel>

      {/* 7. param budget */}
      <Panel title="7 · Parameter budget">
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Component</th>
                <th>Params</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {PARAM_BUDGET.map((p) => (
                <tr
                  key={p.part}
                  className={
                    p.part.includes("subtotal") || p.part.includes("total")
                      ? "font-semibold text-ink"
                      : ""
                  }
                >
                  <td className="text-xs">{p.part}</td>
                  <td className="mono text-xs">{thousands(p.params)}</td>
                  <td className="text-xs text-muted">{p.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-muted">
          Full derivation — every layer, forward and backward pass, loss weights — in{" "}
          <code>docs/system_architecture.md</code>; loss / rollout / ATT&CK / leakage detail
          in <code>docs/technical_reference.md</code>.
        </p>
      </Panel>
    </div>
  );
}

function Problem({ h, b }: { h: string; b: string }) {
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-line bg-elevated p-4">
      <div className="text-sm font-semibold text-ink">{h}</div>
      <p className="text-xs leading-relaxed text-muted">{b}</p>
    </div>
  );
}
