import { Panel } from "@/components/ui/card";
import { SystemStatus } from "@/components/common/SystemStatus";

export default function OverviewPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="grid-bg flex flex-col gap-3 rounded-lg border border-line bg-surface p-6 sm:p-8">
        <span className="w-max rounded-full border border-line px-2 py-[2px] font-mono text-[11px] text-muted">
          SOC ANALYST CONSOLE
        </span>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Forecast where a network intrusion is going —{" "}
          <span className="text-brand">6 windows ahead</span>.
        </h1>
        <p className="max-w-3xl text-sm leading-relaxed text-muted">
          SENTINEL-WM is a world model for network attacker progression. It ingests flow
          telemetry (CIC-IDS-2017 schema), compresses each 10-second window into a
          53-dimensional state, and rolls the learned state-transition dynamics forward to
          predict attack probability, kill-chain progression, and MITRE ATT&amp;CK phase
          for each of the next six windows — with calibrated confidence intervals and
          per-feature explanations.
        </p>
      </header>

      <SystemStatus />

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="The problem">
          <ul className="flex list-disc flex-col gap-2 pl-4 text-sm text-muted">
            <li>
              <b className="text-ink">Detection is reactive.</b> Signature and anomaly IDS
              fire once an attack is already underway — analysts inherit an incident, not a
              warning.
            </li>
            <li>
              <b className="text-ink">Alerts have no trajectory.</b> A SYN-flood spike tells
              you what is happening now, not whether it precedes lateral movement or
              exfiltration.
            </li>
            <li>
              <b className="text-ink">Context is manual.</b> Mapping raw flow features to a
              kill-chain phase and ATT&amp;CK tactic is analyst tribal knowledge.
            </li>
          </ul>
        </Panel>
        <Panel title="The solution">
          <ul className="flex list-disc flex-col gap-2 pl-4 text-sm text-muted">
            <li>
              <b className="text-ink">Probabilistic world model.</b> A Bi-GRU encoder +
              State-Transition Network learns p(state₁ | state, history) and samples K steps
              of rollout.
            </li>
            <li>
              <b className="text-ink">Lead time, quantified.</b> Each anchor reports
              first-alert horizon and lead-time-to-attack in seconds.
            </li>
            <li>
              <b className="text-ink">ATT&amp;CK phase mapping.</b> Every horizon step
              carries a tactic, technique IDs, kill-chain phase, dominant family and a
              rationale.
            </li>
            <li>
              <b className="text-ink">Explainable.</b> Gradient×input and attention saliency
              expose the driving features and history windows.
            </li>
          </ul>
        </Panel>
      </div>

      <Panel title="How the pieces connect">
        <pre className="mono overflow-x-auto text-xs leading-relaxed text-muted sm:text-[13px]">
{`  flow CSV / synthetic test-bed / live capture agent
            │
            ▼
  clean_flow_frame        winsorize on train-day stats, canonicalize columns
            │
            ▼
  build_state_windows     10s aggregation → 53 features/window, progression labels
            │
            ▼
  windows_to_tensors      slide L=12, scale with persisted state_scaler.pkl
            │
            ▼
  simulate_anchor         world-model rollout × MC + system blend + ATT&CK + explain
            │
            ▼
  ForecastResponse        anchors[].horizon[6].{attack_prob, ci, progression, attck}`}
        </pre>
      </Panel>

      <div className="grid gap-6 sm:grid-cols-3">
        <Panel title="Inputs">
          <ul className="flex flex-col gap-1.5 text-sm text-muted">
            <li>CSV of CICFlowMeter flows — matched client-side, re-validated by the API</li>
            <li>Synthetic test-bed — generated scenarios through the real pipeline</li>
            <li>Live capture — a host agent sniffs a chosen interface</li>
            <li>PCAP upload — Phase 2 (endpoint returns 501)</li>
          </ul>
        </Panel>
        <Panel title="Outputs">
          <ul className="flex flex-col gap-1.5 text-sm text-muted">
            <li>Per-horizon P(attack) with 95% CI and alert flag</li>
            <li>Progression state ribbon (NORMAL → CONTINUATION)</li>
            <li>MITRE ATT&amp;CK tactic / technique / kill-chain phase per step</li>
            <li>Driving features + temporal saliency</li>
          </ul>
        </Panel>
        <Panel title="Guarantees">
          <ul className="flex flex-col gap-1.5 text-sm text-muted">
            <li>Leakage-safe split: whole-episode assignment + boundary purge</li>
            <li>Best-F1 threshold calibrated on validation within an FPR budget</li>
            <li>Portable model bundle — backend depends on no research code</li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}
