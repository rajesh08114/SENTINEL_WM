"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Panel } from "@/components/ui/card";
import { KpiTile } from "@/components/panels/KpiTile";
import { round } from "@/lib/format";

const LAYERS: [string, string, string, string][] = [
  ["Input", "x ∈ ℝ^{L×F}, Δt ∈ ℝ^{L}", "L history windows, F=53 scaled state features + elapsed-time channel", "—"],
  ["Elapsed-time positional encoding", "sinusoidal(Δt) → d_model", "real seconds between windows, not just index", "—"],
  ["Bi-GRU encoder", "2 × GRU(d_model=160), bidirectional", "sequence context in both directions", "232,320"],
  ["Additive attention pool", "score → softmax → weighted sum", "collapses L steps to one context vector; the attention-saliency source", "103,040"],
  ["State-Transition Network (STN)", "MLP → (μ, logσ) ; z = μ + σ·ε", "probabilistic next-state; reparameterized sampling drives MC rollout", "103,680"],
  ["Attack head", "Linear → sigmoid", "P(attack) for the next window", "part of 83,562"],
  ["Progression head", "Linear → softmax(5)", "NORMAL / PRE_ATTACK / ONSET / ACTIVE / CONTINUATION", "part of 83,562"],
  ["Next-state head", "Linear → ℝ^F", "reconstructs the next state vector (self-supervised anchor)", "part of 83,562"],
];

const MEMBERS: [string, string, string][] = [
  ["SENTINEL-WM world model", "532,042", "Bi-GRU + STN + heads. Provides progression, rollout, ATT&CK, CI, explanations."],
  ["TCN", "172,644", "dilated causal conv stack — decorrelated P(attack) member"],
  ["LSTM", "247,716", "recurrent P(attack) member"],
  ["GRU", "191,140", "recurrent P(attack) member"],
];

export default function ModelPage() {
  const { data: m } = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const L = m?.history_windows ?? 12;
  const K = m?.horizon_steps ?? 6;
  const F = m?.feature_dim ?? 53;
  const W = m?.window_seconds ?? 10;

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-xl font-semibold">SENTINEL-WM (system) — model card</h1>
      <p className="max-w-3xl text-sm text-muted">
        The world model blended (validation-tuned scalar weight) with three decorrelated
        members. The blend adds P(attack) sharpness only; progression, rollout and ATT&amp;CK
        mapping always come from the world model. Full derivation in{" "}
        <code>docs/system_architecture.md</code>.
      </p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <KpiTile label="History L" value={`${L} windows`} sub={`${L * W} s context`} />
        <KpiTile label="Horizon K" value={`${K} windows`} sub={`${K * W} s lookahead`} />
        <KpiTile label="State dim F" value={`${F}`} sub="features per 10 s window" />
        <KpiTile
          label="Parameters"
          value="1,143,542"
          sub={m?.blend_weight != null ? `blend weight ${round(m.blend_weight, 3)}` : "world model + 3 members"}
        />
      </div>

      <Panel title="Component A — world model, layer by layer">
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Shape / op</th>
                <th>Role</th>
                <th>Params</th>
              </tr>
            </thead>
            <tbody>
              {LAYERS.map((r) => (
                <tr key={r[0]}>
                  <td className="font-medium">{r[0]}</td>
                  <td className="mono text-xs">{r[1]}</td>
                  <td className="text-xs text-muted">{r[2]}</td>
                  <td className="mono text-xs">{r[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Forward pass">
          <ol className="flex list-decimal flex-col gap-1.5 pl-5 text-sm text-muted">
            <li>Embed each window + add elapsed-time positional encoding.</li>
            <li>Bi-GRU over the L steps → hidden sequence.</li>
            <li>Additive attention → context vector c.</li>
            <li>STN(c) → (μ, logσ); sample z.</li>
            <li>Heads on z → P(attack), progression dist, next-state.</li>
            <li>Rollout: feed next-state back as the newest window, repeat K times; MC over ε for the CI ribbon.</li>
          </ol>
        </Panel>
        <Panel title="Backward pass">
          <ul className="flex list-disc flex-col gap-1.5 pl-4 text-sm text-muted">
            <li>joint_loss = 3.0·BCE(attack) + 0.5·CE(progression) + 0.10·MSE(next-state) + 1e-5·KL(q‖𝒩(0,1))</li>
            <li>Two-stage: SSL next-state pretrain, then joint fine-tune.</li>
            <li>Weight snapshots averaged for the served checkpoint.</li>
            <li>Members trained independently; the blend weight is <b className="text-ink">selected on validation, not learned</b>.</li>
          </ul>
        </Panel>
      </div>

      <Panel title="Composition">
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Member</th>
                <th>Params</th>
                <th>Contribution</th>
              </tr>
            </thead>
            <tbody>
              {MEMBERS.map((r) => (
                <tr key={r[0]}>
                  <td className="font-medium">{r[0]}</td>
                  <td className="mono text-xs">{r[1]}</td>
                  <td className="text-xs text-muted">{r[2]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Benchmark & trust caveats">
        <ul className="flex list-disc flex-col gap-1.5 pl-4 text-sm text-muted">
          <li>
            Leakage-safe <code>stratified</code> split: benign backbone contiguous per day,
            whole attack episodes rotated per family, boundary-crossing sequences purged.
          </li>
          <li>
            Ranks trust: PR-AUC / F1* / AUROC. The top cluster (system, LSTM, GRU,
            persistence) sits within noise of a ~76-positive test set.
          </li>
          <li>The benchmark measures now-casting, not onset forecasting; four rare families are excluded.</li>
          <li>
            Threshold is best-F1 on validation within an FPR budget — the FPR-only rule
            collapsed to all-positive on the prevalence-shifted test set.
          </li>
          <li>
            <b className="text-ink">Synthetic-traffic caveat:</b> the model runs hot on
            out-of-distribution generated flows; absolute P(attack) on a test-bed session is
            not calibrated — read the relative ramp, the progression sequence, and the
            ATT&amp;CK mapping.
          </li>
        </ul>
      </Panel>
    </div>
  );
}
