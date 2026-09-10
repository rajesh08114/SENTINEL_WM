# SENTINEL-WM — Plan Validation & Implementation Notes

> **Status note (kept for history).** This document records the original plan
> review. Several things have since moved on — the split is now the leakage-safe
> `stratified` (not `block`), the encoder is a Bi-GRU (not the Temporal
> Transformer), `F=53`, `L=12`, and there is a dual benchmark + flow-level
> augmentation + per-attack-family metrics. The current source of truth is
> [`technical_reference.md`](technical_reference.md) (esp. Part 1.9 — leakage
> controls) and [`../RUN.md`](../RUN.md).

**Reviewed against:** the SIH problem statement, `docs/proposal.md` (v2.0),
`extraction/extractor.py`, and the actual extracted data
`data/unified_Wednesday-WorkingHours_labeled.csv` (692,465 flows, 126 columns).

> File references below use short module names (`models.py`, `attack_stages.py`, …);
> all live under `sentinel_wm/` and are run as `python -m sentinel_wm.<module>`.

**Verdict:** the proposal is sound and buildable. It is a *world-model* design
(learns `P(S_{t+1} | S_{t-L:t})` and rolls it forward), not a re-skinned
classifier, and every mandatory deliverable in the problem statement maps to a
concrete module. Seven items needed a decision or correction before code; they
are listed below with what was done.

---

## 1. What the proposal gets right (keep as-is)

| Proposal choice | Why it holds up |
|---|---|
| 10-second state windows as the unit of analysis | Aggregation kills per-flow noise; a port sweep or a SYN flood is fully visible inside one window. Verified: Wednesday yields ~3,035 windows over 8.5 h. |
| Progression states as the **primary** learned target, ATT&CK as a **secondary** interpretation | CIC-IDS-2017 labels a *flow type*, never a *tactic per instant*. Training on ATT&CK tactics directly would be indefensible. This is the single most important design decision and it is correct. |
| Probabilistic State-Transition Network (μ, σ) | Gives calibrated CIs on the forecast and makes σ a usable "trajectory is unstable" signal. Implemented with reparameterised sampling + light KL. |
| Day-based split to prevent temporal leakage | Random-row splits on CIC-IDS-2017 inflate every metric. Kept — but see §2.1, only one day is currently extracted. |
| Elapsed-time positional encoding, causal masking | Both implemented in `models.py` (`ElapsedTimePositionalEncoding`, triangular mask). |
| Three-tier build (Baseline → MVP → Advanced) | Followed. This delivery is Baseline + MVP. GAT is deferred. |
| Lead Time as the headline metric | Implemented in `metrics.lead_time`; it is what separates the world model from the baselines (see §3). |

---

## 2. Corrections / decisions made during implementation

### 2.1 Split strategy (superseded — see technical_reference.md Part 1.9)
`data/unified_AllDays_labeled.csv` holds all 5 CIC-IDS-2017 days (~2.8 M flows,
15 attack families). The current default is **`stratified`** (leakage-safe):
contiguous per-day benign backbone + whole attack episodes assigned to one split
(rotating per family) + span-boundary sequence purge. `block` (now 5-min) is a
secondary mode. The paragraphs below describe the earlier `block`-default design
and are kept for history.

* **`block`.** Each day is cut into contiguous blocks, assigned
  `train, train, train, val, test` round-robin. Leakage-safe *for windows* (blocks
  are contiguous; RobustScaler fit on train only) but sequences near a block
  boundary still straddle two splits — which is why `stratified` +
  `SequenceConfig.purge_boundary_sequences` replaced it as the default.
* **`day` (secondary).** The proposal's strict Mon-Wed / Thu / Fri split. This is
  a **zero-shot new-attack-family** test — train = DoS/Patator, val =
  Web/Infiltration, test = Botnet/PortScan/DDoS — three disjoint attack regimes.
  Verified: absolute F1 collapses to ≈ 0.3-0.4 for *every* model (LR, RF, XGB,
  world model alike) because no model trained only on DoS nowcasts a port scan.
  Reported as a generalisation stress test, not the headline benchmark.

A plain chronological cut is rejected: on any single day it puts whole attack
episodes on one side (verified on Wednesday — all DoS in the first 60 %).

### 2.2 ATT&CK phase mapping — made explicit and two-layered
This was the area of greatest concern. `attack_stages.py` implements:

* **Layer A — `CIC_LABEL_TO_ATTACK`**: a static, literature-based table
  `attack_family → (tactic, technique IDs, base confidence)`. e.g.
  `DoS Hulk → Impact / T1499, T1499.002 / High`;
  `PortScan → Reconnaissance / T1595, T1046 / High`;
  `Heartbleed → Initial Access / T1190 / Medium`.
* **Layer B — `assess_window(progression_state, dominant_family, prev_family, attack_ratio)`**:
  resolves one window to a single ATT&CK phase and **lowers confidence whenever
  the evidence is thin**:
  * `NORMAL` → phase `None`, High.
  * `PRE_ATTACK` → `Reconnaissance`; Medium if the family is scan-shaped
    (PortScan/Infiltration), else **Low** with rationale
    *"benign window, tactic inferred from timing only"*.
  * `ONSET` → Layer-A tactic, confidence **downgraded one notch** (single-window
    evidence).
  * `ACTIVE` → Layer-A tactic and confidence (downgraded if `attack_ratio < 0.15`).
  * `CONTINUATION` → if the dominant family changed, emit the new tactic + an
    explicit `family_transition` string; else the episode-tail tactic, downgraded.
* **Forecast variant — `assess_forecast(state_probs, horizon_k, …)`**: takes the
  progression-head distribution at step *k*, and downgrades confidence one notch
  per 3 horizon steps (rollout error accumulates). If the model forecasts an
  attack but there is no family context, it emits `Attack (unspecified) / Low`
  rather than silently printing a phase.

Every row/return carries `attck_confidence ∈ {High, Medium, Low}` and a
plain-text `attck_rationale`. Nothing is ever presented as ground truth.
`python -m sentinel_wm.attack_stages` runs an 8-case self-test.

### 2.3 Feature schema pinned to the *actual* columns
The proposal lists `Flow IAT Variance` (not present in the unified CSV) and a
"44-column" set. `config.py` pins the locked Tier-1 (62 cols present), Tier-2
(26 cols present) and Tier-4 exclusion lists to the real column names *after* the
extractor's whitespace strip. The model actually consumes the **41-dim
window-level state vector** built by `state_windows.STATE_FEATURE_COLS`, not the
raw per-flow columns.

### 2.4 CICFlowMeter `Infinity`/`NaN` cells
`extractor.py` reproduces the CICFlowMeter `Flow Bytes/s = ∞` bug on purpose.
`preprocessing.clean_flow_frame` replaces ±inf → NaN → 0 (2,594 cells on
Wednesday) and winsorises the four unbounded rate columns at p99.9 so one
zero-duration flow cannot dominate a window aggregate.

### 2.5 Rollout heads must be supervised
First implementation trained only the direct multi-horizon projection; the shared
attack/progression heads used by `model.rollout()` were left untrained, so the
Monte-Carlo forward simulation produced a flat ~0.48 for everything. Fixed:
`joint_loss` now also supervises `apply_heads(z)` and `apply_heads(z_next)` on
the 1-step target. Rollout is now discriminative (attack anchors ≈ 0.83 at +10 s,
benign ≈ 0.40).

### 2.6 Lead-time definition bounded to the horizon
A K-step forecaster can only legitimately claim lead time up to K windows before
onset. `metrics.lead_time` caps the backward search at `horizon_k` windows and
requires the warning run to be contiguous up to onset, otherwise MLT is inflated
by isolated early warnings.

### 2.7 No Streamlit / Flask
The proposal's dashboard section is replaced by `cli.py` (terminal only) and the
JSON artifacts under `artifacts/reports/`. Everything runs fully offline.

---

### 2.8 Model zoo (expanded well beyond "vs logistic regression")
The problem statement asks for a benchmark against a logistic-regression
baseline. The build now trains and scores, on **identical sequences and the same
`metrics.py` code**:

* **Classical (`baselines.py`)** — LogisticRegression, RandomForest, ExtraTrees,
  HistGradientBoosting, sklearn-MLP, LinearSVC, kNN, GaussianNB, **XGBoost**,
  **LightGBM** — each as K per-horizon binary classifiers, in two input regimes:
  `window` (`S_t`, no temporal context) and `__seq` (flattened `L`-window
  history, same information the world model sees). This directly addresses
  *"baseline performance is weak"*: the `__seq` regime + all-days data roughly
  doubles baseline F1 vs the old single-day `S_t`-only setup.
* **Neural (`nn_zoo.py` + `nn_common.py`)** — **MLP**, **LSTM**, **GRU**,
  **TCN**; same head shape and training protocol as the world model.
* **Graph (`graph_windows.py` + `gat.py`)** — a **from-scratch Graph Attention
  Network** (masked additive attention, no `torch-geometric`) over per-window
  host-interaction graphs → GRU over time. This is the proposal's Advanced-tier
  spatial encoder, now actually built.
* **World model** — Temporal Transformer + probabilistic STN + K-step MC rollout,
  unchanged.

`benchmark.py` discovers every saved model and emits `benchmark_full.csv`
(F1/P/R/FPR/AUROC/Brier/ECE/MLT/detection/params/latency + `f1_k1..k6`),
`per_horizon_f1.csv`, `leadtime.csv`, `benchmark.md`, figures. `registry.py`
writes `research/models/registry.json` — a uniform `load_predictor(name)` for
the serving app. All of it regenerates via `python -m sentinel_wm.research all`
into the `research/` folder (every number linked to its file in
`research/reports/RESEARCH_REPORT.md`).

---

## 3. Does it beat the baselines? (all 5 days, block split)

See `research/benchmarks/benchmark.md` for the live table. Consistent findings:

* **XGBoost / RandomForest on `__seq`** give the strongest *nowcast* F1 — a
  strong, fair floor. A tree ensemble that sees the whole flattened window
  nowcasts a sustained flood well.
* **SENTINEL-WM** is the model that **holds F1 as the forecast horizon grows**,
  produces **non-zero Mean Lead Time** with a *calibrated* probability, and is
  the only one carrying the **progression-state head** and the **K-step MC
  rollout with ATT&CK phase + confidence**. The classical zoo forecasts *onset*
  at ~0 s lead time.
* **LSTM / GRU / TCN / GAT** land between: temporal/graph architecture helps, but
  the probabilistic state-transition core + rollout is what buys the lead time.

The `day` split (§2.1) is the honest hard case: every model drops to F1 ≈ 0.3-0.4
because Friday's attack families are unseen. That is a property of the split, not
the models, and it is reported as such.

---

## 4. Problem-statement coverage

| Required capability | Where |
|---|---|
| Represent network state as feature vectors / graphs | `state_windows.py` (41-dim S_t) **and** `graph_windows.py` (per-window host graphs) |
| Learn state-transition dynamics with a sequence model | `models.TemporalEncoder` + `StateTransitionNet`; also LSTM/GRU/TCN (`nn_zoo.py`) and GAT (`gat.py`) |
| Forecast future states, estimate P(attacker progression) | `models.rollout` (K-step MC), `forward_sim.py` |
| Map to MITRE ATT&CK stages | `attack_stages.py` (Layer A + B, with confidence) |
| Explainability (attention / feature attribution) | `explain.py` (attention saliency + gradient×input + real SHAP) |
| Ingest CSV (and PCAP) → normalised feature matrix | `extraction/extractor.py` (PCAP) + `preprocessing.py` (CSV) |
| Trained model + weights + reproducible config | `artifacts/world_model.pt` + `research/models/` + `registry.json` |
| Infiltration prediction engine (prob + stage + top features) | `forward_sim.simulate_anchor` |
| Benchmark vs logistic-regression baseline, same features | `baselines.py` (11-model zoo) + `benchmark.py` → `research/benchmarks/` |
| Runs fully offline, no cloud | yes — no network calls anywhere |
| Demonstration interface (CLI, not Flask/Streamlit) | `cli.py`, `research.py` |
