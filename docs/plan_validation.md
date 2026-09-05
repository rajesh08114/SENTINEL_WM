# SENTINEL-WM — Plan Validation & Implementation Notes

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

### 2.1 Only one day is extracted — the split had to change (for now)
The proposal's `Mon-Wed / Thu / Fri` split needs ≥3 day files. Only Wednesday
exists. A plain chronological 60/20/20 cut is **worse than useless** here:
verified that it puts *every* DoS episode in the first 60 % and leaves val/test
almost attack-free (Heartbleed = 11 flows).

**Decision (block-interleaved):** `sequences.assign_split` cuts each day into
5-minute blocks and assigns them `train, train, train, val, test` round-robin, so
every split contains attack windows (train k=1 positives 13.6 %, val 7.8 %,
test 16.0 %). The RobustScaler is still fit on **train only**.
`mode="auto"` switches back to the proposal's true day-based split the moment a
second day file is added to `config.RAW_FLOW_CSVS` — no code change.
This is documented as a single-day compromise, not presented as leakage-free.

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

## 3. Does it actually beat the baseline? (single-day, block split)

From `artifacts/reports/benchmark.md` (test split, FPR-calibrated threshold):

| Model | F1 (any-k) | AUROC | **Mean Lead Time** | **Episodes warned** | FA rate |
|---|---|---|---|---|---|
| Logistic Regression (S_t only) | 0.752 | 0.921 | 10 s | 1 / 7 | 0.032 |
| Random Forest (S_t only) | 0.815 | 0.974 | 0 s | 0 / 7 | 0.008 |
| Persistence (A_{t+k}=A_t) | 0.836 | 0.867 | 0 s | 0 / 7 | 0.000 |
| **SENTINEL-WM** | 0.800 | 0.949 | **30 s** (max 60 s) | **4 / 7** | 0.046 |

**Reading this honestly:**
* On *nowcast* F1 for a **sustained** flood, a tree ensemble / persistence is
  hard to beat — the attack is already blatant in `S_t`. That is expected and
  not the point.
* The world model wins where the proposal says it should: **lead time** (30 s vs
  ~0) and **holding F1 as the horizon grows** — SENTINEL-WM F1 goes
  0.796 → 0.725 from +10 s to +60 s, while the baselines are only "good" at long
  horizons *because DoS persists*, not because they forecast onset.
* Progression-state accuracy 0.887; Brier(k1) 0.060; ECE(k1) 0.083 — well
  calibrated.
* Model is 0.77 M parameters, trains in ~10 s on GPU / ~1 min on CPU.

**The demonstration gap:** Wednesday is DoS-only. DoS onsets are abrupt (no
recon ramp-up), so there is genuinely little "pre-attack" signal to forecast —
which caps lead time at ~60 s and detection at 4/7. The proposal's headline
scenario (Thursday **Infiltration**: external recon → compromise → internal
Nmap) is the one that shows large lead time, and it needs the Thursday files.
Everything is wired so that adding them is a data step, not a code step.

---

## 4. Problem-statement coverage

| Required capability | Where |
|---|---|
| Represent network state as feature vectors / graphs | `state_windows.py` (41-dim S_t); graph left as Advanced tier |
| Learn state-transition dynamics with a sequence model | `models.TemporalEncoder` + `StateTransitionNet` |
| Forecast future states, estimate P(attacker progression) | `models.rollout` (K-step MC), `forward_sim.py` |
| Map to MITRE ATT&CK stages | `attack_stages.py` (Layer A + B, with confidence) |
| Explainability (attention / feature attribution) | `explain.py` (attention saliency + gradient×input + SHAP-or-fallback) |
| Ingest CSV (and PCAP) → normalised feature matrix | `extractor.py` (PCAP) + `preprocessing.py` (CSV) |
| Trained model + weights + reproducible config | `artifacts/world_model.pt` (+ embedded config), `config.py` |
| Infiltration prediction engine (prob + stage + top features) | `forward_sim.simulate_anchor` |
| Benchmark vs logistic-regression baseline, same features | `baselines.py` + `evaluate.py` → `benchmark.md` |
| Runs fully offline, no cloud | yes — `cli.py`, no network calls anywhere |
| Demonstration interface (CLI, not Flask/Streamlit) | `cli.py demo` |
