# SENTINEL-WM — Complete Usage Guide

**S**patio-temporal **E**nemy **N**etwork **I**ntelligence with a **L**earned **World Model**.

A fully-offline research prototype that learns how a network's *state* evolves
from traffic telemetry and **forecasts attacker progression K windows ahead**,
with a calibrated probability, a MITRE ATT&CK-aligned phase (with an explicit
confidence), and a feature-level explanation for every prediction.

No Flask, no Streamlit, no cloud. Everything is a terminal command or a notebook.

---

## 0. TL;DR

```bash
pip install -e .            # or:  pip install -r requirements.txt

# one command runs phases 1-6 end to end and prints the benchmark
python -m sentinel_wm.cli all --epochs 40
```

Outputs land in `artifacts/` (parquet, `world_model.pt`, scaler) and
`artifacts/reports/` (`benchmark.md`, `world_model_metrics.json`,
`forward_sim_test.json`, `explainability.json`).

---

## 1. What is in this repo

```
sentinel_wm/                         the Python package (`pip install -e .`)
  config.py              hyper-parameters, paths, LOCKED feature tiers
  preprocessing.py       PHASE 1  raw unified CSV  -> artifacts/clean_flows.parquet
  state_windows.py       PHASE 2  clean flows      -> artifacts/state_windows.parquet
  attack_stages.py       MITRE ATT&CK phase mapping (Layer A static + Layer B context)
  sequences.py           PHASE 2c windows          -> artifacts/sequences.npz  (+ scaler)
  baselines.py           PHASE 3  LogisticRegression / RandomForest comparison floor
  models.py              the world model: Temporal Transformer + probabilistic STN + heads
  train.py               PHASE 4  train / --test the world model
  forward_sim.py         PHASE 5  K-step Monte-Carlo forward simulation + ATT&CK per step
  explain.py             PHASE 6  SHAP (or fallback) + attention + gradient saliency
  evaluate.py            benchmark table: world model vs baselines  -> benchmark.md
  metrics.py             shared metric fns (F1/FPR/AUROC, Brier/ECE, Mean Lead Time)
  cli.py                 the offline command-line interface (all of the above)

extraction/
  extractor.py           PCAP -> unified flow+packet CSV via tshark (standalone)
  label_mapping.ipynb    maps official CIC-IDS-2017 labels onto the extractor output

notebooks/
  01_data_preprocessing.ipynb    PHASE 1 walkthrough with plots
  02_sequence_generation.ipynb   PHASE 2 walkthrough: states, ATT&CK mapping, split
  03_model_training.ipynb        PHASE 3-6 walkthrough: baselines, training, sim, explain

docs/
  proposal.md            the SENTINEL-WM v2.0 design document
  plan_validation.md     what was reviewed / corrected vs the proposal
  guide.md               this file
  problem_statement.pdf

data/                    input CSVs (gitignored) - see data/README.md
artifacts/               all generated data + weights + reports (gitignored)
pyproject.toml  requirements.txt  README.md  .gitignore
```

Input expected: `unified_*_labeled.csv` under `data/`, listed in
`sentinel_wm.config.RAW_FLOW_CSVS`. Ships with
`data/unified_Wednesday-WorkingHours_labeled.csv`.

---

## 2. The pipeline, phase by phase

```
 raw unified CSV  (692k flows, 126 cols)
        │  preprocessing.py           inf/NaN fix, label -> family, day, time axis
        ▼
 artifacts/clean_flows.parquet
        │  state_windows.py           10s windows -> S_t (41 dims)
        │                             + progression_state (episodes)
        │                             + ATT&CK phase/confidence (attack_stages.py)
        ▼
 artifacts/state_windows.parquet
        │  sequences.py               [S_{t-9..t}] -> (S_{t+1}, A_{t+1..6}, Z_{t+1..6})
        │                             block-interleaved split, train-only RobustScaler
        ▼
 artifacts/sequences.npz  +  artifacts/state_scaler.pkl
        │
        ├── baselines.py              LR / RF on S_t only  (no temporal modelling)
        │
        └── train.py                  Temporal Transformer + STN + attack/prog heads
                    │
                    ▼
             artifacts/world_model.pt
                    │
                    ├── forward_sim.py   K-step MC rollout -> P(attack) timeline + ATT&CK
                    ├── explain.py        feature attribution + attention saliency
                    └── evaluate.py       benchmark.md  (world model vs baselines)
```

### PHASE 1 — pre-processing
```bash
python -m sentinel_wm.cli preprocess          # or:  python -m sentinel_wm.preprocessing
```
* strips column whitespace, normalises the `Label` string to a canonical
  `attack_family` (`DoS Hulk`, `PortScan`, `Heartbleed`, …) and an `is_attack` bit;
* replaces CICFlowMeter `±Infinity`/`NaN` rate cells with 0 and winsorises the
  four unbounded rate columns at p99.9;
* derives `day` from `source_file`; sorts by `flow_start_epoch` (the single time axis);
* keeps `Flow ID` / IPs / ports **for windowing only** — they are in
  `config.IDENTITY_COLS` and never enter a model tensor.

### PHASE 2 — network state construction
```bash
python -m sentinel_wm.cli windows             # or:  python -m sentinel_wm.state_windows
```
* buckets flows into `config.WindowConfig.window_seconds` (10 s) windows
  (`stride_seconds` < window ⇒ overlapping windows, rows are replicated);
* aggregates each window into `STATE_FEATURE_COLS` — volume/rate, connection
  dynamics (fan-out/in, unique pairs, failed-conn ratio), TCP-flag rates,
  timing/burstiness, Tier-2 packet stats (TTL var, window size, payload moments,
  retransmission rate), scanning signatures (port-scan entropy / sequential
  ratio), protocol mix, plus two safe time-derived features
  (`time_since_prev_window`, `window_index_in_day`);
* **progression state** (`_derive_progression`): groups consecutive attack
  windows into *episodes* (bridging ≤ `episode_gap_windows` benign holes), then
  labels NORMAL / PRE_ATTACK / ONSET / ACTIVE / CONTINUATION;
* **binary target** `y_attack` = 1 for ONSET/ACTIVE/CONTINUATION;
* attaches the ATT&CK assessment (next section).

### The ATT&CK phase mapping (`attack_stages.py`) — read this
CIC-IDS-2017 gives an attack *type per flow*, never a *tactic per moment*. So the
mapping is a deterministic **two-layer** function applied **after** the model:

**Layer A — `CIC_LABEL_TO_ATTACK`** (static, from literature):

| family | tactic | techniques | base confidence |
|---|---|---|---|
| BENIGN | None | — | High |
| PortScan | Reconnaissance / Discovery | T1595, T1046 | High |
| FTP/SSH-Patator | Credential Access / Initial Access | T1110, T1021 | Medium |
| Heartbleed | Initial Access / Credential Access | T1190 | Medium |
| DoS Hulk / GoldenEye | Impact | T1499, T1499.002 | High |
| DoS slowloris / Slowhttptest | Impact | T1499, T1499.003 | High |
| DDoS | Impact | T1498 | High |
| Bot | Command & Control | T1071, T1572 | Medium |
| Infiltration | Initial Access → Discovery → Lateral Movement | T1190, T1046, T1021 | Medium |

**Layer B — `assess_window(state, family, prev_family, attack_ratio)`**
resolves one window to *one* phase and **lowers confidence when evidence is weak**:

| progression state | phase | confidence rule |
|---|---|---|
| NORMAL | None | High |
| PRE_ATTACK | Reconnaissance | Medium if family is scan-shaped, else **Low** ("inferred from timing only") |
| ONSET | Layer-A tactic | **downgraded one notch** (single-window evidence) |
| ACTIVE | Layer-A tactic | Layer-A confidence, −1 if `attack_ratio < 0.15` |
| CONTINUATION | new tactic if family changed (+`family_transition` string), else episode-tail tactic | downgraded one notch |

**Forecast variant — `assess_forecast(state_probs, horizon_k, …)`**: takes the
progression-head distribution at horizon step *k*, drops confidence one notch per
3 steps, and emits `Attack (unspecified) / Low` if it predicts an attack with no
family context. Every output has `attck_confidence ∈ {High,Medium,Low}` and a
text `attck_rationale`. Self-test: `python -m sentinel_wm.attack_stages`.

To tune the mapping: edit `CIC_LABEL_TO_ATTACK` (add families / techniques) and
the rules in `assess_window` / `assess_forecast`. Nothing else depends on the
internals — `state_windows.py` and `forward_sim.py` only call the public functions.

### PHASE 2c — sequences + split
```bash
python -m sentinel_wm.cli sequences           # or:  python -m sentinel_wm.sequences
```
* builds `X [N,10,41]`, `dt [N,10]`, `x_next [N,41]`, `y_atk [N,6]`,
  `y_prog [N,6]`, `y_now [N]`, `split [N]`;
* **split** (`config.SplitConfig.mode`):
  * `auto` → **`day`** if >1 CIC-IDS-2017 day is present
    (Mon-Wed=train / Thu=val / Fri=test), else **`block`**;
  * `block` (single-day default): 5-minute blocks assigned
    `train,train,train,val,test` round-robin so every split has attack windows;
  * also available: `chronological`, `family` (train on some DoS families,
    test on others);
* RobustScaler is fit on the **training split only** and pickled to
  `artifacts/state_scaler.pkl`; `dt` is `log1p`-compressed.

### PHASE 3 — baselines
```bash
python -m sentinel_wm.cli baseline            # or:  python -m sentinel_wm.baselines
```
Logistic Regression (mandated) + Random Forest, each trained on the **current
window `S_t` only** — one classifier per horizon `k=1..6`. A single alert
threshold is calibrated on validation to FPR ≤ 5 %. Plus a trivial `persistence`
reference (`A_{t+k}=A_t`). Metrics → `artifacts/baselines/baseline_metrics.json`.

### PHASE 4 — world model
```bash
python -m sentinel_wm.cli train --epochs 40                 # GPU auto-detected
python -m sentinel_wm.train --epochs 40 --device cpu
python -m sentinel_wm.train --test              # evaluate an existing checkpoint
```
Architecture (`models.SentinelWorldModel`):

```
X [B,10,41] , dt ──► in_proj ──► + elapsed-time positional enc
                          │
                 3 × causal self-attention blocks (4 heads)   ← attn weights kept
                          │
                        z_t  [B,128]
             ┌────────────┼─────────────────────────────┐
             ▼            ▼                             ▼
   horizon_attack   StateTransitionNet            shared heads
   z_t → [B,6]      z_t → μ,logσ  (clamped)       attack: z→P(A=1)
   horizon_prog                 │                 progress: z→P(Z=c)
   z_t → [B,6,5]        z_{t+1} ~ N(μ,σ)          (also applied to z_{t+1})
```

Joint loss (`models.joint_loss`, weights in `config.ModelConfig`):
`w_attack·BCE(attack_k) + w_progression·CE(prog_k)
 + 0.5·(BCE/CE on the shared heads, 1-step target)
 + w_next_state·MSE(μ_{t+1}, z_{t+1}_target) + w_kl·KL`.
Class-weighted (attack `pos_weight`, progression √-inverse-frequency).
Early-stops on `val F1 + 0.5·prog_acc`; freezes the alert threshold on val;
evaluates the untouched test split. Checkpoint `artifacts/world_model.pt` embeds
the full config, feature names, sequence L/K and the calibrated threshold.

### PHASE 5 — forward simulation
```bash
python -m sentinel_wm.cli simulate --split test --explain
python -m sentinel_wm.forward_sim --split test --index 0     # dump one anchor as JSON
```
For each anchor window: `model.rollout` draws `config.TrainConfig.mc_samples`
(50) Monte-Carlo latent trajectories through the STN, applies the heads at every
step, and returns per horizon `k`:
`attack_prob` + 95 % CI + std, predicted `progression_state` + full distribution,
and `attack_stages.assess_forecast(...)` → ATT&CK phase + confidence + rationale.
`--explain` adds per-anchor gradient×input top features and attention saliency.
Result → `artifacts/reports/forward_sim_<split>.json`; a text timeline is printed.

### PHASE 6 — explainability
```bash
python -m sentinel_wm.cli explain
```
* **SHAP** on a `flattened-sequence → P(attack in next K)` wrapper for the world
  model, and `LinearExplainer`/`TreeExplainer` for the baselines. **If `shap` is
  not installed**, a finite-difference / coefficient fallback returns the same
  ranked-feature structure — clearly labelled `finite-difference-fallback`.
* **Attention saliency**: last-block attention summed over queries → per-history-
  window importance ("the burst 4 windows ago drove this").
* **Gradient×input** on the state sequence → per (window, feature) attribution.
Report → `artifacts/reports/explainability.json`.

### Benchmark
```bash
python -m sentinel_wm.cli evaluate            # -> artifacts/reports/benchmark.md
```

---

## 3. How to read the results

`artifacts/reports/benchmark.md` on the shipped single Wednesday (DoS) day:

| Model | F1 (any-k) | AUROC | **Mean Lead Time** | **Episodes warned** | FA rate |
|---|---|---|---|---|---|
| Logistic Regression | 0.75 | 0.92 | 10 s | 1 / 7 | 0.03 |
| Random Forest | 0.82 | 0.97 | 0 s | 0 / 7 | 0.01 |
| Persistence | 0.84 | 0.87 | 0 s | 0 / 7 | 0.00 |
| **SENTINEL-WM** | 0.80 | 0.95 | **30 s** | **4 / 7** | 0.05 |

* A tree ensemble is strong at *nowcasting a flood that is already obvious in
  `S_t`* — expected, not the goal.
* The world model wins on **lead time** (30 s vs ~0) and **holds F1 as the
  horizon grows** (0.80 → 0.73 from +10 s to +60 s). Progression-state accuracy
  0.89; Brier(k1) 0.06; ECE(k1) 0.08 (well calibrated).
* Wednesday is DoS-only and DoS onsets are abrupt, so there is little pre-attack
  signal to forecast — this caps lead time. The proposal's headline
  **Infiltration** scenario (Thursday) is the one that shows large lead time.

---

## 4. Adding more CIC-IDS-2017 days (recommended next step)

1. Extract each day to `unified_<Day>-WorkingHours_labeled.csv` with
   `extractor.py` + `extraction/label_mapping.ipynb` (same process that produced the Wednesday file).
2. Add the paths to `config.RAW_FLOW_CSVS`.
3. Re-run `python -m sentinel_wm.cli all`. `SplitConfig.mode="auto"` now selects the
   proposal's **day-based** split (Mon-Wed / Thu / Fri) automatically — no code
   change. The ATT&CK table already covers Patator, Web attacks, PortScan, Bot,
   DDoS and Infiltration.

---

## 5. Configuration cheatsheet (`sentinel_wm/config.py`)

| Knob | Default | Effect |
|---|---|---|
| `WindowConfig.window_seconds` | 10 | state-window length |
| `WindowConfig.stride_seconds` | 10 | set to 5 for 50 % overlap |
| `WindowConfig.pre_attack_span` | 3 | windows before onset flagged PRE_ATTACK |
| `WindowConfig.episode_gap_windows` | 2 | benign holes bridged inside an episode |
| `SequenceConfig.history` (L) | 10 | input windows (100 s of history) |
| `SequenceConfig.horizon` (K) | 6 | forecast steps (60 s ahead) |
| `SplitConfig.mode` | `auto` | `day` / `block` / `chronological` / `family` |
| `SplitConfig.block_minutes` | 5 | block size for the single-day split |
| `ModelConfig.d_model / n_heads / n_layers` | 128 / 4 / 3 | transformer size |
| `ModelConfig.w_*` | see file | joint-loss weights |
| `TrainConfig.epochs / batch_size / lr` | 40 / 256 / 1e-4 | training |
| `TrainConfig.mc_samples` (M) | 50 | Monte-Carlo rollout samples |
| `TrainConfig.target_fpr` | 0.05 | alert threshold auto-calibrated to this |

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: No raw flow CSVs` | put `unified_*_labeled.csv` in the folder / fix `config.RAW_FLOW_CSVS` |
| `'Label' column missing` | run `extraction/label_mapping.ipynb` first — the world model needs labelled flows |
| `training split is empty` | wrong `SplitConfig`; with one day keep `mode="auto"`/`block` |
| rollout probs all ≈ 0.4-0.5 | retrain — an old checkpoint predates the shared-head loss fix |
| `shap` import errors | optional; the fallback runs automatically. `pip install shap` to enable |
| CUDA OOM | `--device cpu` (≈ 1 min for 40 epochs on this dataset) |
| `ModuleNotFoundError: sentinel_wm` | run `pip install -e .` from the repo root, or run notebooks from `notebooks/` (they self-bootstrap `sys.path`) |
| PCAP re-extraction | needs the `tshark` binary on PATH (install Wireshark) + `pip install scapy` |

---

## 7. Reproducibility

* Seed: `TrainConfig.seed = 1337` (numpy + torch). GPU cuDNN nondeterminism can
  still move the 3rd decimal.
* `artifacts/world_model.pt` embeds `config`, `feature_names`, `sequence L/K`,
  `alert_threshold` — enough to re-instantiate and score without re-reading
  `sentinel_wm/config.py`.
* Every phase writes a deterministic artifact; delete `artifacts/` to force a
  clean rebuild.
