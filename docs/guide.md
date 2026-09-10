# SENTINEL-WM — Complete Usage Guide

**S**patio-temporal **E**nemy **N**etwork **I**ntelligence with a **L**earned **World Model**.

A fully-offline research prototype that learns how a network's *state* evolves
from traffic telemetry and **forecasts attacker progression K windows ahead**,
with a calibrated probability, a MITRE ATT&CK-aligned phase (with an explicit
confidence), and a feature-level explanation for every prediction.

This guide covers the **`research/`** half — the offline ML pipeline (CLI +
notebooks, no web UI). The serving application (FastAPI + Next.js) is
[`../backend/`](../backend) and [`../frontend/`](../frontend).

---

## 0. TL;DR

```bash
cd research
pip install -e ".[benchmark]"                # core + xgboost + lightgbm + shap (all optional)

# the whole study: data -> every model -> benchmark -> explain -> simulate
python -m sentinel_wm.research all           # ~30-45 min on a GPU
python -m sentinel_wm.research all --quick   # small epochs, ~10 min

python -m sentinel_wm.cli all                # lighter: phase runner, ends at the benchmark
sentinel-wm bundle ../models                 # assemble the deploy bundle for backend/
```

`research all` populates **`../runs/`** (models, benchmarks, figures,
explainability, simulations, `reports/RESEARCH_REPORT.md`). Pipeline working files
stay in **`../artifacts/`** (parquet, `world_model.pt`, `graph_windows.npz`,
scaler). The portable **`../models/`** bundle is what the backend loads.

---

## 1. What is in this repo

```
research/                            the ML workspace (this guide)
 sentinel_wm/                        the Python package (`cd research && pip install -e .`)
  config.py              hyper-parameters, paths, LOCKED feature tiers
  preprocessing.py       PHASE 1  raw unified CSV  -> artifacts/clean_flows.parquet
  state_windows.py       PHASE 2  clean flows      -> artifacts/state_windows.parquet
  attack_stages.py       MITRE ATT&CK phase mapping (Layer A static + Layer B context)
  sequences.py           PHASE 2c windows          -> artifacts/sequences.npz  (+ scaler)
  baselines.py           PHASE 3  classical model zoo (10 models x window/sequence)
  models.py              the world model: Bi-GRU + attention encoder + probabilistic STN + heads
  train.py               PHASE 4  train / --test the world model
  nn_zoo.py              neural baselines: MLP / LSTM / GRU / TCN  (nn.Module defs)
  nn_common.py           shared train / eval / checkpoint loop for every deep model
  graph_windows.py       per-window host-interaction graphs -> artifacts/graph_windows.npz
  gat.py                 from-scratch Graph Attention Network baseline (no torch-geometric)
  forward_sim.py         PHASE 5  K-step Monte-Carlo forward simulation + ATT&CK per step
  explain.py             PHASE 6  SHAP (or fallback) + attention + gradient saliency
  benchmark.py           unified scoreboard: every saved model, same test anchors
  registry.py            one uniform loader for every saved model (for the app)
  research.py            orchestrator -> the runs/ folder
  bundle.py              `sentinel-wm bundle` -> the portable ../models/ deploy bundle
  evaluate.py            back-compat shim -> benchmark.py
  metrics.py             shared metric fns (F1/FPR/AUROC, Brier/ECE, Mean Lead Time)
  cli.py                 the offline command-line interface (all of the above)

 extraction/
  extractor.py           PCAP -> unified flow+packet CSV via tshark (standalone)
  label_mapping.ipynb    maps official CIC-IDS-2017 labels onto the extractor output

 notebooks/               00-05 phase notebooks (run from research/notebooks/)
 pyproject.toml  requirements.txt  RUN.md  README.md

docs/        proposal.md  plan_validation.md  guide.md  technical_reference.md
data/        input CSVs (gitignored) - see data/README.md
artifacts/   pipeline working files (gitignored)
runs/        generated benchmark study (gitignored) - see research/README.md
models/      the deploy bundle the backend loads (gitignored; `sentinel-wm bundle`)
backend/  frontend/   the serving application
```

Input expected: a `unified_*_labeled.csv` under `data/`, listed in
`sentinel_wm.config.RAW_FLOW_CSVS`. Ships with
`data/unified_AllDays_labeled.csv` (all 5 CIC-IDS-2017 days).

---

## 2. The pipeline, phase by phase

```
 data/unified_AllDays_labeled.csv   (~2.8M flows, all 5 days, 15 families)
        │  preprocessing.py           inf/NaN fix, label -> family, day, time axis
        ▼
 artifacts/clean_flows.parquet
        │  state_windows.py           10s windows -> S_t (53 dims: 41 base
        │                             + 4 entropy + 8 first-differences)
        │                             + progression_state (episodes)
        │                             + ATT&CK phase/confidence (attack_stages.py)
        ▼
 artifacts/state_windows.parquet ───────────┐  graph_windows.py -> per-window host graphs
        │  sequences.py                      ▼  artifacts/graph_windows.npz
        │  [S_{t-11..t}] -> (S_{t+1}, A_{t+1..6}, Z_{t+1..6})
        │  leakage-safe stratified split + span-boundary purge, train-only RobustScaler
        ▼
 artifacts/sequences.npz  +  artifacts/state_scaler.pkl
        │
        ├── baselines.py    classical zoo (10 models x window/__seq)   -> runs/models/classical/
        ├── nn_zoo.py        MLP / LSTM / GRU / TCN  (via nn_common)     -> runs/models/nn/
        ├── gat.py           from-scratch Graph Attention Network       -> runs/models/nn/gat.pt
        └── train.py         world model: Bi-GRU + attention + STN      -> artifacts/world_model.pt
                    │
                    ├── forward_sim.py   K-step MC rollout -> P(attack) timeline + ATT&CK
                    ├── explain.py        SHAP + attention + gradient saliency
                    ├── benchmark.py      every model, same test anchors -> runs/benchmarks/
                    └── registry.py       runs/models/registry.json  (uniform load_predictor)
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
* builds `X [N,12,53]`, `dt [N,12]`, `x_next [N,53]`, `y_atk [N,6]`,
  `y_prog [N,6]`, `y_now [N]`, `split [N]` (`"train"|"val"|"test"|"ignore"`),
  `dominant_family [N]`, `is_synthetic [N]`;
* **split** (`config.SplitConfig.mode`, default `"auto"` → `"stratified"`):
  * **`stratified`** (primary, leakage-safe) — benign windows get a contiguous
    per-day 60/20/20 backbone; a **whole attack episode** is assigned to one
    split, rotating per family (`stratified_episode_rotation`) so families with
    ≥3 episodes span all 3 splits; only episodes ≥ `stratified_long_episode_windows`
    (54 = 3·(L+K)) are cut internally. Sequences whose `[t-L+1..t+K]` span crosses
    a split boundary are dropped (`SequenceConfig.purge_boundary_sequences`), and
    `build_sequences` asserts every retained sequence is span-pure. Scaler fit on
    real-train only. Synthetic (flow-augmented) windows are always train. Single-
    burst families (e.g. DoS GoldenEye) may land in only 1–2 splits — a real
    dataset property, shown by the per-family metrics. See
    `docs/technical_reference.md` Part 1.9.
  * **`block`** — each day cut into `block_minutes` (5) contiguous blocks,
    assigned `train,train,train,val,test` round-robin. Kept as a secondary
    benchmark.
  * **`day`** — strict CIC-IDS-2017 day split (Mon-Wed / Thu / Fri) = *zero-shot
    new-attack-family* test. Run into its own folder:
    `python -m sentinel_wm.research all --split day --outdir research_zeroshot`.
  * **`family`** — attack-family hold-out (`family_val` / `family_test` excluded
    from train); the other zero-shot benchmark, same `--outdir research_zeroshot`.
  * also: `episode_chrono` (lead-time-focused), `chronological`.
* RobustScaler is fit on the **training split only** and pickled to
  `artifacts/state_scaler.pkl`; `dt` is `log1p`-compressed.

### PHASE 3 — classical model zoo
```bash
python -m sentinel_wm.cli baseline            # or:  python -m sentinel_wm.baselines
```
LogisticRegression, RandomForest, ExtraTrees, HistGradientBoosting, sklearn-MLP,
LinearSVC, kNN, GaussianNB, **XGBoost**, **LightGBM** (the last two skipped if
not installed) — each as **K independent binary classifiers** (one per horizon
`k=1..6`), in two input regimes:

* `window`   — the current state vector `S_t` only  (`[N, 53]`, "no temporal context")
* `__seq`    — the flattened `L`-window history       (`[N, 636]`, same info as the world model)

Class imbalance: `class_weight="balanced"` where the estimator supports it, else
balanced `sample_weight`. A single alert threshold is FPR-calibrated on
validation. Plus a `persistence` reference (`A_{t+k}=A_t`). Fitted estimators →
`runs/models/classical/<name>.pkl` (+ `.meta.json`); metrics →
`artifacts/baselines/baseline_metrics.json`.

### PHASE 4b — neural sequence baselines
```bash
python -m sentinel_wm.cli nn --kinds mlp lstm gru tcn --epochs 50
```
`nn_zoo.py` defines four `nn.Module`s that consume the SAME `X [B,L,F]`, `dt [B,L]`
tensor and emit the SAME heads (`attack_logits_k [B,K]`, `prog_logits_k [B,K,S]`)
as the world model, so `nn_common.train_nn` trains and scores them with the
identical protocol (class-weighted BCE+CE, AdamW+cosine, early stop, FPR-calibrated
threshold, checkpoint). Checkpoints → `runs/models/nn/{mlp,lstm,gru,tcn}.pt`.

### PHASE 4c — Graph Attention Network
```bash
python -m sentinel_wm.cli graphwindows        # build per-window host graphs
python -m sentinel_wm.cli gat --epochs 50
```
`graph_windows.py` rebuilds, for every 10-s window, a directed host-interaction
graph (nodes = the busiest hosts, 14 features each; edges = `log1p(flow count)`),
padded to `N_max=32`, saved to `artifacts/graph_windows.npz` aligned to
`state_windows.parquet`. `gat.py` runs a **from-scratch** GAT (masked additive
attention, no `torch-geometric`) per window → GRU over the `L` window embeddings →
the shared heads. Trains through `nn_common` via a `batch_forward` hook.
Checkpoint → `runs/models/nn/gat.pt`.

### PHASE 4 — world model
```bash
python -m sentinel_wm.cli train --epochs 40                 # GPU auto-detected
python -m sentinel_wm.train --epochs 40 --device cpu
python -m sentinel_wm.train --test              # evaluate an existing checkpoint
```
Architecture (`models.SentinelWorldModel`; encoder chosen by `ModelConfig.encoder`,
default `"gru"` — the causal Temporal Transformer is still selectable with
`"transformer"`):

```
X [B,12,53] , dt ──► in_proj (+ log-dt channel)
                          │
        Bi-GRU (2 layers, d/2 each dir) + MultiheadAttention read-out on S_t
                          │
                        z_t  [B,160]
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

### Benchmark + registry
```bash
python -m sentinel_wm.cli benchmark          # -> runs/benchmarks/*  + registry.json
```
`benchmark.py` discovers every model in `runs/models/` + `artifacts/world_model.pt`,
scores them all on the **same test anchors** with the same `metrics.py` code, and
writes `benchmark_full.csv` (F1/P/R/FPR/AUROC/Brier/ECE/MLT/detection/params/
infer-ms + `f1_k1..k6`), `per_horizon_f1.csv`, `leadtime.csv`, `benchmark.md`,
`benchmark.json`, and figures (`horizon_f1`, `roc`, `pr`, `lead_time`).
`registry.build_registry()` writes `runs/models/registry.json` — the uniform
index the serving app loads via `registry.load_predictor(name)`.

### The whole study
```bash
python -m sentinel_wm.research all            # primary (stratified) -> runs/
python -m sentinel_wm.research all --split family --outdir research_zeroshot   # zero-shot
python -m sentinel_wm.research report         # fold the zero-shot section into RESEARCH_REPORT.md
python -m sentinel_wm.research <step>         # rerun one stage
python -m sentinel_wm.research flowaug        # (opt-in) build clean_flows_aug.parquet
```
Steps: `flowaug`, `profile`, `baselines`, `worldmodel`, `nn`, `gat`, `benchmark`,
`explain`, `simulate`, `report`. `all` runs `flowaug` first automatically when
`WindowConfig.flow_augment` is `True`.

---

## 3. How to read the results

The live table is `runs/benchmarks/benchmark.md`; the narrative with every
claim linked to its file is `runs/reports/RESEARCH_REPORT.md`. Regenerate
with `python -m sentinel_wm.research all`.

What to look for (`stratified` split, all 5 days), ranked by **PR-AUC**:

* **XGBoost / RandomForest on `__seq`** are the strongest *nowcast* F1 — a fair,
  strong floor. The proposal only required logistic regression; this is much more.
* **SENTINEL-WM** is the model that (a) **holds F1 as the horizon grows**,
  (b) produces **non-zero Mean Lead Time** with a calibrated probability,
  (c) carries the **progression-state head** and the **K-step Monte-Carlo rollout
  with ATT&CK phase + confidence** — none of which the baselines have. A tree
  ensemble that only sees `S_t` (or even `__seq`) nowcasts a sustained flood well
  but forecasts *onset* at ~0 s lead time.
* **LSTM / GRU / TCN / GAT** show that temporal/graph architecture helps, but the
  probabilistic state-transition core + rollout is what buys the lead time.
* Progression-state accuracy, Brier(k1), ECE(k1) are in the `*_metrics.json` and
  the benchmark CSV.

**Per-attack-family breakdown** — `runs/benchmarks/per_family.csv` +
`benchmark.md` block scores each family against the shared benign background;
families with < 8 positive test windows (Heartbleed / Infiltration / SQL-Injection)
are reported as excluded, not failures.

**Harder secondary benchmark** — `python -m sentinel_wm.research all --split family
--outdir research_zeroshot` (or `--split day`): whole attack families are held out
of training. Zero-shot; absolute F1 drops to ~0.3–0.5 for *every* model (that is
the point). `python -m sentinel_wm.research report` then folds a "Zero-shot
generalisation" section into the primary `RESEARCH_REPORT.md`.

---

## 4. Data

`data/unified_AllDays_labeled.csv` (all 5 days, ~2.8 M flows, 15 families) is the
default input. To rebuild it or add other datasets see
[`../data/README.md`](../data/README.md): run `extraction/extractor.py` on each
day's PCAP then `extraction/label_mapping.ipynb`, and point
`config.RAW_FLOW_CSVS` at the result. The ATT&CK table in `attack_stages.py`
already covers Patator, Web attacks, PortScan, Bot, DDoS and Infiltration.

---

## 5. Configuration cheatsheet (`sentinel_wm/config.py`)

| Knob | Default | Effect |
|---|---|---|
| `WindowConfig.window_seconds` | 10 | state-window length |
| `WindowConfig.stride_seconds` | 10 | set to 5 for 50 % overlap |
| `WindowConfig.pre_attack_span` | 3 | windows before onset flagged PRE_ATTACK |
| `WindowConfig.episode_gap_windows` | 2 | benign holes bridged inside an episode |
| `SequenceConfig.history` (L) | 12 | input windows (120 s of history) |
| `SequenceConfig.horizon` (K) | 6 | forecast steps (60 s ahead) |
| `SplitConfig.mode` | `auto` (→`stratified`) | `stratified` / `block` / `day` / `family` / `episode_chrono` / `chronological` |
| `SplitConfig.stratified_fracs` | (0.6, 0.2, 0.2) | benign backbone + long-episode internal cut |
| `SplitConfig.stratified_benign` | `contiguous` | `contiguous` (leakage-safe) or `block` |
| `SplitConfig.stratified_episode_rotation` | `(train,val,train,test)` | per-family whole-episode split cycle |
| `SplitConfig.stratified_long_episode_windows` | 54 | episodes ≥ this are cut internally 60/20/20 |
| `SequenceConfig.purge_boundary_sequences` | `True` | drop sequences whose span crosses a split boundary |
| `SplitConfig.block_minutes` | 5 | block size for the `block` split |
| `WindowConfig.flow_augment` | `False` | train-only flow-level augmentation (`flow_augment.py`) |
| `ModelConfig.d_model / n_heads / n_layers` | 128 / 4 / 3 | transformer size |
| `ModelConfig.w_*` | see file | joint-loss weights |
| `TrainConfig.epochs / batch_size / lr` | 40 / 256 / 1e-4 | training |
| `TrainConfig.mc_samples` (M) | 50 | Monte-Carlo rollout samples |
| `TrainConfig.target_fpr` | 0.05 | alert threshold auto-calibrated to this |

Graph model: `graph_windows.N_MAX` (32) hosts/window, `N_NODE_FEAT` (14).

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: No raw flow CSVs` | put `unified_*_labeled.csv` in the folder / fix `config.RAW_FLOW_CSVS` |
| `'Label' column missing` | run `extraction/label_mapping.ipynb` first — the world model needs labelled flows |
| `training split is empty` | wrong `SplitConfig`; the default `"stratified"` always yields all 3 splits |
| rollout probs all ≈ 0.4-0.5 | retrain — an old checkpoint predates the shared-head loss fix |
| `shap` / `xgboost` / `lightgbm` import errors | all optional; the zoo skips them, explain falls back. `cd research && pip install -e ".[benchmark]"` to enable |
| CUDA OOM (esp. GAT) | `--device cpu`, or lower `graph_windows.N_MAX` / GAT `d_model` |
| LightGBM "access violation" on Windows | already set `n_jobs=1`; if it still crashes it is skipped and the zoo continues |
| `ModuleNotFoundError: sentinel_wm` | run `pip install -e ./research`, or run notebooks from `research/notebooks/` (they self-bootstrap `sys.path`) |
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
