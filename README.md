# SENTINEL-WM

**S**patio-temporal **E**nemy **N**etwork **I**ntelligence with a **L**earned **World Model**

A fully-offline research prototype for **predictive cyber defence**. Instead of
labelling each network flow benign/malicious in isolation, SENTINEL-WM aggregates
traffic into **10-second network-state windows**, learns how those states *evolve*
(`P(S_{t+1} | S_{t-L:t})`), and **rolls the world forward K steps** to estimate
the probability — and the MITRE ATT&CK phase — of attacker progression **before
the kill chain completes**.

Built for the SIH problem *"AI systems that learn network behaviour, anticipate
attacker progression and support proactive cyber defence using World Models."*

> Full design rationale: [`docs/proposal.md`](docs/proposal.md) ·
> what was reviewed/changed vs. the proposal: [`docs/plan_validation.md`](docs/plan_validation.md) ·
> deep usage manual: [`docs/guide.md`](docs/guide.md) ·
> **why the metrics look the way they do, how to improve them, and every model /
> loss / optimizer / rollout / ATT&CK detail: [`docs/technical_reference.md`](docs/technical_reference.md)**

---

## Why a world model, not a classifier

| | Traditional ML IDS | SENTINEL-WM |
|---|---|---|
| Unit of analysis | one flow | evolving 10-s network-state window `S_t` (53 features) |
| Temporal model | none | learned state-transition dynamics (Bi-GRU + attention encoder + probabilistic STN) |
| Output | binary label | `P(attack)` for the next `k·10 s`, `k=1..6` + progression state + ATT&CK phase |
| Headline metric | F1 at prediction time | **PR-AUC / AUROC** (threshold-free) + **Mean Lead Time** + per-attack-family breakdown |
| Explainability | static feature importance | SHAP + attention saliency + gradient×input, per prediction |

Trained and benchmarked on **all five CIC-IDS-2017 days** (~2.8 M flows, 15
attack families) against a full model zoo — Logistic Regression, Random Forest,
Extra-Trees, HistGB, **XGBoost**, **LightGBM**, sklearn-MLP, kNN, GaussianNB,
LinearSVC, and neural **MLP / LSTM / GRU / TCN / GAT** — all on identical
sequences. The whole comparison regenerates with one command
(`python -m sentinel_wm.research all`) into the [`research/`](#the-research-folder)
folder. See [Results](#results).

---

## Architecture

```
 raw unified CSV  (flow + packet features, mapped labels)
        │  sentinel_wm/preprocessing.py     inf/NaN fix, label→family, time axis
        │                                   (rate-column winsor fit on train days only)
        ▼
 artifacts/clean_flows.parquet
        │  (optional) sentinel_wm/flow_augment.py   train-only synthetic attack
        │             episodes → artifacts/clean_flows_aug.parquet
        │  sentinel_wm/state_windows.py     10-s windows → S_t (53 dims: 41 base
        │                                   + 4 entropy + 8 first-differences)
        │                                   + progression state (episode logic)
        │                                   + ATT&CK phase/confidence  ◄── attack_stages.py
        ▼
 artifacts/state_windows.parquet
        │  sentinel_wm/sequences.py         [S_{t-11..t}] → (S_{t+1}, A_{t+1..6}, Z_{t+1..6})
        │                                   leakage-safe stratified split, span-boundary
        │                                   sequence purge, train-only RobustScaler
        ▼
 artifacts/sequences.npz  (+ state_scaler.pkl)
        │
        ├── baselines.py    LogReg/RF/ET/HGB/XGB/LGBM/… on S_t and on the flat sequence
        │
        └── models.py + train.py
              Bi-GRU + attention read-out encoder (elapsed-time channel)
              (Temporal Transformer still selectable via ModelConfig.encoder)
                     │  z_t
              ┌──────┼───────────────────────────────┐
              ▼      ▼                               ▼
        horizon    Probabilistic State-Transition    shared heads
        heads      Network:  z_t → μ,logσ            attack:   z → P(A=1)
        z_t→[K]    z_{t+1} ~ N(μ,σ)                  progress: z → P(Z=c)
        z_t→[K,5]         │
                          ▼  K-step Monte-Carlo rollout  (forward_sim.py)
              per-horizon: P(attack) + 95% CI, progression state, ATT&CK phase + confidence
                          │
                   explain.py (SHAP / attention / gradient)   evaluate.py (benchmark.md)
```

---

## Repository layout

```
.
├── README.md                     ← you are here
├── pyproject.toml                ← installable package definition
├── requirements.txt
├── .gitignore
│
├── sentinel_wm/                  ← the Python package
│   ├── config.py                   paths, hyper-parameters, LOCKED feature tiers
│   ├── preprocessing.py            PHASE 1  raw CSV → clean_flows.parquet
│   ├── flow_augment.py             (opt-in) train-only synthetic attack episodes
│   ├── state_windows.py            PHASE 2  flows → 10-s state windows + labels
│   ├── attack_stages.py            MITRE ATT&CK mapping (Layer A static + Layer B context)
│   ├── sequences.py                PHASE 2c windows → sequences.npz + leakage-safe split + scaler
│   ├── baselines.py                PHASE 3  classical model zoo (10 models × window/seq)
│   ├── models.py                   world model: Bi-GRU + attention encoder + probabilistic STN
│   ├── pretrain.py                 SSL masked-window encoder pre-training
│   ├── train.py                    PHASE 4  train / --test the world model
│   ├── nn_zoo.py                   neural baselines: MLP / LSTM / GRU / TCN
│   ├── nn_common.py                shared train/eval loop for every deep model
│   ├── graph_windows.py            per-window host-interaction graphs (for GAT)
│   ├── gat.py                      from-scratch Graph Attention Network baseline
│   ├── forward_sim.py              PHASE 5  K-step Monte-Carlo forward simulation
│   ├── explain.py                  PHASE 6  SHAP + attention + gradient saliency
│   ├── benchmark.py                unified scoreboard: every model, same test anchors
│   ├── registry.py                 one loader for every saved model (for the app)
│   ├── research.py                 orchestrator → the research/ folder
│   ├── evaluate.py                 back-compat shim → benchmark.py
│   ├── metrics.py                  shared metrics (F1/FPR/AUROC, Brier/ECE, Lead Time)
│   └── cli.py                      the offline command-line interface
│
├── extraction/                   ← raw feature extraction (standalone, needs tshark)
│   ├── extractor.py                from-scratch CICFlowMeter + packet features (PCAP → CSV)
│   └── label_mapping.ipynb         transfer official CIC-IDS-2017 labels onto the output
│
├── notebooks/
│   ├── 00_complete_pipeline.ipynb       one notebook, every phase end to end
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_sequence_generation.ipynb     state windows + ATT&CK mapping + split
│   ├── 03_model_zoo_and_benchmark.ipynb train & compare every model
│   ├── 04_explainability.ipynb          SHAP / attention / gradient
│   └── 05_forward_simulation.ipynb      K-step rollouts + ATT&CK stage forecast
│
├── docs/
│   ├── proposal.md                 the SENTINEL-WM v2.0 design document
│   ├── plan_validation.md          review verdict + the corrections made
│   ├── guide.md                    full usage manual, config cheatsheet, troubleshooting
│   └── problem_statement.pdf
│
├── data/                         ← input CSVs (gitignored; see data/README.md)
│   ├── README.md
│   └── unified_AllDays_labeled.csv   all 5 CIC-IDS-2017 days
│
├── research/                     ← full study output, each number backed by a file (gitignored)
│   ├── reports/RESEARCH_REPORT.md
│   ├── data_profile/  models/  benchmarks/  figures/  explainability/  simulations/  logs/
│   └── models/registry.json      uniform index of every trained model
│
└── artifacts/                    ← pipeline working files (gitignored)
    ├── clean_flows.parquet  state_windows.parquet  sequences.npz  state_scaler.pkl
    ├── graph_windows.npz    world_model.pt         baselines/
    └── reports/  benchmark.md  world_model_metrics.json  forward_sim_test.json  explainability.json
```

---

## Install

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # Linux/macOS

pip install -e .                  # core (numpy, pandas, torch, sklearn, matplotlib)
pip install -e ".[benchmark]"     # + xgboost, lightgbm, shap  (the full model zoo)
pip install -e ".[notebooks]"     # + jupyter/nbconvert to run the notebooks
```

`xgboost` / `lightgbm` / `shap` are **optional** — the zoo skips them and
explainability falls back to gradient×input if they are absent. The **GAT**
baseline is implemented from scratch, so **no `torch-geometric`** is needed.

Python 3.10+; a CUDA GPU is auto-detected but not required. No internet access is
used at any stage.

The repo ships with `data/unified_AllDays_labeled.csv` (all 5 CIC-IDS-2017 days),
so the whole pipeline runs immediately.

---

## Quickstart

```bash
# the whole study: data → all models → benchmark → explain → simulate → research/
python -m sentinel_wm.research all            # ~30-45 min on a GPU
python -m sentinel_wm.research all --quick    # small epoch budgets, ~10 min

# or the lighter phase runner (adds the model zoo, ends at the benchmark)
python -m sentinel_wm.cli all
```

Run pieces individually:

| Command | Does | Output |
|---|---|---|
| `python -m sentinel_wm.cli preprocess` | PHASE 1 | `artifacts/clean_flows.parquet` |
| `python -m sentinel_wm.cli windows` | PHASE 2 | `artifacts/state_windows.parquet` |
| `python -m sentinel_wm.cli sequences` | PHASE 2c | `artifacts/sequences.npz`, `state_scaler.pkl` |
| `python -m sentinel_wm.cli baseline` | PHASE 3 | classical zoo → `research/models/classical/` |
| `python -m sentinel_wm.cli train --epochs 40` | PHASE 4 | `artifacts/world_model.pt` |
| `python -m sentinel_wm.cli nn --kinds mlp lstm gru tcn` | 4b | `research/models/nn/*.pt` |
| `python -m sentinel_wm.cli graphwindows` | 4c | `artifacts/graph_windows.npz` |
| `python -m sentinel_wm.cli gat` | 4c | `research/models/nn/gat.pt` |
| `python -m sentinel_wm.cli simulate --split test --explain` | PHASE 5 | `artifacts/reports/forward_sim_test.json` |
| `python -m sentinel_wm.cli explain` | PHASE 6 | `artifacts/reports/explainability.json` |
| `python -m sentinel_wm.cli benchmark` | — | `research/benchmarks/benchmark.md` + `registry.json` |

`python -m sentinel_wm.research <step>` (`profile`, `baselines`, `worldmodel`,
`nn`, `gat`, `benchmark`, `explain`, `simulate`, `report`) reruns just one stage.
`python -m sentinel_wm.train --test` re-scores an existing checkpoint.

Or open the notebooks: **`00_complete_pipeline.ipynb`** is the one-notebook tour;
`01`–`05` are the phase deep-dives.

---

## The MITRE ATT&CK mapping (`sentinel_wm/attack_stages.py`)

CIC-IDS-2017 labels an attack *type per flow*, never a *tactic per moment* — so
the mapping is a deterministic, auditable **two-layer** function applied *after*
the model, never a training target:

* **Layer A — `CIC_LABEL_TO_ATTACK`**: static, literature-based
  `attack_family → (tactic, technique IDs, base confidence)`
  (e.g. `DoS Hulk → Impact / T1499,T1499.002 / High`;
  `PortScan → Reconnaissance / T1595,T1046 / High`).
* **Layer B — `assess_window(state, family, prev_family, ratio)`**: resolves one
  window to a single phase and **lowers confidence when evidence is thin**
  (`PRE_ATTACK` → Recon, Low unless scan-shaped; `ONSET` → tactic downgraded one
  notch; `ACTIVE` → full; `CONTINUATION` → emits a `family_transition` if the
  family changed). `assess_forecast(...)` additionally drops confidence one notch
  per 3 horizon steps.

Every output carries `confidence ∈ {High, Medium, Low}` and a plain-text
`rationale`. `python -m sentinel_wm.attack_stages` runs an 8-case self-test.

---

## Results

The benchmark regenerates on every run — the **authoritative, always-current**
numbers live in
[`research/benchmarks/benchmark.md`](research/benchmarks/benchmark.md)
(ranked table + per-horizon F1 + per-attack-family breakdown),
[`research/reports/RESEARCH_REPORT.md`](research/reports/RESEARCH_REPORT.md)
(narrative, every claim linked to a file), and
[`research/reports/technical_reference_addendum.md`](research/reports/technical_reference_addendum.md)
(live hard numbers: F, L, K, param counts, per-split / per-family counts).

**Setup (primary benchmark).** All 5 CIC-IDS-2017 days, **leakage-safe
`stratified` split** (contiguous per-day benign backbone + whole attack episodes
assigned to one split, rotating per family; sequences whose history+horizon span
crosses a split boundary are purged — see
[`docs/technical_reference.md`](docs/technical_reference.md) Part 1.9). Ranked by
**PR-AUC** (threshold-free; robust to the val→test prevalence shift). The
leakage-free positive-sequence rate is ~6 % train / ~6 % val / ~3 % test.

**What to expect.** The `__seq` gradient boosters (XGBoost / RandomForest /
HistGB on the flattened `L·F` sequence) and the from-scratch **GAT** are the
strong floor. **`SENTINEL-WM (system)`** — the world-model self-ensemble
(direct head + K-step rollout + snapshots) val-blended with those members — tops
the board on PR-AUC; raw **`SENTINEL-WM`** is just behind and is the only model
that also carries a **progression-state head**, a **calibrated K-step
Monte-Carlo rollout with ATT&CK phase + confidence**, and per-family
explainability. Classical models roughly double from window-only to `__seq`
input. Rare families (Heartbleed / Infiltration / SQL-Injection: 11–36 flows
total) produce no learnable windows and are reported as excluded, not failures.

`python -m sentinel_wm.research report` writes the narrative with every claim
linked to its CSV/JSON/PNG.

### Split regimes — two benchmarks

| `SplitConfig.mode` | what it tests | how to run |
|---|---|---|
| `stratified` (default) | forecast progression of **known** families, leakage-free; every family with ≥3 episodes spans all 3 splits | `python -m sentinel_wm.research all` → `research/` |
| `family` / `day` | **zero-shot to a novel attack family** (whole families held out of training) | `python -m sentinel_wm.research all --split family --outdir research_zeroshot` → `research_zeroshot/`, then `research report` folds a "Zero-shot generalisation" section into the primary report |
| `block` / `episode_chrono` / `chronological` | secondary / lead-time-focused variants | set `SplitConfig.mode` or pass `--split` |

The `family` / `day` splits are zero-shot: whole attack families never appear in
training, so absolute F1 drops to ~0.3–0.5 for *every* model — that is the point,
and it is reported as a secondary generalisation benchmark, not the headline.

---

## The research/ folder

`python -m sentinel_wm.research all` populates a self-contained study directory —
**every number in the report is backed by a file**:

| path | contents |
|---|---|
| `research/reports/RESEARCH_REPORT.md` | narrative, each claim linked to proof |
| `research/data_profile/` | label & timeline stats, split balance |
| `research/models/` | every trained model — `classical/*.pkl`, `nn/*.pt`, `world_model.pt` — + `registry.json` |
| `research/benchmarks/` | `benchmark_full.csv` / `.md` / `.json`, `per_horizon_f1.csv`, `per_family.csv`, `leadtime.csv` |
| `research/data_profile/split_family_windows.csv` | attack windows per (family, split) — proof of family coverage |
| `research/figures/` | horizon-F1, ROC, PR, lead-time, SHAP, rollout-timeline PNGs |
| `research/explainability/` | SHAP / attention / gradient attribution JSON |
| `research/simulations/` | K-step forward-simulation runs + `attck_stage_forecast.csv` |
| `research/logs/` | per-model training curves |

### Using the saved models (for the FastAPI/Next.js app)

```python
from sentinel_wm.registry import list_models, load_predictor
list_models()                       # every trained model
p = load_predictor("SENTINEL-WM")   # uniform wrapper
out = p.predict(X, dt)              # X [N,L,F] scaled, dt [N,L] log1p
out["attack_prob_k"]     # [N, K]   P(attack) per horizon
out["progression_k"]     # [N, K]   progression-state index (or None)
```

---

## Configuration

All knobs live in `sentinel_wm/config.py` (dataclasses: `WindowConfig`,
`SequenceConfig`, `SplitConfig`, `ModelConfig`, `TrainConfig`). Common ones:
window size (10 s), stride, history `L` (12), horizon `K` (6),
`SplitConfig.mode` (`auto`→`stratified`), `SequenceConfig.purge_boundary_sequences`
(leakage guard), `WindowConfig.flow_augment`, `ModelConfig.encoder` (`gru`),
joint-loss weights, MC samples (50), target FPR (5 %).
Full table in [`docs/guide.md`](docs/guide.md#5-configuration-cheatsheet-configpy).

---

## License

MIT (see `pyproject.toml`). CIC-IDS-2017, CTU-13 and UNSW-NB15 are the property
of their respective publishers and are **not** redistributed here — download them
from the original sources. This is a research prototype, not a production IDS.
