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
> deep usage manual: [`docs/guide.md`](docs/guide.md)

---

## Why a world model, not a classifier

| | Traditional ML IDS | SENTINEL-WM |
|---|---|---|
| Unit of analysis | one flow | evolving 10-s network-state window `S_t` |
| Temporal model | none | learned state-transition dynamics (Temporal Transformer + probabilistic STN) |
| Output | binary label | `P(attack)` for the next `k·10 s`, `k=1..6` + progression state + ATT&CK phase |
| Headline metric | F1 at prediction time | **Mean Lead Time** — how much time the defender gets |
| Explainability | static feature importance | SHAP + attention saliency + gradient×input, per prediction |

On the shipped single day (CIC-IDS-2017 Wednesday, DoS): the world model gives
**30 s mean lead time** and **holds F1 across the 60 s horizon (0.80 → 0.73)**,
where logistic-regression / random-forest baselines on the same features give
~0 s lead time. See [Results](#results).

---

## Architecture

```
 raw unified CSV  (flow + packet features, mapped labels)
        │  sentinel_wm/preprocessing.py     inf/NaN fix, label→family, time axis
        ▼
 artifacts/clean_flows.parquet
        │  sentinel_wm/state_windows.py     10-s windows → S_t (41 dims)
        │                                   + progression state (episode logic)
        │                                   + ATT&CK phase/confidence  ◄── attack_stages.py
        ▼
 artifacts/state_windows.parquet
        │  sentinel_wm/sequences.py         [S_{t-9..t}] → (S_{t+1}, A_{t+1..6}, Z_{t+1..6})
        │                                   leakage-safe split, train-only RobustScaler
        ▼
 artifacts/sequences.npz  (+ state_scaler.pkl)
        │
        ├── baselines.py    LogisticRegression / RandomForest on S_t only  (comparison floor)
        │
        └── models.py + train.py
              Temporal Transformer encoder (elapsed-time pos-enc, causal mask, attn kept)
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
│   ├── state_windows.py            PHASE 2  flows → 10-s state windows + labels
│   ├── attack_stages.py            MITRE ATT&CK mapping (Layer A static + Layer B context)
│   ├── sequences.py                PHASE 2c windows → sequences.npz + split + scaler
│   ├── baselines.py                PHASE 3  LogReg / RandomForest comparison floor
│   ├── models.py                   Temporal Transformer + probabilistic STN + heads
│   ├── train.py                    PHASE 4  train / --test the world model
│   ├── forward_sim.py              PHASE 5  K-step Monte-Carlo forward simulation
│   ├── explain.py                  PHASE 6  SHAP (or fallback) + attention + saliency
│   ├── evaluate.py                 benchmark table: world model vs baselines
│   ├── metrics.py                  shared metrics (F1/FPR/AUROC, Brier/ECE, Lead Time)
│   └── cli.py                      the offline command-line interface
│
├── extraction/                   ← raw feature extraction (standalone, needs tshark)
│   ├── extractor.py                from-scratch CICFlowMeter + packet features (PCAP → CSV)
│   └── label_mapping.ipynb         transfer official CIC-IDS-2017 labels onto the output
│
├── notebooks/                    ← step-by-step walkthroughs (run top to bottom)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_sequence_generation.ipynb
│   └── 03_model_training.ipynb
│
├── docs/
│   ├── proposal.md                 the SENTINEL-WM v2.0 design document
│   ├── plan_validation.md          review verdict + the 7 corrections made
│   ├── guide.md                    full usage manual, config cheatsheet, troubleshooting
│   └── problem_statement.pdf
│
├── data/                         ← input CSVs (gitignored; see data/README.md)
│   └── README.md
│
└── artifacts/                    ← all generated outputs (gitignored)
    ├── clean_flows.parquet  state_windows.parquet  sequences.npz  state_scaler.pkl
    ├── world_model.pt       baselines/
    └── reports/  benchmark.md  world_model_metrics.json  forward_sim_test.json  explainability.json
```

---

## Install

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # Linux/macOS

pip install -e .                # core package (numpy, pandas, torch, sklearn, matplotlib)
pip install -e ".[notebooks]"   # + jupyter/nbconvert to run the notebooks
pip install -e ".[explain]"     # + shap (optional; a fallback runs without it)
```

Python 3.10+; a CUDA GPU is auto-detected but not required (40 epochs ≈ 10 s on
GPU, ≈ 1 min on CPU for the single-day dataset). No internet access is used at
any stage.

The repo ships with `data/unified_Wednesday-WorkingHours_labeled.csv` already
built, so you can run the whole pipeline immediately.

---

## Quickstart

```bash
# phases 1-6 end to end, then print the benchmark
python -m sentinel_wm.cli all --epochs 40
#   or, after `pip install -e .`:
sentinel-wm all --epochs 40
```

Run phases individually:

| Command | Phase | Output |
|---|---|---|
| `python -m sentinel_wm.cli preprocess` | 1 | `artifacts/clean_flows.parquet` |
| `python -m sentinel_wm.cli windows` | 2 | `artifacts/state_windows.parquet` |
| `python -m sentinel_wm.cli sequences` | 2c | `artifacts/sequences.npz`, `state_scaler.pkl` |
| `python -m sentinel_wm.cli baseline` | 3 | `artifacts/baselines/baseline_metrics.json` |
| `python -m sentinel_wm.cli train --epochs 40` | 4 | `artifacts/world_model.pt` |
| `python -m sentinel_wm.cli simulate --split test --explain` | 5 | `artifacts/reports/forward_sim_test.json` |
| `python -m sentinel_wm.cli explain` | 6 | `artifacts/reports/explainability.json` |
| `python -m sentinel_wm.cli evaluate` | — | `artifacts/reports/benchmark.md` |
| `python -m sentinel_wm.cli demo` | — | full pipeline + printed timeline |

`python -m sentinel_wm.train --test` re-evaluates an existing checkpoint.
Every module is also runnable directly, e.g. `python -m sentinel_wm.state_windows`.

Or open the notebooks in order — `notebooks/01_ → 02_ → 03_` — each drives one
stage with inline plots and explanations.

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

Single CIC-IDS-2017 Wednesday (DoS) day, block-interleaved split, FPR-calibrated
threshold — from `artifacts/reports/benchmark.md`:

| Model | F1 (any-k) | AUROC | **Mean Lead Time** | **Episodes warned** | FA rate |
|---|---|---|---|---|---|
| Logistic Regression (`S_t` only) | 0.75 | 0.92 | 10 s | 1 / 7 | 0.03 |
| Random Forest (`S_t` only) | 0.82 | 0.97 | 0 s | 0 / 7 | 0.01 |
| Persistence (`A_{t+k}=A_t`) | 0.84 | 0.87 | 0 s | 0 / 7 | 0.00 |
| **SENTINEL-WM** | 0.80 | 0.95 | **30 s** (max 60) | **4 / 7** | 0.05 |

Forecast-horizon F1: 0.80 → 0.73 from +10 s to +60 s (baselines only look strong
at long horizons *because DoS persists*, not because they forecast onset).
Progression-state accuracy 0.89; Brier(k1) 0.06; ECE(k1) 0.08; 0.77 M parameters.

**Known limitation:** Wednesday is DoS-only with abrupt onsets, so lead time is
inherently capped. The proposal's headline **Infiltration** scenario (Thursday:
external recon → compromise → internal Nmap) is the one that shows large lead
time — add the Thursday/Friday files (see below) to demonstrate it.

---

## Adding more CIC-IDS-2017 days

1. Extract each day with `extraction/extractor.py` + `extraction/label_mapping.ipynb`
   → `data/unified_<Day>-WorkingHours_labeled.csv` (details in
   [`data/README.md`](data/README.md)).
2. Append the path to `RAW_FLOW_CSVS` in `sentinel_wm/config.py`.
3. Re-run `python -m sentinel_wm.cli all`.

`SplitConfig.mode="auto"` then switches from the single-day block-interleaved
split to the proposal's proper **day-based** split (Mon-Wed / Thu / Fri)
automatically — no code change.

---

## Configuration

All knobs live in `sentinel_wm/config.py` (dataclasses: `WindowConfig`,
`SequenceConfig`, `SplitConfig`, `ModelConfig`, `TrainConfig`). Common ones:
window size (10 s), stride, history `L` (10), horizon `K` (6), split mode,
transformer size, joint-loss weights, MC samples (50), target FPR (5 %).
Full table in [`docs/guide.md`](docs/guide.md#5-configuration-cheatsheet-configpy).

---

## License

MIT (see `pyproject.toml`). CIC-IDS-2017, CTU-13 and UNSW-NB15 are the property
of their respective publishers and are **not** redistributed here — download them
from the original sources. This is a research prototype, not a production IDS.
