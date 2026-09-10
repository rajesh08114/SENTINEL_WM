# How to run the pipeline

Everything is in the code. Since the last full run the changes are:

* **Data-leakage fixes** (biggest): leakage-safe `stratified` split, span-boundary
  sequence purge, GAT/state scalers fit on the real train split, train-only
  winsorisation. Full list: [`docs/technical_reference.md`](docs/technical_reference.md) Part 1.9.
* **Per-attack-family metrics** + a **dual benchmark** (primary `stratified` +
  zero-shot `family`/`day` holdout).
* **Flow-level augmentation** (opt-in, train-only).
* Earlier: SSL encoder pre-training, distillation, snapshot + self ensembling,
  GRU encoder, the `SENTINEL-WM (system)` blend.

Because the split and the cleaning both changed, **every artifact must be
rebuilt** — including `clean_flows.parquet`.

## 1. Clear the stale derived artifacts

Keep only `data/`. Drop every derived artifact — **including `clean_flows.parquet`**
this time, because the leakage fixes changed the winsorisation (now fit on train
days only), the split (leakage-safe `stratified`), and sequence construction
(span-boundary purge). See `docs/technical_reference.md` Part 1.9.

**Git Bash / Linux / macOS**
```bash
cd /path/to/SIH
rm -f artifacts/clean_flows.parquet artifacts/clean_flows_aug.parquet \
      artifacts/state_windows.parquet artifacts/sequences.npz \
      artifacts/graph_windows.npz artifacts/state_scaler.pkl \
      artifacts/graph_node_scaler.pkl artifacts/world_model.pt
rm -rf research/models research/benchmarks
```

**PowerShell**
```powershell
cd C:\Users\chall\OneDrive\Desktop\SIH
Remove-Item artifacts\clean_flows.parquet, artifacts\clean_flows_aug.parquet, `
            artifacts\state_windows.parquet, artifacts\sequences.npz, `
            artifacts\graph_windows.npz, artifacts\state_scaler.pkl, `
            artifacts\graph_node_scaler.pkl, artifacts\world_model.pt -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force research\models, research\benchmarks -ErrorAction SilentlyContinue
```

> The leakage-free split yields ~6 % train / ~6 % val / ~3 % test positive
> **sequence** rate (the old ~11 % double-counted boundary sequences). Test
> positives are thin (~70–90); that is honest, not a regression.

## 2. Run the whole study

```bash
python -m sentinel_wm.research all --device cuda      # ~25-40 min on a GPU
#   CPU:            python -m sentinel_wm.research all --device cpu   (~2-3 h)
#   fast sanity:    python -m sentinel_wm.research all --quick        (~12 min, small epochs)
```

It runs: **data rebuild (flowaug if enabled) → profile → classical zoo →
world model (SSL pre-train → two-stage → distill → snapshots) → MLP/LSTM/GRU/TCN
→ GAT → benchmark (+ `SENTINEL-WM (system)` blend + per-family) → explainability
→ forward simulation → report**. Progress prints per phase; per-model lines look
like `[lstm] TEST F1=… AUROC=… …`. Watch for `[seq] leakage check OK … span-pure`.

If it dies partway, resume one stage:
`python -m sentinel_wm.research <flowaug|profile|baselines|worldmodel|nn|gat|benchmark|explain|simulate|report>`

## 3. Split modes & the two benchmarks

**Primary — `stratified`** (`SplitConfig.mode = "auto"` → `stratified`,
`research/`). Leakage-safe: benign windows get a contiguous per-day 60/20/20
backbone; a **whole attack episode** is assigned to one split, rotating per
family so families with ≥3 episodes span all 3 splits; only episodes ≥ 54
windows are cut internally. Any sequence whose `[t-L+1..t+K]` span crosses a
split boundary is dropped (`SequenceConfig.purge_boundary_sequences`), and
`build_sequences` asserts every retained sequence is span-pure. Single-burst
families (e.g. DoS GoldenEye) land in only 1–2 splits — a real dataset property,
shown by the per-family table.

**Secondary — zero-shot family/day holdout** (`research_zeroshot/`). Whole
attack families never appear in training:

```bash
python -m sentinel_wm.research all --split family --outdir research_zeroshot
# or  --split day   (Mon-Wed / Thu / Fri)
```

Then re-run the primary report so it folds in the holdout numbers:

```bash
python -m sentinel_wm.research report
```

`step_report` reads `research_zeroshot/benchmarks/benchmark.json` and appends a
"## 3. Zero-shot generalisation" section (F1 ≈ 0.3–0.5 there is expected — that
is the point).

Other modes: `block` (5-min interleave), `episode_chrono` (lead-time-focused),
`chronological`. Set `SplitConfig.mode` or pass `--split`.

### Optional: flow-level augmentation

`WindowConfig.flow_augment = True` (or `python -m sentinel_wm.research flowaug`)
synthesises extra **train-only** attack episodes (IAT jitter / flow dropout /
port shuffle / cross-day transplant) into `artifacts/clean_flows_aug.parquet`,
tagged `is_synthetic=1` and pinned to train — val/test stay byte-identical.
`build_state_windows` / `build_graph_windows` pick the file up automatically when
the flag is on. `research all` runs the `flowaug` step first when the flag is set.

## 4. What to check / send back for analysis

* `research/benchmarks/benchmark.md` ← ranked scoreboard + per-attack-family block
* `research/benchmarks/per_family.csv` ← full (model × family) matrix
* `research/data_profile/split_family_windows.csv` ← proves family coverage; only
  Heartbleed / Infiltration / SQL-Injection should be all-zero
* `research/data_profile/split_summary.csv` ← per-split window counts + attack frac
* `research/reports/RESEARCH_REPORT.md` and `technical_reference_addendum.md`
* the run log tail — look for `[seq] leakage check OK ... span-pure`,
  `[graph] node scaler fit on N/W real-train windows`, and the per-model
  `TEST …` lines
* if you ran the zero-shot benchmark: `research_zeroshot/benchmarks/benchmark.md`

Expect the leakage-free positive-sequence rate to read ~6 % train / ~6 % val /
~3 % test in `split_summary` — the pre-fix ~11 % double-counted boundary
sequences, so a lower headline F1 than the old GAT 0.69 / SENTINEL-WM 0.62 is
the *correct* number, not a regression.
