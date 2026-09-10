# SENTINEL-WM — Technical Reference

Deep detail on **why the metrics are what they are**, **how to push them up**, and
**exactly how every model, the training loop, the forward simulation, and the
ATT&CK phase logic work**. Read alongside [`guide.md`](guide.md) (how to run) and
[`plan_validation.md`](plan_validation.md) (design decisions).

---

## Part 1 — Why the metrics look "low" (F1 ≈ 0.55–0.69, AUROC ≈ 0.85–0.91)

The headline is **not** a weak model. It is the combination of *what task we are
scoring* and *what the CIC-IDS-2017 test data actually contains*. Every claim
below is checked against the pipeline output.

### 1.1 We score **forecasting**, not per-flow classification

Papers that report F1 ≈ 0.99 on CIC-IDS-2017 do **per-flow, nowcast, random-split**
classification: "is *this* flow malicious right now", with attack and benign rows
from the same session on both sides of the split. That is a different, much
easier problem.

SENTINEL-WM reports:

* **`any_horizon` F1** — did the model predict an attack in **any** of the next
  `t+1 … t+6` windows (i.e. up to **60 s ahead**), from a 10-window history.
* **per-horizon F1** — accuracy at `+10 s, +20 s, … +60 s` separately.

Forecasting 60 s ahead from aggregated state is intrinsically lossy. The
per-horizon table shows the honest decay (world model 0.669 → 0.612 across the
horizon; XGBoost 0.767 → 0.713). AUROC ≈ 0.90 says the *ranking* is good; F1 at a
strict operating point is what looks modest.

### 1.2 Window-level aggregation **dilutes the signal** — the dominant cause

Measured on `artifacts/state_windows.parquet`:

| quantity | value | consequence |
|---|---|---|
| median flows per 10-s window | **77** | aggregates computed over few samples → noisy |
| median `attack_ratio` in windows labelled *attack* | **0.109** | the median positive window is **~11 % malicious flows, ~89 % benign** |
| attack windows with `attack_ratio < 0.10` | **47.6 %** | nearly half the positive labels are a <10 % perturbation of aggregate rates |

A per-flow classifier sees DoS Hulk as a blatant outlier. Our model sees a window
whose `syn_rate` / `byte_rate` moved ~10 % because 8 of 77 flows were malicious.
That is the difference between F1 ≈ 0.99 and F1 ≈ 0.6. It is inherent to modelling
*network state* rather than *individual flows* — which is the entire point of a
world model, but it costs raw F1.

### 1.3 The test split is dominated by the **hardest** attack family

Friday (the block-split test day) attack windows by dominant family:

| family | attack windows | why it is hard at 10-s window scale |
|---|---|---|
| **Bot (ARES)** | **484** | low-and-slow C2 beaconing — a handful of small periodic flows per window; almost indistinguishable from benign after aggregation |
| DDoS (LOIC) | 119 | easy (volumetric) |
| PortScan | 97 | spread thin; often < 10 % of a window's flows |

**69 % of the positives the model is graded on are Botnet windows** that are
genuinely near-invisible at this resolution. Metrics on DDoS-only windows are
much higher; the number is pulled down by Botnet + the rare-family tail
(Infiltration = 36 flows in the *entire dataset*, Heartbleed = 11).

### 1.4 Operating point: FPR ≤ 5 %

The alert threshold is calibrated so False-Positive-Rate ≤ 5 % on validation
(`metrics.calibrate_threshold`, `TrainConfig.target_fpr`). That is an
operationally sane constraint (a SOC cannot absorb more), but it deliberately
sacrifices recall → F1. At FPR = 15 % every model's F1 rises ~8–12 points; that
is not a useful system.

### 1.5 Label noise

* CIC-IDS-2017 labels are known to be imperfect (Rosay et al., ICISSP 2022):
  benign flows mislabelled during attack hours, ± seconds of timing offset.
* Our episode/progression labelling (`state_windows._derive_progression`) uses
  `min_attack_flows = 1`, `min_attack_ratio = 0.0` — **one** stray attack-labelled
  flow flips a window to positive. Combined with 1.2 this injects a floor of
  irreducible error.

### 1.6 Mean Lead Time ≈ 10 s for *every* model (incl. persistence)

Not a model failure — a **split artefact**. Block-interleaving cuts each 30–60 min
attack episode into 15-min blocks, so test windows land *mid-episode*, not at
onset. There is almost nothing to "warn early" about. `metrics.lead_time` also
caps the look-back at `K` windows. The `day` split *does* contain onsets, but is
zero-shot to new families (§Part 1.7). A *chronological-within-attack-day* split
(train on each episode's ramp-up, test on its continuation) would restore the
lead-time signal — see Part 2.

### 1.7 The `day` split is even lower — and that is correct

Train = Mon-Wed (DoS, Patator), val = Thu (Web, Infiltration), test = Fri (Botnet,
PortScan, DDoS). Three **disjoint** attack regimes. Every model — LogReg, XGBoost,
GAT, world model — drops to F1 ≈ 0.30–0.40 because **no model trained only on DoS
can nowcast a port scan it has never seen**. That is a property of the split
(zero-shot generalisation), reported as a stress test, not the headline.

### 1.8 Why the world model is *mid-pack* on raw F1 specifically

| factor | effect |
|---|---|
| joint loss shares capacity across 4 objectives (attack + progression + next-state MSE + KL) | a pure attack classifier of the same size scores higher *on attack F1* — but loses the progression head + rollout |
| early-stop metric = `val_f1 + 0.5·val_prog_acc` | the checkpoint is a compromise, not the attack-F1 argmax |
| world model early-stops at ~epoch 17–24 | undertrained vs convergence |
| linear→MLP horizon heads (fixed) but still no HP search | `d_model`, `L`, `K`, `window_seconds`, dropout are reasonable guesses, untuned |

The world model's value is not raw F1: it is the **only** model with a
progression-state head (0.854 acc), the highest episode-detection rate (0.22), and
a **calibrated K-step Monte-Carlo rollout with ATT&CK phase + confidence**.

### 1.9 Data-leakage controls (what stops the numbers being fake)

| vector | control |
|---|---|
| **sequence span straddling a split boundary** — a train anchor's horizon `[t+1..t+K]` reaching into val/test means it trains on val/test *labels*; a val/test anchor scored on a window a train anchor already forecast | `SequenceConfig.purge_boundary_sequences` (default **True**): any sequence whose window span `[t-L+1 .. t+K]` is not entirely one split is dropped (`split="ignore"`). `build_sequences` then **asserts** every retained sequence is span-pure. Typical cost: ~1.9k sequences → `ignore`. |
| **half-episode splits** make almost every in-episode sequence straddle a cut | `stratified` assigns a **whole episode** to one split (rotating per family); only episodes ≥ `3*(L+K)` are cut internally. |
| **benign backbone with many boundaries** (5-min interleaved blocks) | `stratified_benign="contiguous"` (default): per-day 60/20/20 by time → 2 internal boundaries/day, not dozens. |
| **RobustScaler (state) fit** | `sequences.build_sequences` fits on `split=="train" & is_synthetic==0` sequences only — which, post-purge, are span-pure train. |
| **RobustScaler (GAT node features) fit** | `graph_windows` now fits on the **real per-window `assign_split` result**, not `DAY_SCHEDULE` day names (under `stratified`, Wednesday holds val/test windows too). |
| **global winsorisation** of the 4 unbounded rate columns | `preprocessing.clean_flow_frame` computes the 99.9-pct clip on `DAY_SCHEDULE` train days only. |
| **flow-augment benign transplant** | `flow_augment._benign_pad` samples benign context only from `_split=="train"` windows of other days. |
| threshold / system-blend weight | calibrated / tuned on **val**, frozen for the test report. |
| SSL pre-training, distillation teachers, class weights, balanced sampler | all fit on the **train** split only. |
| residual (accepted) | `label_smooth_windows=1` majority-vote is applied per day before splits are known, so ~1 window per split boundary can carry a label smoothed across it (≈10 windows dataset-wide). Set `label_smooth_windows=0` for zero residual, or `stratified_purge_windows≥1`. |

Effect of removing the leakage: the effective **positive-sequence rate drops from
~11 % to ~6 % train / ~6 % val / ~3 % test** — the ~11 % was partly boundary
sequences double-counted across splits. Test positives are genuinely thin
(~70–90 sequences); `flow_augment=True` adds *train* positives only, and
`per_family_scores` auto-excludes families below `min_support=8`.

---

## Part 2 — How to improve the metrics (ranked by expected impact)

> **Status (this build).** Items 1, 2, 5, 6, 7, 9, 10, 14 are **implemented and
> on by default**; item 8 (flow-level augmentation) is **implemented, opt-in**
> (`WindowConfig.flow_augment`). Re-run `python -m sentinel_wm.research all`
> to pick them up. The rest are follow-ups.
>
> Live hard numbers (F, L, K, d_model, param count, split mode, per-split /
> per-family window counts) are regenerated every run into
> **`research/reports/technical_reference_addendum.md`** — trust that over any
> number hard-coded below.
>
> * **split is now `stratified`** (`SplitConfig.mode = "auto"` → `stratified`,
>   item 14 superseded), **leakage-safe** (see Part 1.9): benign windows get a
>   contiguous per-day 60/20/20 backbone; a **whole attack episode** goes to one
>   split, rotating per family so families with ≥3 episodes span all 3; only
>   episodes ≥ `3*(L+K)` are cut internally. Sequences whose span crosses a split
>   boundary are purged (`SequenceConfig.purge_boundary_sequences`). `block`
>   (5-min) is kept as a secondary benchmark; `day` / `family` are the zero-shot
>   holdouts, run into `research_zeroshot/` via
>   `research all --split family --outdir research_zeroshot` and surfaced as the
>   report's "## 3. Zero-shot generalisation" section.
> * **per-attack-family metrics** — `metrics.per_family_scores`, written to
>   `research/benchmarks/per_family.csv` and a `benchmark.md` block; families with
>   < 8 positive test windows (Heartbleed / Infiltration / SQLi) are reported as
>   excluded, not as failures.
> * **flow-level augmentation** (`flow_augment.py`, item 8) — synthesises extra
>   **train-only** attack episodes (±15 % IAT jitter, 20 % flow dropout, dst-port
>   shuffle, cross-day transplant of rare families) into
>   `artifacts/clean_flows_aug.parquet`, tagged `is_synthetic=1` on `__aug_<fam>__`
>   days that `assign_split` pins to train. Val/test stay byte-identical to a
>   `flow_augment=False` run (train-only proof).
>
> What changed in code:
> * label tightened — `WindowConfig.min_attack_ratio = 0.05`, `min_attack_flows = 2`,
>   `label_smooth_windows = 1` (majority-vote smoothing). Positive-window rate
>   0.152 → 0.108; removes the noise-level windows.
> * 12 new features — 4 distribution-entropy (`dstport/dstip/srcip/flowsize`) +
>   8 first-differences (`d_packet_rate`, `d_syn_rate`, …). `F` 41 → 53.
> * `SequenceConfig.history` 10 → 12; `ModelConfig.d_model` 128 → 160.
> * **focal loss** (`ModelConfig.focal_gamma = 1.0`) on every attack head.
> * **balanced batch sampler** (`TrainConfig.balanced_sampler_min_pos = 0.30`).
> * **two-stage world-model training** (`TrainConfig.two_stage = True`): 35 % of
>   the epoch budget trains encoder + attack head alone, then the STN /
>   progression head are added.
> * **cosine warm restarts** (`warm_restart_period = 40`), `epochs` 70 → 120,
>   patience 14 → 22.
> * **probability ensemble** row in the benchmark (mean `attack_prob_k` of
>   `SENTINEL-WM + gat + xgboost__seq + random_forest__seq`).
> * benchmark now reports **PR-AUC** and **F1\*** (best F1 over all thresholds)
>   and **ranks by PR-AUC** — threshold-free, robust to the val→test attack-
>   prevalence shift that made the fixed-FPR F1 noisy.
> * new split `SplitConfig.mode = "episode_chrono"` — keeps real onsets in
>   val/test so Mean Lead Time is a live metric again (now also sprinkles benign
>   windows into val/test so FPR stays measurable).
> * `metrics.lead_time` bug fixed (window-index contiguity + horizon cap).
> * **split family-coverage bug fixed** — `block_minutes` 15 → 5. At 15 min a
>   short attack episode (e.g. DoS Hulk, ~24 min on Wednesday) fell entirely
>   inside blocks that all went to one split: **DoS Hulk was 0 train / 82 val /
>   24 test windows**, so the model was tested on a family it never trained on.
>   At 5 min every family appears in the train set (1 empty family×split cell of
>   36, vs 8+ before).
> * **`augment.py`** — train-only sequence augmentation, ON by default
>   (`TrainConfig.augment`): per-feature jitter, positive↔positive mixup,
>   window dropout (collection-gap sim), small temporal roll. Applied on the
>   scaled `sequences.npz` in the DataLoader (not the GAT graph path).
> * **world-model encoder swapped Transformer → Bi-GRU + attention read-out**
>   (`ModelConfig.encoder = "gru"`). On ~8.5k sequences of 12 short steps the
>   causal Transformer ceilings at val AUROC ~0.90 and plateaus by epoch ~13
>   (the curve is in `training_curve.csv`); a Bi-GRU reaches ~0.94 with **half
>   the parameters** (532k vs 1.23M). The Transformer stays available for when
>   data ≫ 10k. Loss re-weighted `w_attack 1.5→3.0`, `w_next_state 0.25→0.10`,
>   `w_kl 1e-4→1e-5` so the benchmarked attack head is the clear priority.
>   Two-stage: stage A shortened (0.35→0.20 of the budget), and stage B now
>   *starts from stage A's best-val checkpoint* (not its overfit tail).
> * **getting to #1** (all `TrainConfig` flags, ON by default):
>   * **SSL pre-training** (`pretrain.py`) — masked-window reconstruction on all
>     ~8.5k train sequences; encoder weights loaded into stage A.
>   * **distillation** (`distill`) — decaying soft-target BCE from the strong
>     `__seq` boosters' train probs.
>   * **snapshot ensembling** — a checkpoint at every cosine warm-restart trough
>     (period 30, patience 40 so it survives a dip), averaged at inference.
>   * **self-ensemble** — the benchmarked `SENTINEL-WM` prob is
>     `0.6·direct-head + 0.4·K-step-rollout`, averaged over the snapshots.
>   * **`SENTINEL-WM (system)` row** — the WM self-ensemble blended (val-tuned
>     weight) with `{xgboost__seq, random_forest__seq, hist_gradient_boosting__seq,
>     gat}`. This is the deployed artifact (`registry.load_system_predictor()`);
>     a plain unweighted mean of the same members is strictly weaker and is not
>     shown as a separate row.
>   **Effect** (100-epoch smoke, 2 snapshots): **`SENTINEL-WM (system)` #1**
>   (PR-AUC 0.883, AUROC 0.954, F1\* 0.809, ProgAcc 0.913); **raw `SENTINEL-WM`
>   #2** (PR-AUC 0.853, AUROC 0.942) — both ahead of every classical / nn / graph
>   model. The full 150-epoch run adds 2-3 more snapshots.

| # | change | where | expected Δ F1 | effort |
|---|---|---|---|---|
| 1 | **Tighten the positive label.** `min_attack_ratio ≥ 0.15` (or scale `min_attack_flows` with `flow_count`); optionally a 3-window majority smooth. Removes the ~48 % of near-benign "attack" windows that are unlearnable. | `state_windows._derive_progression`, `WindowConfig` | **+4–8** | low |
| 2 | **Report PR-AUC / AUROC as headline**, F1 at a *fixed recall* (e.g. R = 0.8) as secondary. AUROC is already ~0.90; F1 at FPR≤5 % structurally understates the models. | `metrics.py`, `benchmark.py` | reframing (+0 real, +honest) | low |
| 3 | **Hyperparameter search** (Optuna, val-scored): `window_seconds ∈ {5,10,30}`, `L ∈ {8,12,20}`, `d_model`, `lr`, `dropout`, loss weights, `episode_gap_windows`. | new `sentinel_wm/tune.py` | **+5–10** | medium |
| 4 | **Two-stage world-model training.** Stage A: encoder + attack head only, to convergence. Stage B: freeze encoder, attach STN + progression head. Removes gradient dilution. | `train.py` | **+3–6** (attack F1) | medium |
| 5 | **Probability ensemble** of `gat + xgboost__seq + SENTINEL-WM` (mean of `attack_prob_k`). Nearly free; models are decorrelated (graph / tree / transformer). | `benchmark.py` add an `ensemble` row | **+3–5** | low |
| 6 | **Focal loss** (`γ=2`) in place of weighted BCE for the attack head — down-weights the easy majority. | `models.joint_loss`, `nn_common._loss` | **+2–4** | low |
| 7 | **Class-balanced batch sampler** — guarantee ≥ 30 % positive windows per batch (proposal §7.4, not yet built). | `train.py`, `nn_common.py` DataLoader | **+2–4** | low |
| 8 ✅ | **Flow-level augmentation** (proposal §7.5): dst-port shuffle, ±15 % IAT jitter, 20 % flow dropout + re-aggregation, cross-day transplant of rare families — synthesises new TRAIN attack episodes at the flow level so aggregation makes genuinely new positive windows. Opt-in `WindowConfig.flow_augment`; provably train-only. | `sentinel_wm/flow_augment.py` | **+2–6** on mid-count families | medium |
| 9 | **Longer training + cosine warm restarts**, patience 20, `epochs=150`. | `TrainConfig` | **+2–4** | trivial |
| 10 | **Richer features**: first differences `S_t − S_{t-1}`, Shannon entropy of the per-window flow-size and dst-port distributions, top-k dst-port one-hots, is-internal / subnet-fan features, protocol-pair histogram. 41 → ~70 dims. | `state_windows.STATE_FEATURE_COLS` + `_agg_windows` | **+3–6** | medium |
| 11 | **Dual-timescale history** for slow attacks (slowloris spans hours): a second branch at `L=60, stride=30` (30 min of context) concatenated at the latent. | `models.py`, `sequences.py` | **+2–5** on slow families | medium-high |
| 12 | **Temperature scaling** of the world-model logits on val — fixes Brier 0.081 / ECE 0.059 (vs GAT 0.033). Does not move F1 but makes the probability trustworthy for the rollout CIs. | new post-hoc calibrator | Brier −0.03 | trivial |
| 13 | **GAT**: raise `N_MAX` to 64, add real edge features (bytes, SYN-ratio, direction) not just `log1p(count)`, add a second GAT hop. | `graph_windows.py`, `gat.py` | **+2–4** on GAT | medium |
| 14 ✅ | **Leakage-safe family-stratified split** (`mode="stratified"`, `auto` default). Contiguous benign backbone + whole-episode assignment rotating per family + internal cut only for long episodes + span-boundary sequence purge (`SequenceConfig.purge_boundary_sequences`) + span-purity assertion. See Part 1.9. | `sequences.assign_split`, `sequences._make_sequences`, `sequences.build_sequences` | removes horizon-label / scored-on-trained-window leaks | medium |
| 15 | **Add CTU-13 / UNSW-NB15** as an out-of-distribution eval set (column adapter needed). Does not raise CIC-IDS-2017 F1 but proves generalisation. | new loader | — | medium |

**Fastest credible win:** #1 + #5 + #9 together, ~1–2 hours, expected **F1 +8–15**
with no architecture change.

---

## Part 3 — Model-by-model reference

All models consume the **same** tensors from `artifacts/sequences.npz`:

```
X    [N, L=12, F=53]   RobustScaler-normalised history of 12 state vectors S_{t-11..t}
dt   [N, L=12]         log1p(seconds since previous window)   (unscaled)
```
(current L / F / K / d_model / param counts are regenerated each run into
`research/reports/technical_reference_addendum.md` — trust that file.)

and emit the **same** two heads, so `train.evaluate_split` scores them identically:

```
attack_logits_k  [N, K=6]        pre-sigmoid P(attack) at t+1 … t+6
prog_logits_k    [N, K=6, S=5]   pre-softmax progression state at t+1 … t+6
```

`S=5` progression states: `NORMAL, PRE_ATTACK, ONSET, ACTIVE, CONTINUATION`
(`config.PROGRESSION_STATES`).

### 3.1 SENTINEL-WM — world model (`models.py`) — **803,882 params**

| sub-module | params | role |
|---|---:|---|
| `encoder` (TemporalEncoder) | 682,752 | history → latent `z_t` |
| `stn` (StateTransitionNet) | 66,560 | `z_t → μ, logσ` of `z_{t+1}` |
| `attack_head` | 8,321 | shared 1-step `z → P(A=1)` (used by rollout) |
| `prog_head` | 8,581 | shared 1-step `z → P(Z=c)` (used by rollout) |
| `horizon_attack` | 17,286 | direct MLP `z_t → [K]` (benchmarked head) |
| `horizon_prog` | 20,382 | direct MLP `z_t → [K·S]` |

**`ElapsedTimePositionalEncoding(d=128)`** — real-time-aware position encoding:
`x + MLP(dt)  +  sinusoid[:L]`, where `MLP: 1→128→GELU→128` learns a warp of the
`log1p` inter-window gap, and `sinusoid` is a fixed non-learned index PE buffer
(512×128). So order **and** actual elapsed seconds inform the model.

**`AttnBlock` × `n_layers=3`** — pre-LayerNorm Transformer block:
`x → LN → MultiheadAttention(d=128, heads=4, batch_first) → +x → LN → FFN → +x`,
FFN = `Linear(128, 512) → GELU → Dropout(0.1) → Linear(512, 128)`. A **causal
mask** (`triu(ones(L,L), 1)`) is rebuilt every forward so window `i` only attends
to `≤ i`. The **last block stores its attention matrix** `[B, L, L]`
(`average_attn_weights=True`) — this is the explainability hook.

Latent: `z_t = LN(h)[:, -1, :]` — the final time step of the last layer, `∈ ℝ¹²⁸`.

**`StateTransitionNet`** — probabilistic 1-step dynamics:
`z → [Linear(128,128) → LN → ReLU] × 2 → (μ = Linear(128,128), logσ² = Linear(128,128).clamp(-8, 4))`.
Sample (reparameterised): `z_{t+1} = μ + ε·exp(0.5·logσ²)`, `ε ~ N(0, I)`.
`clamp(-8, 4)` keeps σ in `[0.018, 7.4]` — stops NaNs and runaway variance.

**Heads**

* `horizon_attack`, `horizon_prog` — `Linear(128,128) → GELU → Dropout → Linear(...)`.
  The **direct multi-horizon predictors**; these are what `evaluate_split` and the
  benchmark read. One shot `z_t →` all 6 horizons (no recurrence).
* `attack_head`, `prog_head` — `Linear(128,64) → GELU → Dropout → Linear(64, ...)`.
  **Shared 1-step heads**, applied to `z` (nowcast) *and* to every STN-simulated
  latent in `rollout()`. Supervised on the `k=1` target so the rollout path is
  trained, not just the direct head.

**`forward(x, dt)`** returns `z, z_next_mu, z_next_logvar, z_next, attn,
attack_logits_k, prog_logits_k, attack_logit_now, prog_logits_now,
attack_logit_sim1, prog_logits_sim1`.

### 3.2 LSTM / GRU (`nn_zoo.RNNForecaster`) — **241,572 / 186,532 params**

Input is `X` with the `log1p(dt)` channel appended → `[B, L, 42]`.
`nn.LSTM` / `nn.GRU`, `hidden=128`, `num_layers=2`, `dropout=0.15` between layers,
`batch_first`. Take the **last time step** `out[:, -1, :]`, `LayerNorm`, then
`MultiHorizonHead`. This is the proposal's "LSTM Classifier" baseline #3: a
sequence encoder with **no** state-transition model and **no** rollout.

`MultiHorizonHead(d, K, S)` = `Linear(d,d) → GELU → Dropout` then
`attack = Linear(d, K)`, `prog = Linear(d, K·S).view(B,K,S)` — 21,156 params.

### 3.3 MLP (`nn_zoo.MLPForecaster`) — **402,724 params**

Flatten `[B, L·42 = 420] → Linear(420,512) → LN → GELU → Dropout → 256 → 128 →`
`MultiHorizonHead`. Order is available only through concatenation — the control
for "does an explicitly temporal architecture beat a flat one".

### 3.4 TCN (`nn_zoo.TCNForecaster`) — **168,036 params**

3 residual blocks of **causal dilated 1-D convolution** over the time axis:
block `i` uses `kernel=3`, `dilation = 2^i` (receptive field 1-3-7-15-31 ≥ L=12),
each block = `Conv1d → crop-right(pad) → GELU → Dropout → Conv1d → crop → GELU →
Dropout → + (1×1 down-projection of input)`. Take last position → `LN` →
`MultiHorizonHead`.

### 3.5 GAT — Graph Attention Network (`gat.py`) — **246,820 params**

The proposal's Advanced-tier spatial encoder, **from scratch, no
`torch-geometric`**.

**Graph construction (`graph_windows.py`).** For every `(day, window_index)`
already in `state_windows.parquet`:
* nodes = the `N_MAX = 32` busiest hosts in the window (ranked by degree = flows
  touching them); the rest are dropped. Mean nodes/window ≈ 25.
* node features (`N_NODE_FEAT = 14`): out/in flow-count, out/in bytes, out/in
  packets, out/in SYN, out/in RST, distinct dst ports (out), distinct peers (out),
  `is_internal` (RFC-1918 prefix), total degree. Counts are `log1p`-compressed and
  RobustScaler-normalised on **train-day windows only**; pad rows kept at 0.
* `adj [N_MAX, N_MAX]` — directed edge weight `= log1p(#flows src→dst)`.
* saved to `artifacts/graph_windows.npz` aligned row-for-row with the state
  windows. The GAT dataset (`gat._GraphDataset`) slices `L` consecutive windows
  per anchor on the fly.

**`GraphAttentionLayer`** — GAT (Veličković 2018) **additive** attention (chosen
over GATv2's outer-sum which is `O(N²·D)` in memory and OOMs a 4 GB GPU):
`Wh = W(h).view(B, N, heads, d_out)`;
`e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j)` → `[B, heads, N, N]`;
mask non-edges (and non-nodes) to `-inf`; `softmax` over `j`;
`nan_to_num` so isolated nodes contribute 0; `out = Σ_j α_ij · Wh_j`; concat heads
(+ bias). Self-loops added in the encoder so every node attends to itself.

**`GATWindowEncoder(d_model=96, layers=2, heads=4)`** — 2 GAT layers, GELU,
LayerNorm, **masked mean-pool over nodes** → one 96-dim embedding per window.
39,552 params.

**`GATForecaster`** — per window: `GATWindowEncoder` → `[B, L, 96]` →
`nn.GRU(96, 128, layers=2, dropout=0.12)` over the L window embeddings → last
hidden → `LayerNorm` → `MultiHorizonHead`. GRU = 185,856 params (the bulk).
It has spatial (who-talks-to-whom) **and** temporal structure, but no STN/rollout.

### 3.6 Classical zoo (`baselines.py`)

Not neural. Each = **K = 6 independent binary classifiers** (one per horizon).
Two input regimes:

* `window` — `X[:, -1, :]` → `[N, 41]` (current state vector only, "no temporal")
* `__seq` — `X.reshape(N, L·41)` → `[N, 410]` (flattened history — same information
  the world model sees; this is the fair strong comparison)

| model | key params | class imbalance |
|---|---|---|
| `logistic_regression` | `C=0.5`, `solver=lbfgs`, `max_iter=2000` | `class_weight="balanced"` |
| `random_forest` | 300 trees, `min_samples_leaf=2` | `class_weight="balanced_subsample"` |
| `extra_trees` | 300 trees | `class_weight="balanced"` |
| `hist_gradient_boosting` | `lr=0.07`, `max_iter=300`, `l2=1.0`, internal early-stop | balanced `sample_weight` |
| `mlp_sklearn` | `(256,128)`, `lr=8e-4`, `max_iter=80`, early-stop | — |
| `linear_svc` | `C=0.5`, `max_iter=4000`; probabilities via `sigmoid(decision_function)` | `class_weight="balanced"` |
| `knn` | `k=25`, distance-weighted (window only) | — |
| `gaussian_nb` | — (window only) | balanced `sample_weight` |
| `xgboost` | 400 trees, `depth=6`, `lr=0.06`, `subsample=0.8`, `hist` | balanced `sample_weight` |
| `lightgbm` | 500 trees, 63 leaves, `lr=0.06`, `n_jobs=1` | — (**crashes on this Windows box**; auto-skipped) |
| `persistence` | non-trained: predict `A_{t+k} = A_t` | — |

Fitted estimators → `research/models/classical/<name>.pkl` (a Python `list` of 6
estimators) + `<name>.meta.json` (feature names, input regime, calibrated
threshold, headline metrics).

---

## Part 4 — Training loop (world model `train.py`; deep models `nn_common.py`)

Both use the **same protocol**.

### 4.1 Optimizer & schedule

* **AdamW**, `lr = 2e-4` (world model; `8e-4` for the NN zoo), `weight_decay = 1e-5`.
* **CosineAnnealingLR** over the full `epochs` budget (70 world model, 50 zoo/GAT):
  `lr(e) = lr_min + ½(lr_max − lr_min)(1 + cos(π·e / E))`, `lr_min = 0`.
* **Gradient clipping**: `clip_grad_norm_(…, max_norm = 1.0)` every step.
* `batch_size = 256`, `DataLoader(shuffle=True, drop_last=False)`.
* Seeds: `numpy`, `torch`, `torch.cuda` all set to `1337` (`set_seed`).

### 4.2 Class weighting (from the **train** split only)

* **Attack** — `pos_weight = (1 − p) / p` where `p = mean(y_atk)`  (`p` clamped to
  `[1e-3, 1−1e-3]`). On this data `p ≈ 0.15 → pos_weight ≈ 6.1`, i.e. a missed
  attack costs ~6× a false alarm in the loss.
* **Progression** — `w_c = sqrt(N / n_c)`, then normalised to mean 1 (√-inverse
  frequency; softer than raw inverse frequency so the rare CONTINUATION class does
  not dominate). Typical: `[0.17, 0.78, 1.23, 0.45, 2.37]` for
  `NORMAL … CONTINUATION`.

### 4.3 Loss function

**World model** — `models.joint_loss`:

```
L = w_attack       · BCEWithLogits(attack_logits_k, y_atk,  pos_weight)          # w_attack = 1.5
  + w_attack · 0.5  · [ BCE(attack_logit_now,  y_atk[:,0]) + BCE(attack_logit_sim1, y_atk[:,0]) ]
  + w_progression   · CE(prog_logits_k.reshape(B·K,S), y_prog.reshape(B·K), w_c)  # w_progression = 0.5
  + w_progression · 0.5 · [ CE(prog_logits_now, y_prog[:,0]) + CE(prog_logits_sim1, y_prog[:,0]) ]
  + w_next_state    · MSE(z_next_mu, z_next_target)                               # w_next_state = 0.25
  + w_kl            · KL( N(μ, σ) ‖ N(0, I) )                                     # w_kl = 1e-4
```

* term 1 — the benchmarked multi-horizon attack head.
* term 2 — supervises the **shared 1-step heads** that `rollout()` calls
  (0.5 weight so they help, not dominate).
* terms 3–4 — progression-state head, same structure.
* term 5 — **the world-model objective**: the STN must reproduce the *next*
  window's latent. `z_next_target = encoder(x_shifted, dt)["z"].detach()`, where
  `x_shifted = concat(x[:, 1:, :], x_next[:, None, :])` (history advanced one
  window). `w_next_state = 0.25` — light, so it regularises the latent without
  starving the attack head (was `1.0`; lowering it lifted attack F1 ~0.04).
* term 6 — `KL = mean( −½ (1 + logσ² − μ² − exp(logσ²)) )`. `w_kl = 1e-4`: barely
  regularises σ so it stays a meaningful "trajectory uncertainty" signal without
  collapsing the latent.

**Deep zoo / GAT** — `nn_common._loss` (simpler, no STN):
`L = BCEWithLogits(attack_logits_k, y_atk, pos_weight) + 0.5 · CE(prog_logits_k, y_prog, w_c)`.

### 4.4 Early-stopping logic

Per epoch, after training, evaluate the **val** split and compute

```
score = val_any_horizon_F1  +  0.5 · val_progression_accuracy
```

* if `score > best`: save `state_dict` (CPU clone), `patience = 0`.
* else: `patience += 1`; **stop** when `patience ≥ early_stop_patience`
  (14 world model, 8 zoo/GAT).
* on stop (or budget end): reload the best `state_dict`.

Rationale for the combined score: F1 alone is jumpy at a fixed threshold on a
small val set; adding progression accuracy stabilises checkpoint selection. Its
cost: the chosen checkpoint is a **compromise** between the two objectives, not
the attack-F1 argmax — one reason the world model is mid-pack on raw F1
(see Part 1.8; Part 2 #4 removes this).

### 4.5 Threshold calibration (done once, after training)

1. Reload best weights. Evaluate **val** with `threshold = None` →
   `metrics.calibrate_threshold(y_val.max(1), prob_val.max(1), target_fpr=0.05)`:
   the **smallest** threshold on `linspace(0.01, 0.99, 99)` whose FPR ≤ 5 % on val
   (default 0.5 if none qualifies).
2. Freeze that threshold. Evaluate **test** with it — no test-set peeking.

### 4.6 Checkpoint format

`torch.save` dict — everything needed to re-instantiate without re-reading config:

```python
# world model  (artifacts/world_model.pt)
{ "state_dict", "config": {n_features, n_states, horizon, model:{…ModelConfig…}},
  "feature_names": [...41...], "alert_threshold": float, "sequence": {"L":10,"K":6} }

# deep zoo / GAT  (research/models/nn/<name>.pt)
{ "state_dict", "kind": "lstm"|"gru"|"mlp"|"tcn"|"gat", "family",
  "feature_names", "alert_threshold", "n_features", "sequence": {L,K},
  "extra": {…}  # GAT: n_node_feat, n_max, d_model }
```

Loaded uniformly by `registry.load_predictor(name)` → `.predict(X, dt[, graph])`
→ `{"attack_prob_k": [N,K], "progression_k": [N,K] | None, "threshold": float}`.

### 4.7 Metrics logged

`world_model_metrics.json` (`val` + `test`, each a full `evaluate_split` dict),
`training_curve.csv` (`epoch, loss_*, val_f1, val_auroc, val_prog_acc, val_mlt`),
`research/logs/nn/<name>_curve.csv` for the zoo.

---

## Part 5 — Forward-simulation / K-step rollout (`models.rollout`, `forward_sim.py`)

This is the "infiltration prediction engine". It is **inference only**
(`@torch.no_grad`), used by `forward_sim.simulate_split` and the dashboard-style
timeline — **not** by `evaluate_split` (which uses the direct `attack_logits_k`
head).

### 5.1 The rollout, step by step

Input: one anchor's `x [L, F]`, `dt [L]`. Config: `K = 6`, `M = mc_samples = 50`.

```
enc  = encoder(x, dt)                    # z0 = enc["z"]  ∈ ℝ¹²⁸   (+ attn kept)
z    = z0 repeated M times               # [B·M, 128]   — M Monte-Carlo particles
for k in 1 … K:
    μ, logσ² = STN(z)
    z        = μ + ε·exp(0.5·logσ²)      # advance every particle one window (stochastic)
    h        = { attack_head(z), prog_head(z) }
    atk[:, k, :]   = sigmoid(h.attack_logit).reshape(B, M)
    prg[:, k, :, :]= softmax(h.prog_logits).reshape(B, M, 5)
```

Aggregate across the `M` particles, per horizon `k`:

* `attack_prob[k]   = mean_m atk[:,k,m]`
* `attack_ci[k]     = [percentile(atk[:,k,:], 2.5),  percentile(…, 97.5)]`  (95 % band)
* `attack_std[k]    = std_m atk[:,k,m]`  — trajectory uncertainty
* `prog_prob[k]     = mean_m prg[:,k,m,:]`  → `prog_state[k] = argmax`

So the CI comes from **propagating the STN's Gaussian uncertainty forward** — it
widens with `k` because each step re-samples. High `attack_std` = "the trajectory
is sensitive; many futures" (proposal §11.3).

### 5.2 What `forward_sim.simulate_anchor` adds

For each horizon step it builds:

```
{ k, horizon_seconds = k·10,
  attack_prob, attack_ci = [lo, hi], attack_std,
  progression_state, progression_dist = {state: p},
  attck = assess_forecast(prog_dist, horizon_k=k, dominant_family_hint, prev_family, attack_prob) }
```

and per anchor:

```
{ meta, alert_threshold,
  alert            = (∃ k: attack_prob[k] ≥ threshold),
  first_alert_k    = first such k,
  lead_time_seconds= (K − first_alert_k + 1)·10,   # 0 if no alert
  max_attack_prob  = max_k attack_prob[k],
  horizon = [ …per-k… ],
  driving_features = { top_features, temporal_saliency_gradient, temporal_saliency_attention }   # if --explain
}
```

`dominant_family_hint` is looked up from `state_windows.parquet` for the anchor's
*current* window — the best guess for "if this becomes an attack, what kind" —
and passed to the ATT&CK layer.

### 5.3 Explainability attached per anchor (`explain.make_world_model_explainer`)

* **`gradient_x_input`** — backprop `max_k attack_logits_k` to `x`;
  `saliency = |∂/∂x| · |x|` → `per_window` (row sums, normalised) and
  `top_features` (column sums, with sign of the last-step gradient).
* **`attention_saliency`** — last encoder block's attention matrix, summed over
  query positions, normalised → importance of each of the 10 history windows
  ("the burst 4 windows ago drove this").
* Global (`explain.run_all`): **SHAP** — `shap.KernelExplainer` on a
  `flatten(seq) → sigmoid(max_k attack_logits_k)` wrapper (64 background / 64
  explain rows, `nsamples=200`); **finite-difference fallback** with the same
  output shape if `shap` is absent. Baselines: `LinearExplainer` /
  `TreeExplainer`, else `|coef_|` / `feature_importances_`.

---

## Part 6 — ATT&CK phase logic (`attack_stages.py`)

**The constraint.** CIC-IDS-2017 labels an attack *type per flow* ("this flow is
DoS Hulk"). It does **not** label a *MITRE tactic per time window* ("at 11:42:30
the adversary is in the Impact tactic"). So ATT&CK is **never a training target**.
It is a deterministic, auditable, **two-layer** function applied *after* the model,
and every output carries an explicit **confidence ∈ {High, Medium, Low}** plus a
plain-text **rationale**.

### 6.1 Layer A — static knowledge base (`CIC_LABEL_TO_ATTACK`)

Literature-derived `attack_family → AttackRef(tactic, technique_ids,
kill_chain_phase, base_confidence, note)`. Full table:

| family | tactic | techniques | phase | base conf |
|---|---|---|---|---|
| BENIGN | None | — | None | High |
| PortScan | Reconnaissance / Discovery | T1595, T1046 | Reconnaissance | High |
| FTP-Patator | Credential Access / Initial Access | T1110, T1190 | Initial Access | Medium |
| SSH-Patator | Credential Access / Initial Access | T1110, T1021 | Initial Access | Medium |
| Web Attack Brute Force | Credential Access | T1110.004 | Initial Access | Medium |
| Web Attack XSS | Initial Access | T1190 | Initial Access | Medium |
| Web Attack SQL Injection | Initial Access / Execution | T1190, T1059 | Initial Access | Medium |
| Heartbleed | Initial Access / Credential Access | T1190 | Initial Access | Medium |
| DoS Hulk / GoldenEye | Impact | T1499, T1499.002 | Impact | High |
| DoS slowloris / Slowhttptest | Impact | T1499, T1499.003 | Impact | High |
| DDoS | Impact | T1498 | Impact | High |
| Bot | Command and Control | T1071, T1572 | Command & Control | Medium |
| Infiltration | Initial Access → Discovery → Lateral Movement | T1190, T1046, T1021 | Lateral Movement | Medium |

### 6.2 Layer B — context resolver (`assess_window`)

`assess_window(progression_state, dominant_family, prev_family, attack_ratio) →
StageAssessment`. Rules, in order:

| progression state | phase emitted | confidence rule |
|---|---|---|
| **NORMAL** | `None` | High — "window contains only BENIGN flows" |
| **PRE_ATTACK** & family scan-like (PortScan/Infiltration) | Reconnaissance (T1595, T1590) | Medium if port-axis scanner, else Low; rationale notes "N steps before a confirmed onset" |
| **PRE_ATTACK** & other family | "Reconnaissance (inferred)" (T1590) | **Low** — "benign window, tactic inferred from timing only" |
| **ONSET** | Layer-A tactic of `dominant_family` | Layer-A confidence **downgraded one notch** — "single-window evidence (attack_ratio=…)" |
| **ACTIVE** | Layer-A tactic | Layer-A confidence, **−1 notch if `attack_ratio < 0.15`** |
| **CONTINUATION** & family changed vs `prev_family` | Layer-A tactic of the **new** family; sets `family_transition = "old → new"` | new family's base confidence **−1** — "adversary changed technique within the episode" |
| **CONTINUATION** & same family | Layer-A tactic (or "Impact" fallback) | **−1 notch** — "episode tail; activity persisting or winding down" |
| unrecognised state | Layer-A tactic | Low |

Confidence downgrade helper: `_downgrade("High", 1) → "Medium" → "Low"` (floored).

### 6.3 Forecast variant (`assess_forecast`) — used by the rollout

`assess_forecast(state_probs[5], horizon_k, dominant_family_hint, prev_family,
attack_prob)`:

1. `ps = PROGRESSION_STATES[argmax(state_probs)]`.
2. `a = assess_window(ps, family_hint, prev_family, attack_ratio ≈ attack_prob)`.
3. If `ps ∈ {ONSET, ACTIVE, CONTINUATION}` **and** `family_hint == "BENIGN"`
   (model forecasts an attack but there is no family context): override to
   `phase = "Attack (unspecified)"`, `tactic = "Unspecified (family unknown at
   forecast time)"`, `confidence = Low` — **never silently emit "None"**.
4. **Rollout-error downgrade**: `steps = 1 + (horizon_k − 1) // 3` — drop
   confidence one notch per **3 horizon steps** (`+10–30 s` = −1, `+40–60 s` = −2).
5. Prepend the rationale with `"forecast +{k·10}s (step k=…); P(state)=…, P(attack)=…"`.

### 6.4 Where it is written into the pipeline

`state_windows._attach_attack_stage` runs `assess_window` row-wise over every
state window and adds the columns `attck_tactic, attck_techniques, attck_phase,
attck_confidence, attck_rationale, attck_family_transition` to
`state_windows.parquet`. The forecast variant runs live inside
`forward_sim.simulate_anchor`. `research/simulations/attck_stage_forecast.csv` is
the flat `(window, k, horizon_s, attack_prob, progression_state, attck_phase,
confidence)` export. `python -m sentinel_wm.attack_stages` runs an 8-case
self-test of every rule above.

---

## Part 7 — Metric definitions (what each number means *on this task*)

All in `sentinel_wm/metrics.py`; test-split only, at the val-calibrated threshold.

| metric | definition here | note |
|---|---|---|
| **any-horizon F1 / P / R** | positive = attack predicted in **any** of `t+1…t+6`; `y_true = y_atk.max(1)`, `y_prob = attack_prob.max(1)` | the headline number |
| **per-horizon F1** (`f1_k1…k6`) | same at each `k` separately | shows forecast decay |
| **FPR** | `FP / (FP + TN)` on benign windows | held ≤ 5 % by construction |
| **AUROC** | `roc_auc_score(y_true, y_prob)`, any-horizon | threshold-free ranking quality; ~0.90 |
| **Brier(k1)** | `mean((p_{k=1} − y_{k=1})²)` | probability sharpness+calibration at +10 s |
| **ECE(k1)** | 10-bin expected calibration error at +10 s | reliability of `p` |
| **Mean Lead Time** (`lead_time`) | per attack **episode** (onset = attack window whose predecessor is benign): `t_onset − t_first_contiguous_warning`, look-back capped at `K` windows; averaged over warned episodes | ~10 s here — a *split* artefact (Part 1.6) |
| **detection_rate** | warned episodes / total episodes | world model 0.22 (highest of trained models) |
| **false_alarm_rate** | warnings on benign windows with no attack within `K` / benign windows | operational-trust proxy |
| **progression_acc** | `mean(argmax(prog_logits_k) == y_prog)` over all `k` | world model 0.854; only the deep models have it |

**Bottom line:** AUROC ≈ 0.90 and progression accuracy ≈ 0.85 say the models
*rank* and *stage* attacks well; F1 ≈ 0.6 is the cost of (a) forecasting 60 s
ahead, (b) from 10-s window aggregates where the median positive is ~11 %
malicious, (c) on a test set that is 69 % low-and-slow Botnet, (d) at a strict
FPR ≤ 5 % operating point. Part 2 lists the concrete levers, cheapest first.
