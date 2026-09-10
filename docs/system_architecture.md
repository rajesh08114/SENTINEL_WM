# SENTINEL-WM (system) — full architecture

Every layer, every parameter, the forward and backward passes, and the reason for
each choice. `SENTINEL-WM (system)` is the row that tops the benchmark
(PR-AUC ≈ 0.992). It is a **composite**: the SENTINEL-WM *world model* blended with
three decorrelated neural sequence models. This document describes all four
components plus the composition.

Code: `research/sentinel_wm/{models.py, nn_zoo.py, nn_common.py, train.py,
pretrain.py, benchmark.py, registry.py, config.py}`.

---

## 0. What it is (one picture)

```
 flow CSV / telemetry
        │  preprocessing.clean_flow_frame → state_windows.build_state_windows
        ▼
 X_scaled [B, L=12, F=53]   (RobustScaler, fit on real-train only)
 dt       [B, L=12]         (log1p seconds since previous window)
        │
        ├──────────────── COMPONENT A : SENTINEL-WM world model (532,042 params) ─┐
        │   GRU encoder → z_t [B,160]                                             │
        │     ├─ StateTransitionNet  z_t → μ,logσ  → z_{t+1} ~ N(μ,σ)             │
        │     ├─ horizon_attack  z_t → attack_logits_k [B,6]   ← BENCHMARKED head │
        │     ├─ horizon_prog    z_t → prog_logits_k  [B,6,5]                     │
        │     └─ shared attack_head / prog_head  (applied to z_t AND to every     │
        │        rolled-out latent → the K-step Monte-Carlo forward simulation)   │
        │   wm_predict = mean( direct-head sigmoid of {base ckpt + snapshots} )   │
        │        → wm_prob [B,6]   (+ progression state, + rollout CIs in forward_sim)
        │                                                                        │
        ├── COMPONENT B : TCN  (172,644)   sigmoid(attack_logits_k) → p_tcn [B,6] │
        ├── COMPONENT C : LSTM (247,716)                            → p_lstm[B,6] │
        └── COMPONENT D : GRU  (191,140)                            → p_gru [B,6] │
                                        │                                        │
                         member_prob = mean(p_tcn, p_lstm, p_gru)  [B,6]         │
                                        │                                        │
   system_prob = w · wm_prob + (1 − w) · member_prob     ◄── w tuned on validation┘
                                        │
              alert = system_prob.max(1) ≥ threshold      (threshold = best-F1 on val)
              progression state / ATT&CK phase / CIs      ← from COMPONENT A only
```

`B` = batch, `L` = history windows (12), `F` = state features (53), `K` = horizon
steps (6), `S` = progression states (5: NORMAL, PRE_ATTACK, ONSET, ACTIVE,
CONTINUATION). Window = 10 s, so the forecast covers +10 s … +60 s.

**The composition has zero learnable parameters** — `w` is *selected* by a grid
search on validation, not trained. So "the system" is 4 independently trained
networks (**1,143,542 params total**) plus a fixed scalar blend.

---

## 1. The input tensor (what all four components consume)

`sequences.build_sequences` produces, per anchor window `t`:

| tensor | shape | content |
|---|---|---|
| `X` | `[B, 12, 53]` | history `S_{t-11..t}`, each a 53-dim state vector, **RobustScaler-normalised** (scaler fit on `split=="train" & is_synthetic==0` only) |
| `dt` | `[B, 12]` | `log1p(seconds since previous window)` per history step, unscaled |

The 53 features (`state_windows.STATE_FEATURE_COLS`, order matters): 41 base
(volume/rate, connection dynamics, TCP-flag rates, timing/burstiness, Tier-2
packet stats, scan signatures, protocol mix, 2 time features) + 4
distribution-entropy + 8 first-differences.

Every component prepends the `dt` channel → effective input width **54**.

---

## 2. COMPONENT A — the SENTINEL-WM world model

`models.SentinelWorldModel`, built by `build_model(n_features=53, cfg)`.
**532,042 parameters.** `ModelConfig`: `d_model=160`, `n_heads=4`,
`dropout=0.15`, `stn_hidden=160`, `encoder="gru"`.

### 2.1 GRU encoder — `GRUEncoder` — 344,800 params

Turns the 12-step history into one latent `z_t ∈ ℝ¹⁶⁰`.

| # | layer | shape | params | note |
|---|---|---|---|---|
| 1 | `in_proj` = `Linear(54 → 160)` | W (160,54), b (160) | 8,640 + 160 | 53 features + 1 log-dt channel, concatenated |
| 2 | `norm_in` = `LayerNorm(160)` | γ, β (160) | 320 | stabilise the projected input |
| 3 | `gru` = `nn.GRU(160 → 80, num_layers=2, bidirectional=True, dropout=0.15)` | — | **232,320** | see split below |
| 4 | `attn` = `nn.MultiheadAttention(160, 4 heads)` | in_proj (480,160)+b, out_proj (160,160)+b | **103,040** | 1-query read-out over the L steps |
| 5 | `ln` = `LayerNorm(160)` | γ, β (160) | 320 | post read-out |

GRU internals (per direction, per layer: 3 gates × d_out): `weight_ih (3·80, 160)`
= 38,400, `weight_hh (3·80, 80)` = 19,200, two bias vectors 240 each. Four such
blocks (2 layers × 2 directions) → **232,320**. `d//2 = 80` per direction so the
concatenated hidden is 160 = `d_model`.

**Forward** (`GRUEncoder.forward(x, dt)`):
```
h0   = norm_in( in_proj( cat([x, dt.unsqueeze(-1)], dim=-1) ) )      # [B,12,160]
seq, _ = gru(h0)                                                     # [B,12,160]  (fwd⊕bwd)
q    = seq[:, -1:, :]                                                # [B,1,160]  query = last step
ctx, w = attn(q, seq, seq)         ; last_attn = w                   # [B,1,160], w [B,1,12]
z    = ln( seq[:, -1, :] + dropout(ctx.squeeze(1)) )                 # [B,160]
return {seq, z, attn=w}
```

**Why a Bi-GRU, not the Temporal Transformer:** on ~8.5 k training sequences of
only 12 short steps, RNN recurrence is far more sample-efficient. The causal
Transformer (still selectable via `ModelConfig.encoder="transformer"`) ceilinged
~3 AUROC points lower and plateaued by epoch 13. **Bidirectional** because at
*encoding* time the whole history window is available (only the *forecast* is
causal), and backward context sharpens `z_t`. The `+1` log-dt channel lets the
encoder see the real elapsed time between windows. The **attention read-out**
pools the 12 steps into `z_t` and its weights `w [B,1,12]` are the
per-history-window saliency used by `explain.attention_saliency`.

### 2.2 State-Transition Network — `StateTransitionNet` — 103,680 params

The learned dynamics: `z_t → distribution over z_{t+1}`.

| # | layer | params |
|---|---|---|
| 1 | `Linear(160 → 160)` + `LayerNorm(160)` + `ReLU` | 25,600 + 160 + 320 |
| 2 | `Linear(160 → 160)` + `LayerNorm(160)` + `ReLU` | 25,600 + 160 + 320 |
| 3 | `mu` = `Linear(160 → 160)` | 25,600 + 160 |
| 4 | `logvar` = `Linear(160 → 160)` | 25,600 + 160 |

**Forward:** `h = body(z)`; `μ = mu(h)`; `logσ² = logvar(h).clamp(-8, 4)`.
Sample: `z_next = μ + randn_like(μ) · exp(0.5·logσ²)` — the **reparameterisation
trick**, so gradients flow through `μ` and `logvar`.

**Why probabilistic:** a point estimate of `z_{t+1}` cannot express *predictive
uncertainty*. Sampling `M` latents and pushing each through the heads gives the
95 % confidence band on `P(attack at t+k)` and a distribution over progression
states — the core of the "forward simulation" deliverable. `logvar` is clamped
for numerical stability; a **tiny KL(N(μ,σ)‖N(0,1))** term (weight `1e-5`) keeps
`σ` from collapsing to 0 without pulling `μ` toward the origin. LayerNorm (not
BatchNorm) because batches are small and the data is sequential.

### 2.3 The four heads — 83,562 params

| head | layers | params | used by |
|---|---|---|---|
| `horizon_attack` | `Linear(160→160)` · GELU · Dropout · `Linear(160→6)` | 25,600+160 + 960+6 = **26,726** | **the benchmarked `attack_logits_k [B,6]`** |
| `horizon_prog` | `Linear(160→160)` · GELU · Dropout · `Linear(160→30)` | 25,600+160 + 4,800+30 = **30,590** | `prog_logits_k [B,6,5]` (reshaped) |
| `attack_head` | `Linear(160→80)` · GELU · Dropout · `Linear(80→1)` | 12,800+80 + 80+1 = **12,961** | shared — applied to `z_t` and every rolled-out latent |
| `prog_head` | `Linear(160→80)` · GELU · Dropout · `Linear(80→5)` | 12,800+80 + 400+5 = **13,285** | shared — as above |

**Why two families of heads.** The `horizon_*` heads project *directly* from
`z_t` to all `K` steps in one shot — stable gradients, and this is the output the
benchmark scores. The shared `attack_head` / `prog_head` take *any* latent, so
`rollout()` can apply them to each of the `K` STN-simulated latents. Both paths
are supervised (§2.5) so the rollout is genuinely trained, not a dead branch.
The heads are 2-layer MLPs, not linear: a linear head under-fit next to the
LSTM/GAT heads.

### 2.4 Forward pass — training (`SentinelWorldModel.forward`)

```
enc            = encoder(x, dt)                      # z_t = enc["z"]  [B,160]
μ, logσ²       = stn(z_t)
z_next         = stn.sample(μ, logσ²)                # reparameterised  [B,160]

attack_logits_k = horizon_attack(z_t)               # [B,6]
prog_logits_k   = horizon_prog(z_t).view(B,6,5)     # [B,6,5]

h0 = apply_heads(z_t)      → attack_logit_now,  prog_logits_now       (nowcast)
h1 = apply_heads(z_next)   → attack_logit_sim1, prog_logits_sim1      (STN 1-step)
```
Returns all of the above + `z, z_next_mu, z_next_logvar, z_next, attn`.

### 2.5 Backward pass — training (`joint_loss` + `train.train`)

Per batch, first compute the **teacher-forced next latent** under `no_grad`:
```
x_shift        = cat([x[:,1:,:], x_next.unsqueeze(1)], dim=1)        # window shifted +1
z_next_target  = encoder(x_shift, dt)["z"]                           # detached
```
Then `out = model(x, dt)` and:

| term | formula | weight |
|---|---|---|
| `l_attack` | `focal_bce(attack_logits_k, y_atk, γ=1.0, pos_weight≈8–11)` **+** `0.5·[focal_bce(attack_logit_now, y¹) + focal_bce(attack_logit_sim1, y¹)]` | `w_attack = 3.0` |
| `l_prog` | `CE(prog_logits_k, y_prog, weight=prog_w)` **+** `0.5·[CE(prog_logits_now, yp¹) + CE(prog_logits_sim1, yp¹)]` | `w_progression = 0.5` |
| `l_next` | `MSE(z_next_mu, z_next_target)` | `w_next_state = 0.10` |
| `l_kl` | `mean(−0.5·(1 + logσ² − μ² − e^{logσ²}))` | `w_kl = 1e-5` |
| `l_kd` | soft-BCE to teacher probs — **off** (`distill=False`) | `w_distill = 0.5` |

`focal_bce`: `BCEWithLogits · (1−p_t)^γ`, `γ=1.0` (gentle — `γ>1.5` distorts
probabilities). `pos_weight = (1−p)/p` from the train positive rate ≈ 8–11.
`prog_w = normalise( sqrt(total/count_c) )` — √-inverse-frequency class weights.

**Two-stage schedule** (`two_stage=True`, `two_stage_frac=0.20`, `epochs=150`):

* **Stage A (epochs 1–30):** `total = 3.0·l_attack` only. Trains the encoder +
  `horizon_attack` + shared `attack_head` on the classification signal alone.
  Tracks the best-val state by `f1 + 0.5·auroc`; no early stop.
* **A→B transition:** reload stage-A's *best-val* state (not its overfit tail);
  new `AdamW` at `lr·0.5`; new `CosineAnnealingLR`.
* **Stage B (epochs 31–…):** full `total`. Adds the STN + progression heads.
  Early stop on `val_f1 + 0.5·val_prog_acc`, patience 40.

Optimiser: **AdamW**, `lr=2.5e-4` (A) / `1.25e-4` (B), `weight_decay=2e-5`,
grad-norm clip `1.0`. Scheduler: `CosineAnnealingWarmRestarts(T_0=30)` in A,
`CosineAnnealingLR` in B. Batch 256. Balanced `WeightedRandomSampler` lifts
positive-window frequency to ≥ 30 %/batch. `AugmentedSeqDataset` applies
train-only tensor-space jitter / positive-mixup / window-dropout / time-roll.

**Gradient routes:**
`l_attack` → `horizon_attack`, `attack_head`, encoder.
`l_prog`   → `horizon_prog`, `prog_head`, encoder.
`l_next`   → `stn.mu` and (via reparam) `stn.logvar`, `stn.body`; **encoder does
not** get `l_next` gradient (target is detached).
`l_kl`     → `stn.mu`, `stn.logvar`.
The reparam sample `z_next = μ + ε·σ` passes `∂/∂μ = 1`, `∂/∂σ = ε` — so
`l_attack`/`l_prog` on `attack_logit_sim1`/`prog_logits_sim1` also train the STN.

### 2.6 Self-supervised pre-training (`pretrain.py`)

Before stage A, if `ssl_pretrain=True`: a `_MaskedReconstructor` (= the same
`build_encoder` + a `mask_vec` parameter + a `Linear(160→53)` recon head) is
trained for `ssl_epochs=40` on **all ~8.5 k train sequences** (no labels). 25 %
of the L windows (never the last) are replaced by `mask_vec`; loss is
`MSE(pred[mask]/σ_f, x[mask]/σ_f)` (per-feature-std-normalised). Optimiser AdamW
at `lr·3`. The encoder weights are then loaded into the world model
(`strict=False`). Rationale: ~1.2 k labelled positives is too few to learn the
state manifold; masked reconstruction learns it label-free.

### 2.7 Inference paths

**`rollout(x, dt, K, M=50)`** — the forward simulation. `@torch.no_grad`.
```
z0 = encoder(x, dt)["z"]                       # [B,160]
z  = z0 repeated M times                       # [B·M,160]
for k in 1..K:
    μ,logσ² = stn(z);  z = μ + ε·σ             # advance one 10-s step
    p = sigmoid(attack_head(z));  q = softmax(prog_head(z))
    atk[:,k,:] = p ;  prg[:,k,:,:] = q
attack_prob   = atk.mean(M)                    # [B,K]
attack_ci_lo/hi = percentile(atk, 2.5 / 97.5, M)
prog_prob     = prg.mean(M) ;  prog_state = argmax
```
This produces the per-horizon `P(attack)` **with a 95 % CI**, the progression-state
trajectory, and (via `attack_stages.assess_forecast`) the ATT&CK phase +
confidence per step. Used by `forward_sim.simulate_anchor` — the backend product.

**`wm_predict(model, X, dt, snapshots, self_ensemble=True, mc)`** — the
*benchmarked* SENTINEL-WM prob.
```
for mdl in {base checkpoint} ∪ {snapshot checkpoints}:
    direct = sigmoid( mdl(X, dt)["attack_logits_k"] )        # [N,6]
    if self_ensemble and self_ensemble_direct_w < 0.999:     # currently 1.0 → skipped
        direct = w·direct + (1−w)·rollout(mdl).attack_prob
wm_prob = mean over models
prog    = base model's prog_logits_k.argmax(-1)
```
**Snapshots**: `train.py` saves a checkpoint at every cosine warm-restart trough
in stage B (`(ep−30) % 30 == 0`) → typically ep 60, 90, (120). Averaging their
direct-head probs is a cheap decorrelated variance reducer.
**`self_ensemble_direct_w = 1.0`**: on the leakage-safe split the test anchors
sit mid-episode (now-cast regime) where the rollout regresses toward the base
rate and *lowers* F1, so the benchmarked prob is the direct head only. The
rollout stays in `forward_sim` for the CIs / progression / ATT&CK.

---

## 3. COMPONENTS B, C, D — the blend members

Three pure sequence→multi-horizon classifiers (`nn_zoo.py`). **No STN, no
rollout, no progression trajectory** — they exist to produce a decorrelated
second opinion on `P(attack)`. All three share one head.

### 3.1 Shared `MultiHorizonHead` (`d` = body width)

`body` = `Linear(d→d)` · GELU · Dropout(0.1) ; `attack` = `Linear(d→6)` ;
`prog` = `Linear(d→30)` → `view(B,6,5)`. Only `attack_logits_k` is used by the
system.

### 3.2 COMPONENT B — TCN (`TCNForecaster`) — 172,644 params

Causal dilated 1-D conv stack over time. Input `[B, 54, 12]` (transposed).

| block | layers | dilation | receptive field | params |
|---|---|---|---|---|
| `_TCNBlock 0` | `Conv1d(54→96,k=3)` ·GELU·Drop· `Conv1d(96→96,k=3)` ·GELU·Drop· + `Conv1d(54→96,k=1)` residual | 1 | 1→3 | 15,552+96 + 27,648+96 + 5,184+96 |
| `_TCNBlock 1` | `Conv1d(96→96,k=3)` ×2 + identity residual | 2 | →7 | 27,648+96 + 27,648+96 |
| `_TCNBlock 2` | `Conv1d(96→96,k=3)` ×2 + identity residual | 4 | →15 (≥ L=12) | 27,648+96 + 27,648+96 |
| `norm` | `LayerNorm(96)` | — | — | 192 |
| `head` | `MultiHorizonHead(96, 6, 5)` | — | — | 9,216+96 + 576+6 + 2,880+30 |

Each block: `y = GELU(crop(conv1(x)))`; `y = GELU(crop(conv2(y)))`;
`out = GELU(y + residual)`. `crop` removes the `(k−1)·dilation` right-padding →
**causal** (a step only sees the past). `forward`: stack → `LayerNorm(h[:,:,-1])`
(the last time step) → head.

**Why TCN:** dilations 1-2-4 give receptive field 15 ≥ L=12 in 3 blocks, it is
translation-equivariant, has no recurrence (fast, no cuDNN-eval issues), and is
the most sample-efficient now-caster on ~7 k sequences.

### 3.3 COMPONENT C — LSTM (`RNNForecaster("lstm")`) — 247,716 params

`nn.LSTM(54 → 128, num_layers=2, batch_first, dropout=0.15)` → `LayerNorm(128)` on
the last step → `MultiHorizonHead(128, 6, 5)`.
LSTM params: layer 0 `weight_ih (512,54)`=27,648, `weight_hh (512,128)`=65,536,
biases 512+512; layer 1 `weight_ih (512,128)`=65,536, `weight_hh (512,128)`=65,536,
biases 512+512 (`4·128` = input/forget/cell/output gates). Head 20,930.
`forward`: `out,_ = lstm(cat([x,dt],−1))` ; `head(norm(out[:,−1,:]))`.

### 3.4 COMPONENT D — GRU (`RNNForecaster("gru")`) — 191,140 params

Identical to C but `nn.GRU(54 → 128, 2 layers)` (`3·128` gates instead of 4) →
`weight_ih (384,·)`, `weight_hh (384,128)`. Head 20,930.

### 3.5 Backward pass — members (`nn_common.train_nn` + `_loss`)

Each member is trained **independently**, same recipe:
```
la   = focal_bce(attack_logits_k, y_atk, γ=1.0, pos_weight=(1−p)/p)
lp   = CE(prog_logits_k.reshape(B·6, 5), y_prog, weight=√-inv-freq)
loss = la + 0.5·lp
loss.backward() ; clip_grad_norm(1.0) ; AdamW(lr=8e-4).step()
```
Scheduler `CosineAnnealingWarmRestarts(T_0=30)`; batch 256; balanced sampler
(≥30 %/batch); `AugmentedSeqDataset` on; early stop on `val_f1 + 0.5·prog_acc`,
patience 40; alert threshold frozen on validation, then test scored. Checkpoint →
`runs/models/nn/{tcn,lstm,gru}.pt`.

**Why LSTM *and* GRU *and* TCN** (three, not one): they make *decorrelated*
errors — gated recurrence with a cell state (LSTM), gated recurrence without one
(GRU), and dilated convolution (TCN). Averaging their probabilities cancels
independent noise. They were chosen over GAT (needs per-window host graphs as a
side input) and the classical `__seq` boosters (which overfit the ~440 positive
train sequences and collapsed on the held-out families).

---

## 4. The composition (`benchmark._score_world_model`, `registry._SystemPredictor`)

### 4.1 Benchmark path (what produced the 0.992)

```
wm_prob   = wm_predict(WM, X_test, dt_test, snapshots, self_ensemble)     # [N,6]
mp_te     = mean( predict(TCN), predict(LSTM), predict(GRU) )             # [N,6]

# tune the blend weight on VALIDATION
wm_va, mp_va = same on the val split
best_w = argmax over w ∈ linspace(0.2, 0.85, 14) of  PR-AUC( y_val , (w·wm_va + (1−w)·mp_va).max(axis=1) )

system_prob = best_w · wm_prob + (1 − best_w) · mp_te                     # [N,6]
threshold   = f1_threshold( y_val , (best_w·wm_va + (1−best_w)·mp_va).max(1) , max_fpr=0.15 )
```
`prog` / ATT&CK / CIs are taken from the **world model** part only. `best_w`
typically lands ≈ 0.35–0.55.

### 4.2 Deploy path (`registry.load_system_predictor`)

`_SystemPredictor` loads `SENTINEL-WM` + `system_members` via the uniform
`load_predictor`, and blends with a **fixed** `wm_weight` (default 0.55) — no
per-request val tuning:
`blend = w·wm_prob + (1−w)·mean(member_probs)`.

### 4.3 Why blend at all

The world model and the members have PR-AUC 0.957 vs ~0.98 individually but their
*errors are on different windows*. A convex combination of decorrelated,
individually-calibrated probability estimates is a **variance reduction**: the
ensemble PR-AUC rises to 0.992 while every component keeps its own progression /
rollout / explanation surface. Tuning `w` on validation (a 14-point grid, not
gradient descent) means the composition adds **no parameters** and cannot overfit
the test set.

---

## 5. End-to-end forward (system, inference)

1. **Ingest** → `X_scaled [B,12,53]`, `dt [B,12]` (`inference/pipeline.py` for the
   backend; `sequences.npz` for the benchmark).
2. **A – world model**: for the base checkpoint and each snapshot,
   `sigmoid(SentinelWorldModel(X,dt)["attack_logits_k"])`; average → `wm_prob [B,6]`.
   `argmax` of the base model's `prog_logits_k` → progression state. In the
   backend, `forward_sim.simulate_anchor` additionally runs `rollout(M=50)` for
   the CIs and calls `attack_stages.assess_forecast` per step for the ATT&CK
   phase + confidence + rationale.
3. **B/C/D – members**: `sigmoid(TCN/LSTM/GRU(X,dt)["attack_logits_k"])`;
   mean → `member_prob [B,6]`.
4. **Blend**: `system_prob = w·wm_prob + (1−w)·member_prob`.
5. **Decide**: `alert = system_prob.max(axis=1) ≥ threshold`;
   `lead_time = (K − first_k_over_threshold + 1)·10 s`.
6. **Emit**: per-horizon `{attack_prob, attack_ci, progression_state,
   progression_dist, attck:{tactic, technique_ids, kill_chain_phase, confidence,
   rationale}}` + `driving_features` (`explain.gradient_x_input` +
   `attention_saliency` on the world model).

Cost: 1 GRU-encoder + 4 MLP heads + 3 member forwards ≈ 1.1 M params, a few ms per
anchor on CPU; +50 MC latent steps through the STN for the rollout in
`forward_sim`.

---

## 6. End-to-end backward (what actually trains)

There is **no** end-to-end backward pass for "the system". Training is the union
of five independent optimisations, run by `research.py` in order:

| # | what | loss | optimiser | trains |
|---|---|---|---|---|
| 1 | SSL pretrain (`pretrain.py`) | masked-window MSE | AdamW `lr·3`, 40 ep | encoder weights |
| 2 | world model stage A (`train.py`) | `3.0·focal_bce(attack)` | AdamW `2.5e-4`, cos-warm-restart | encoder + `horizon_attack` + `attack_head` |
| 3 | world model stage B | `3.0·l_attack + 0.5·l_prog + 0.10·l_next + 1e-5·l_kl` | AdamW `1.25e-4`, cos | + STN + `horizon_prog` + `prog_head` |
| 4 | TCN / LSTM / GRU (`nn_common.py`), ×3 | `focal_bce(attack) + 0.5·CE(prog)` | AdamW `8e-4`, cos-warm-restart | each member's body + head |
| 5 | **blend weight `w`** | — | 14-point grid on val PR-AUC | *no gradient* — a selection |

Everything is `grad_clip=1.0`, batch 256, balanced sampler ≥ 30 % positive,
train-only augmentation, early stop on `val_f1 + 0.5·prog_acc` (patience 40).
Snapshots for the WM self-ensemble are saved during step 3.

---

## 7. Parameter budget

| component | params | role |
|---|---|---|
| **A · SENTINEL-WM world model** | **532,042** | forecast + progression + rollout + ATT&CK substrate |
| &nbsp;&nbsp;GRU encoder | 344,800 | history → `z_t` |
| &nbsp;&nbsp;State-Transition Net | 103,680 | `z_t` → `N(μ,σ)` over `z_{t+1}` |
| &nbsp;&nbsp;`horizon_attack` / `horizon_prog` | 26,726 / 30,590 | direct K-step heads (benchmarked) |
| &nbsp;&nbsp;`attack_head` / `prog_head` (shared) | 12,961 / 13,285 | rollout heads |
| **B · TCN** | **172,644** | decorrelated now-cast prob |
| **C · LSTM** | **247,716** | decorrelated now-cast prob |
| **D · GRU** | **191,140** | decorrelated now-cast prob |
| blend weight `w` | 0 | selected on validation |
| **SENTINEL-WM (system) total** | **1,143,542** | |

(WM snapshots for the self-ensemble are copies of A, ~532 k each on disk, not
extra trainable parameters.)

---

## 8. Hyper-parameter reference (`config.py`)

| group | knob | value | why |
|---|---|---|---|
| Window | `window_seconds` / `stride` | 10 / 10 | proposal 10-s state windows, non-overlapping |
| Sequence | `history L` / `horizon K` | 12 / 6 | 120 s of context → forecast +10…+60 s |
| Model | `d_model` | 160 | fits ~8.5 k sequences without overfitting |
| | `n_heads` | 4 | attention read-out + (unused) transformer |
| | `stn_hidden` | 160 | = `d_model` |
| | `dropout` | 0.15 | |
| | `encoder` | `gru` | sample-efficiency on short sequences |
| | `focal_gamma` | 1.0 | gentle — γ>1.5 worsens ECE |
| Loss weights | `w_attack` | 3.0 | benchmarked output dominates |
| | `w_progression` | 0.5 | secondary target |
| | `w_next_state` | 0.10 | light STN signal |
| | `w_kl` | 1e-5 | keep σ meaningful, don't collapse μ |
| Train | `epochs` | 150 (ran 104) | + early stop patience 40 |
| | `lr` / `weight_decay` | 2.5e-4 / 2e-5 | AdamW |
| | `warm_restart_period` | 30 | `T_0`; snapshot troughs land here |
| | `two_stage_frac` | 0.20 | stage A budget (plateaus fast) |
| | `balanced_sampler_min_pos` | 0.30 | ≥30 % positive windows / batch |
| | `ssl_pretrain` / `ssl_epochs` | True / 40 | label-free manifold learning |
| | `distill` | **False** | every classical teacher weaker than WM F1\* |
| | `snapshot_ensemble` | True | cheap decorrelated variance cut |
| | `self_ensemble_direct_w` | **1.0** | rollout hurts F1 on a now-cast test; stays in `forward_sim` |
| | `system_members` | `(tcn, lstm, gru)` | strongest decorrelated sequence models, no side inputs |
| | `mc_samples` | 50 | rollout MC latents |
| | `target_fpr` | 0.05 | feeds the FPR budget of the best-F1 calibrator (`max(0.15, 3·target_fpr)`) |

---

## 9. Design rationale in one paragraph

The task is to **forecast attacker progression**, which needs a model of how
network *state evolves* — hence a world model (`encoder → STN → heads`), not a
classifier. The GRU encoder is chosen for sample efficiency on short sequences;
the probabilistic STN gives a Monte-Carlo rollout with real confidence bands and
a progression-state trajectory, which the ATT&CK layer turns into a phase +
confidence per horizon step. Two head families keep the direct forecast stable
while still training the rollout heads. SSL pretraining, focal loss, a
pos-weighted BCE, a ≥30 %-positive sampler, augmentation and a two-stage schedule
all exist to survive a ~3–6 % positive class. The **system** row then blends the
world model with three architecturally decorrelated sequence models (TCN/LSTM/GRU)
using a validation-tuned scalar — a zero-parameter variance reduction that lifts
detection PR-AUC to 0.992 while preserving the world model's unique forecast /
progression / explanation surface.
