#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  config.py
# -----------------------------------------------------------------------------
# Single source of truth for:
#   * file paths / artifact locations
#   * the LOCKED feature tiers (Tier 1 flow / Tier 2 packet / Tier 3 behaviour /
#     Tier 4 metadata) exactly as described in the proposal (section 6.2)
#   * window / horizon / sequence hyper-parameters
#   * train / val / test split configuration
#
# Nothing in this file trains or transforms anything. Import it everywhere so
# that the notebooks, the CLI and the training scripts all agree on column
# names and hyper-parameters.
# =============================================================================
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# -----------------------------------------------------------------------------
# 1. Paths
# -----------------------------------------------------------------------------
# ROOT = the repo root that holds `data/`, `artifacts/`, `runs/`, `models/`.
# The package lives at <ROOT>/research/sentinel_wm/, so ROOT is two levels up
# from this file - but resolve it robustly (walk up for a repo marker) so an
# editable install or a moved checkout still finds it. Override with
# SENTINEL_WM_ROOT.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_root() -> str:
    env = os.environ.get("SENTINEL_WM_ROOT")
    if env:
        return os.path.abspath(env)
    d = PKG_DIR
    for _ in range(6):
        d = os.path.dirname(d)
        # a directory that carries repo-level markers (data/ or .git/ or the
        # sibling research/ + backend/ layout)
        if (os.path.isdir(os.path.join(d, "data"))
                or os.path.isdir(os.path.join(d, ".git"))
                or os.path.isdir(os.path.join(d, "backend"))):
            return d
    # fallback: two up from the package dir (<ROOT>/research/sentinel_wm)
    return os.path.dirname(os.path.dirname(PKG_DIR))


ROOT = _find_root()
DATA_DIR = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

# A portable, self-contained model bundle assembled by `sentinel-wm bundle`.
# Its canonical home is `<ROOT>/backend/models/` so the backend ships code AND
# models together (it never imports research code - only reads this directory).
# Override with SENTINEL_WM_MODEL_DIR.
_DEFAULT_MODEL_DIR = os.path.join(ROOT, "backend", "models")
MODEL_DIR = os.environ.get("SENTINEL_WM_MODEL_DIR") or _DEFAULT_MODEL_DIR


def model_dir() -> str:
    """Live value of the deploy-bundle dir (re-reads SENTINEL_WM_MODEL_DIR each
    call, so a test / server that sets it after import still takes effect)."""
    return os.environ.get("SENTINEL_WM_MODEL_DIR") or _DEFAULT_MODEL_DIR


def bundled(*rel: str):
    """Return `<model_dir>/<rel...>` if a deploy bundle is present and holds that
    file, else None. Readers do `path = C.bundled('world_model.pt') or
    C.WORLD_MODEL_PT` so the serving backend never touches the research tree."""
    cand = os.path.join(model_dir(), *rel)
    return cand if os.path.exists(cand) else None

# Raw labelled unified-flow CSVs produced by extraction/label_mapping.ipynb.
# `unified_AllDays_labeled.csv` holds all 5 CIC-IDS-2017 days (Mon-Fri) in one
# file with a `source_day` column, so the proposal's day-based split
# (Mon-Wed=train / Thu=val / Fri=test) is active. Swap back to the single
# `unified_Wednesday-WorkingHours_labeled.csv` for a fast single-day smoke run.
RAW_FLOW_CSVS: List[str] = [
    os.path.join(DATA_DIR, "unified_AllDays_labeled.csv"),
]
# fallback used automatically if the file above is missing
RAW_FLOW_CSVS_FALLBACK: List[str] = [
    os.path.join(DATA_DIR, "unified_Wednesday-WorkingHours_labeled.csv"),
]

# Intermediate + output artifacts
CLEAN_FLOWS_PARQUET = os.path.join(ARTIFACTS, "clean_flows.parquet")
CLEAN_FLOWS_AUG_PARQUET = os.path.join(ARTIFACTS, "clean_flows_aug.parquet")
STATE_WINDOWS_PARQUET = os.path.join(ARTIFACTS, "state_windows.parquet")
SEQUENCE_NPZ = os.path.join(ARTIFACTS, "sequences.npz")
SCALER_PKL = os.path.join(ARTIFACTS, "state_scaler.pkl")
BASELINE_DIR = os.path.join(ARTIFACTS, "baselines")
WORLD_MODEL_PT = os.path.join(ARTIFACTS, "world_model.pt")
REPORT_DIR = os.path.join(ARTIFACTS, "reports")
for _d in (BASELINE_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)


# Name of the generated-output folder under ROOT. It is NOT `research/` any more:
# `<ROOT>/research/` is now the ML SOURCE tree (package + notebooks + extraction),
# so benchmark artefacts land in `<ROOT>/runs/` to avoid the collision.
RUN_DIR_NAME = os.environ.get("SENTINEL_WM_RESEARCH_DIR_NAME", "runs")
ZEROSHOT_DIR_NAME = RUN_DIR_NAME + "_zeroshot"


def research_dir() -> str:
    """`<ROOT>/runs` unless SENTINEL_WM_RESEARCH_DIR overrides it (used to write a
    zero-shot benchmark run into runs_zeroshot/ without clobbering the primary)."""
    return os.environ.get("SENTINEL_WM_RESEARCH_DIR") or os.path.join(ROOT, RUN_DIR_NAME)


# -----------------------------------------------------------------------------
# 2. CIC-IDS-2017 day schedule  (Appendix A2 of the proposal)
# -----------------------------------------------------------------------------
# Maps a substring of the `source_file` column to (day_name, ordinal, split).
DAY_SCHEDULE = {
    "Monday":    ("Monday",    0, "train"),
    "Tuesday":   ("Tuesday",   1, "train"),
    "Wednesday": ("Wednesday", 2, "train"),
    "Thursday":  ("Thursday",  3, "val"),
    "Friday":    ("Friday",    4, "test"),
}


# -----------------------------------------------------------------------------
# 3. LOCKED feature schema
# -----------------------------------------------------------------------------
# Column names as they appear in the unified labelled CSV *after* the
# extractor's `str.strip()` normalisation (the released ISCX CSVs have leading
# spaces; the unified files here do not).

# ---- Identity: used for graph construction / windowing ONLY. NEVER a raw
#      neural-network input (proposal 6.2, risk table). ------------------------
IDENTITY_COLS: List[str] = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Protocol", "Timestamp",
]

# ---- Tier 1 : Flow features (CIC-IDS-2017 CSV / CICFlowMeter-derived) --------
TIER1_FLOW_COLS: List[str] = [
    "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd Header Length", "Bwd Header Length",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "min_seg_size_forward", "act_data_pkt_fwd",
    "Down/Up Ratio",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# ---- Tier 2 : Packet features (PCAP via extractor.py, tshark front-end) ------
TIER2_PACKET_COLS: List[str] = [
    "pkt_ttl_mean", "pkt_ttl_std", "pkt_ttl_min", "pkt_ttl_max",
    "pkt_win_mean", "pkt_win_std", "pkt_win_min", "pkt_win_max",
    "pkt_payload_min", "pkt_payload_max", "pkt_payload_var",
    "pkt_payload_skew", "pkt_payload_kurtosis", "pkt_payload_nonzero_ratio",
    "pkt_framelen_mean", "pkt_framelen_std",
    "frag_more_flag_present", "frag_dont_flag_present", "frag_max_offset",
    "retransmission_count", "syn_packet_count", "non_tcp_udp_packet_count",
    "port_scan_entropy", "unique_dst_ports_per_src",
    "port_scan_max_sequential_run", "port_scan_sequential_ratio",
]

# ---- Honest flag counters (extractor emits these alongside the scrambled
#      CICFlowMeter flag columns; prefer them for window aggregation). --------
TRUE_FLAG_COLS: List[str] = [
    "flag_true_fin", "flag_true_syn", "flag_true_rst", "flag_true_psh",
    "flag_true_ack", "flag_true_urg", "flag_true_cwr", "flag_true_ece",
]

# ---- Tier 4 : Metadata / lineage. EXCLUDED from every model input. ----------
METADATA_COLS: List[str] = [
    "source_file", "source_day", "profile", "t_min", "timestamp_window",
    "flow_start_epoch",          # ordering key only
    "Fwd Header Length.1",       # duplicated column (ISCX 85-col quirk)
    "subflow_count_raw", "packet_count",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
]

LABEL_COL = "Label"
ORDER_COL = "flow_start_epoch"          # canonical time axis (float seconds)


# -----------------------------------------------------------------------------
# 4. Progression states  (proposal 6.7) — the model's PRIMARY learned target
# -----------------------------------------------------------------------------
PROGRESSION_STATES: List[str] = [
    "NORMAL",        # 0 - all flows BENIGN
    "PRE_ATTACK",    # 1 - benign window 1..N windows before a confirmed onset
    "ONSET",         # 2 - first window of an attack episode
    "ACTIVE",        # 3 - sustained attack window
    "CONTINUATION",  # 4 - attack persists but the dominant family changed / tail
]
STATE_TO_IDX = {s: i for i, s in enumerate(PROGRESSION_STATES)}
IDX_TO_STATE = {i: s for s, i in STATE_TO_IDX.items()}


# -----------------------------------------------------------------------------
# 5. Hyper-parameters
# -----------------------------------------------------------------------------
@dataclass
class WindowConfig:
    window_seconds: int = 10          # proposal: 10 s state windows
    stride_seconds: int = 10          # 10 => non-overlapping; 5 => 50% overlap
    pre_attack_span: int = 3          # windows before onset flagged PRE_ATTACK
    episode_gap_windows: int = 2      # <=N benign windows inside an episode are bridged
    # ---- positive-label rule (tightened - see docs/technical_reference.md P2 #1)
    # a window is an ATTACK window only if it has >= min_attack_flows malicious
    # flows AND >= min_attack_ratio of its flows are malicious. Raising the ratio
    # from 0.0 removes the ~48% of "attack" windows that are < 10% malicious and
    # therefore not learnable from aggregate state.
    min_attack_flows: int = 2
    min_attack_ratio: float = 0.05
    label_smooth_windows: int = 1    # majority-vote smoothing radius (0 = off)
    add_derived_features: bool = True  # first differences + distribution entropy
    # ---- flow-level augmentation (train-only, opt-in - see flow_augment.py) ----
    # synthesises new attack episodes from TRAIN attack flows: +/- IAT jitter,
    # flow dropout + re-aggregation, destination-port shuffle, cross-day transplant
    # of rare families onto benign stretches. Every synthetic flow is tagged
    # is_synthetic=1 and lands on a `__aug_<family>__` day that assign_split pins
    # to train, so val/test stay byte-identical to a flow_augment=False run.
    flow_augment: bool = False
    flow_aug_families: tuple = ()          # () => the learnable set (never Heartbleed/Infiltration/SQLi)
    flow_aug_max_variants: int = 3         # synthetic copies per real train episode
    flow_aug_jitter_pct: float = 0.15     # +/- fractional jitter on IAT / duration / active-idle
    flow_aug_dropout_frac: float = 0.20   # fraction of an episode's attack flows dropped
    flow_aug_port_shuffle: bool = True
    flow_aug_transplant: bool = True      # cross-day transplant for rare families
    flow_aug_transplant_max_windows: int = 120  # "rare" = < this many train windows
    flow_aug_cap_frac: float = 1.0        # synthetic positive windows <= cap_frac * real train positives
    flow_aug_seed: int = 1337


@dataclass
class SequenceConfig:
    history: int = 12                 # L : input windows  [S_{t-L+1} .. S_t]
    horizon: int = 6                  # K : forecast steps  (K x window_seconds)
    # LEAKAGE GUARD: a sequence touches windows [t-L+1 .. t+K] (history + horizon).
    # If those windows are not ALL in one split, the sequence is dropped (label
    # "ignore"): otherwise a train anchor would learn val/test window labels via
    # its horizon target, and a val/test anchor would be scored on windows a
    # train anchor already trained on. Keep True.
    purge_boundary_sequences: bool = True


@dataclass
class SplitConfig:
    # "stratified" (default, `auto` -> this) -> per DAY, per attack EPISODE, cut
    #             the episode's window range 60/20/20 CHRONOLOGICALLY into
    #             train/val/test, so EVERY family with >=1 episode gets windows
    #             in all 3 splits proportionally (block interleaving left DoS
    #             GoldenEye 0-in-val). Benign windows split by `stratified_benign`
    #             ("contiguous" = per-day 60/20/20 by time, few boundaries;
    #             "block" = 5-min round-robin, many boundaries). Sequences whose
    #             history+horizon span crosses a split boundary are dropped by
    #             SequenceConfig.purge_boundary_sequences; `stratified_purge_windows`
    #             is an extra window-level guard band.
    # "block"  -> time-block-interleaved across every day: each day cut into
    #             `block_minutes` contiguous blocks, assigned train/train/train/
    #             val/test round-robin. Kept as a secondary benchmark.
    # "day"    -> strict CIC-IDS-2017 day split (Mon-Wed / Thu / Fri) = ZERO-SHOT
    #             new-attack-family test; run separately into research_zeroshot/.
    # "family" -> attack-family HOLD-OUT (train excludes family_val/family_test) =
    #             the other zero-shot benchmark.
    # "episode_chrono" -> like stratified but only around attack episodes + a
    #             sprinkle of benign; for a lead-time-focused run.
    # "chronological" -> per-day first 60/20/20 by window rank.
    # "auto"   -> "stratified".
    mode: str = "auto"
    # ---- family-stratified ----
    stratified_fracs: tuple = (0.6, 0.2, 0.2)
    # windows at the END of the train (and val) chunk of each episode, i.e. the
    # ones whose K-step forecast horizon would reach into the next split, are
    # dropped (label "ignore"). 0 = accept mild boundary leakage (like `block`).
    stratified_purge_windows: int = 0
    stratified_benign: str = "contiguous"  # "contiguous" (leakage-safe, default) | "block"
    # a whole attack episode is assigned to ONE split, cycling this pattern per
    # family (train-favoured) so families with >=3 episodes span all 3 splits.
    stratified_episode_rotation: tuple = ("train", "val", "train", "test")
    # episodes with >= this many windows are instead cut 60/20/20 internally
    # (a lone long burst still reaches every split). Default 3*(L+K)=54.
    stratified_long_episode_windows: int = 54
    # ---- block-interleaved ----
    block_minutes: int = 5
    block_assignment: tuple = ("train", "train", "train", "val", "test")
    # episode_chrono: also carry this fraction of benign windows into val/test
    # (else they become ~100% attack and FPR is unmeasurable)
    episode_chrono_benign_frac: float = 0.25
    # ---- chronological fractions ----
    chrono_fracs: tuple = (0.6, 0.2, 0.2)
    # ---- attack-family holdout ----
    family_train: tuple = ("DoS Hulk", "DoS slowloris")
    family_val: tuple = ("DoS Slowhttptest",)
    family_test: tuple = ("DoS GoldenEye", "Heartbleed")


@dataclass
class ModelConfig:
    # "gru"  -> Bi-GRU + attention read-out. Default: on ~8.5k sequences of 12
    #           short steps a GRU is far more sample-efficient than a pure
    #           Transformer (which ceilings ~3 AUROC pts lower here).
    # "transformer" -> the causal Temporal Transformer (use once data >> 10k).
    encoder: str = "gru"
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.15
    stn_hidden: int = 160
    # joint-loss weights  (proposal 6.5:  L = g*attack + d*prog + a*mse + b*KL).
    # The benchmarked output is the attack head, so it dominates; the STN gets
    # only a light next-state signal and the KL is minimal.
    w_next_state: float = 0.10
    w_kl: float = 1e-5
    w_attack: float = 3.0
    w_progression: float = 0.5
    # focal loss for the attack head (down-weights the easy majority). gamma=0
    # -> plain weighted BCE. gamma>~1.5 distorts probabilities (worse ECE) so
    # keep it gentle; PR-AUC / f1_best are the threshold-free headline numbers.
    focal_gamma: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 256
    lr: float = 2.5e-4
    weight_decay: float = 2e-5
    grad_clip: float = 1.0
    seed: int = 1337
    early_stop_patience: int = 40    # > warm_restart_period so it survives a restart dip
    warm_restart_period: int = 30    # CosineAnnealingWarmRestarts T_0 (0 => plain cosine)
    balanced_sampler_min_pos: float = 0.30   # >=30% positive windows per batch (0 => off)
    augment: bool = True             # train-only sequence augmentation (see augment.py)
    two_stage: bool = True           # world model: pretrain encoder+attack head, then add STN/prog
    two_stage_frac: float = 0.20     # stage A budget (it plateaus fast; 0.35 wasted epochs)
    two_stage_freeze_encoder: bool = False  # True => stage B freezes the encoder at its stage-A best
                                            # (caps at stage-A ceiling; usually just start-from-best is enough)
    # --- getting SENTINEL-WM to #1 (docs/technical_reference.md Part 2) --------
    ssl_pretrain: bool = True         # masked-window encoder pre-training (pretrain.py)
    ssl_epochs: int = 40
    # Distillation OFF: on the leakage-safe `stratified` split EVERY classical
    # `__seq` teacher (best is mlp_sklearn__seq, F1* 0.62) is weaker than the
    # world model's own F1* (0.90), so KD can only drag it down. Re-enable and
    # point `distill_teachers` at the NN zoo (lstm/tcn) once `_teacher_probs`
    # learns to load `.pt` checkpoints (train.py TODO).
    distill: bool = False
    distill_teachers: tuple = ("mlp_sklearn__seq", "xgboost__seq",
                               "logistic_regression__seq")
    w_distill: float = 0.5            # KD loss weight (soft BCE to the teacher probs)
    snapshot_ensemble: bool = True    # save a checkpoint at each warm-restart trough, average at inference
    self_ensemble: bool = True        # snapshot-average the direct multi-horizon head at inference
    # weight on the DIRECT multi-horizon head vs the K-step MC rollout in the
    # benchmarked self-ensemble. 1.0 = direct head only (the rollout regresses
    # toward the base rate on a now-cast-dominated test and hurts F1). The
    # forward-simulation product (forward_sim.simulate_anchor) always uses the
    # full rollout regardless; this only affects the benchmark prob. Lower it
    # (-> 0.6) on a lead-time-focused split where onsets are in the test set.
    self_ensemble_direct_w: float = 1.0
    # the deployed "SENTINEL-WM system" = the world-model self-ensemble blended
    # (val-tuned weight) with the STRONGEST sequence models. The tree boosters
    # collapse on this split, so blend with the neural zoo instead.
    system_blend_teachers: bool = True
    system_members: tuple = ("tcn", "lstm", "gru")
    device: str = "auto"             # "auto" | "cpu" | "cuda"
    mc_samples: int = 50             # M : Monte-Carlo rollout samples
    alert_threshold: float = 0.7     # P(attack) alert gate (proposal 6.6)
    target_fpr: float = 0.05         # threshold auto-calibrated to this FPR


@dataclass
class Config:
    window: WindowConfig = field(default_factory=WindowConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
