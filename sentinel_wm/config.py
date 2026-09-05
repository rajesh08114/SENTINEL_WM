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
# ROOT = repo root (the parent of this package directory). Override with the
# SENTINEL_WM_ROOT environment variable if the package is installed elsewhere.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SENTINEL_WM_ROOT", os.path.dirname(PKG_DIR))
DATA_DIR = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

# Raw labelled unified-flow CSVs produced by extraction/label_mapping.ipynb.
# Add more day files here as they are extracted; the day-based split (see
# SplitConfig) activates automatically once >1 day is present.
RAW_FLOW_CSVS: List[str] = [
    os.path.join(DATA_DIR, "unified_Wednesday-WorkingHours_labeled.csv"),
]

# Intermediate + output artifacts
CLEAN_FLOWS_PARQUET = os.path.join(ARTIFACTS, "clean_flows.parquet")
STATE_WINDOWS_PARQUET = os.path.join(ARTIFACTS, "state_windows.parquet")
SEQUENCE_NPZ = os.path.join(ARTIFACTS, "sequences.npz")
SCALER_PKL = os.path.join(ARTIFACTS, "state_scaler.pkl")
BASELINE_DIR = os.path.join(ARTIFACTS, "baselines")
WORLD_MODEL_PT = os.path.join(ARTIFACTS, "world_model.pt")
REPORT_DIR = os.path.join(ARTIFACTS, "reports")
for _d in (BASELINE_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)


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
    "source_file", "profile", "t_min", "timestamp_window",
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
    min_attack_flows: int = 1        # flows needed to call a window "attack"
    min_attack_ratio: float = 0.0    # OR ratio threshold (0 => pure count rule)


@dataclass
class SequenceConfig:
    history: int = 10                 # L : input windows  [S_{t-L+1} .. S_t]
    horizon: int = 6                  # K : forecast steps  (K x window_seconds)


@dataclass
class SplitConfig:
    # "auto"  -> day-based if >1 day present else block-interleaved
    # "day" | "block" | "family" | "chronological"
    mode: str = "auto"
    # ---- block-interleaved (single-day) ----
    block_minutes: int = 5
    block_assignment: tuple = ("train", "train", "train", "val", "test")
    # ---- chronological fractions ----
    chrono_fracs: tuple = (0.6, 0.2, 0.2)
    # ---- attack-family holdout ----
    family_train: tuple = ("DoS Hulk", "DoS slowloris")
    family_val: tuple = ("DoS Slowhttptest",)
    family_test: tuple = ("DoS GoldenEye", "Heartbleed")


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    stn_hidden: int = 128
    # joint-loss weights  (proposal 6.5:  L = a*mse + b*KL + g*attack + d*prog)
    w_next_state: float = 1.0
    w_kl: float = 1e-3
    w_attack: float = 1.0
    w_progression: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 1337
    early_stop_patience: int = 8
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
