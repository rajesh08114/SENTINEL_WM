"""LOCKED feature schema + slim inference-time config.

Vendored subset of research/sentinel_wm/config.py. Has NO filesystem side
effects (no paths, no os.makedirs) and NO training knobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# -----------------------------------------------------------------------------
# CIC-IDS-2017 day schedule (used by preprocess.derive_day + the winsor fit)
# -----------------------------------------------------------------------------
DAY_SCHEDULE = {
    "Monday":    ("Monday",    0, "train"),
    "Tuesday":   ("Tuesday",   1, "train"),
    "Wednesday": ("Wednesday", 2, "train"),
    "Thursday":  ("Thursday",  3, "val"),
    "Friday":    ("Friday",    4, "test"),
}

# -----------------------------------------------------------------------------
# LOCKED feature column names (must match research/sentinel_wm/config.py exactly)
# -----------------------------------------------------------------------------
IDENTITY_COLS: List[str] = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Protocol", "Timestamp",
]

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

TRUE_FLAG_COLS: List[str] = [
    "flag_true_fin", "flag_true_syn", "flag_true_rst", "flag_true_psh",
    "flag_true_ack", "flag_true_urg", "flag_true_cwr", "flag_true_ece",
]

LABEL_COL = "Label"
ORDER_COL = "flow_start_epoch"

# -----------------------------------------------------------------------------
# Progression states (model's primary target)
# -----------------------------------------------------------------------------
PROGRESSION_STATES: List[str] = [
    "NORMAL", "PRE_ATTACK", "ONSET", "ACTIVE", "CONTINUATION",
]
STATE_TO_IDX = {s: i for i, s in enumerate(PROGRESSION_STATES)}
IDX_TO_STATE = {i: s for s, i in STATE_TO_IDX.items()}


# -----------------------------------------------------------------------------
# inference config (arch + windowing knobs only)
# -----------------------------------------------------------------------------
@dataclass
class WindowConfig:
    window_seconds: int = 10
    stride_seconds: int = 10
    pre_attack_span: int = 3
    episode_gap_windows: int = 2
    min_attack_flows: int = 2
    min_attack_ratio: float = 0.05
    label_smooth_windows: int = 1
    add_derived_features: bool = True


@dataclass
class SequenceConfig:
    history: int = 12
    horizon: int = 6


@dataclass
class ModelConfig:
    encoder: str = "gru"
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.15
    stn_hidden: int = 160
    w_next_state: float = 0.10
    w_kl: float = 1e-5
    w_attack: float = 3.0
    w_progression: float = 0.5
    focal_gamma: float = 1.0


@dataclass
class InferConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # inference-time self-ensemble weight (1.0 = direct head only; the K-step
    # rollout still runs in forecast.simulate_anchor for the CIs / ATT&CK).
    self_ensemble_direct_w: float = 1.0
    mc_samples: int = 50


CONFIG = InferConfig()


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
