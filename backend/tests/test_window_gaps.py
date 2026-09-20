"""Unit test for window gap filling and continuous anchor sequence generation."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from app.sentinel_infer import schema as C
from app.sentinel_infer.preprocess import clean_flow_frame
from app.sentinel_infer.windows import (
    _assign_windows, _agg_windows, _fill_idle_windows, build_state_windows
)


def _make_flows_with_idle_gap() -> pd.DataFrame:
    """Create flows with an intentional 10s gap (no flows in window 5)."""
    rows = []
    t0 = 1_700_000_000.0
    flag_defaults = {
        "flag_true_syn": 1,
        "flag_true_ack": 1,
        "flag_true_rst": 0,
        "flag_true_fin": 0,
        "flag_true_psh": 0,
        "flag_true_urg": 0,
        "SYN Flag Count": 1,
        "RST Flag Count": 0,
        "ACK Flag Count": 1,
    }
    # Windows 0, 1, 2, 3, 4 have flows
    for w in range(5):
        for _ in range(3):
            r = {
                "flow_start_epoch": t0 + w * 10.0 + 1.0,
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 5000,
                "Destination Port": 80,
                "Protocol": 6,
                "Flow Duration": 100.0,
                "Flow IAT Mean": 10.0,
                "Total Fwd Packets": 2,
                "Total Backward Packets": 2,
                "Total Length of Fwd Packets": 100.0,
                "Total Length of Bwd Packets": 100.0,
                "is_attack": 0,
                "attack_family": "BENIGN",
                "Label": "BENIGN",
            }
            r.update(flag_defaults)
            rows.append(r)

    # Window 5 is EMPTY (zero traffic)
    # Windows 6 through 20 have flows
    for w in range(6, 21):
        for _ in range(3):
            r = {
                "flow_start_epoch": t0 + w * 10.0 + 1.0,
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 5000,
                "Destination Port": 80,
                "Protocol": 6,
                "Flow Duration": 100.0,
                "Flow IAT Mean": 10.0,
                "Total Fwd Packets": 2,
                "Total Backward Packets": 2,
                "Total Length of Fwd Packets": 100.0,
                "Total Length of Bwd Packets": 100.0,
                "is_attack": 0,
                "attack_family": "BENIGN",
                "Label": "BENIGN",
            }
            r.update(flag_defaults)
            rows.append(r)

    return pd.DataFrame(rows)


def test_fill_idle_windows_bridges_gap():
    df = _make_flows_with_idle_gap()
    clean = clean_flow_frame(df, verbose=False)
    w = C.CONFIG.window
    fw = _assign_windows(clean, w)
    sw_raw = _agg_windows(fw, w)

    raw_wins = set(sw_raw.reset_index()["window_index"].tolist())
    assert 5 not in raw_wins, "Window 5 should be missing from raw aggregation"

    sw_filled = _fill_idle_windows(sw_raw.reset_index(), w)
    filled_wins = sw_filled["window_index"].tolist()
    assert 5 in filled_wins, "Window 5 should be filled in sw_filled"
    assert filled_wins == list(range(21)), "Windows 0 to 20 should be strictly consecutive"


def test_build_state_windows_has_no_gaps():
    df = _make_flows_with_idle_gap()
    clean = clean_flow_frame(df, verbose=False)
    sw = build_state_windows(clean)
    wins = sw["window_index"].tolist()
    assert wins == list(range(21)), "Complete state windows must have no gaps"
    # Idle window 5 has 0 flows and 0 byte rate
    w5 = sw[sw["window_index"] == 5].iloc[0]
    assert w5["flow_count"] == 0
    assert w5["byte_rate"] == 0.0
    assert w5["dominant_family"] == "BENIGN"
