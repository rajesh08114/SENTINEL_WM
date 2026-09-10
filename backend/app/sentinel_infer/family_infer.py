"""Heuristic attack-family inference from a single scaled state-window vector.

When a forecast has no explicit `family_hint` (a raw CSV / PCAP / live capture
where the label is unknown), `assess_forecast` would otherwise fall back to
"Unspecified". This picks the most likely CIC-IDS family from which features are
most elevated in the (RobustScaler-)scaled window, so the ATT&CK mapping can
still name a tactic. It is explicitly a guess — callers mark the result
`inferred` and confidence is capped at Medium. When nothing is distinctive it
returns BENIGN and the caller keeps the neutral "attack forecast" wording.

Feature notes (empirical, on CIC-IDS-2017 10 s windows, RobustScaler space):
the connection-churn features (`failed_conn_ratio`, `rst_rate`,
`unique_dst_ports_per_src`) have a tiny benign IQR, so they blow up under any
attack; `packet_rate` / `byte_rate` barely move because their benign variance is
already large. So volumetric attacks are told apart by *target concentration* +
*flow count* + *deltas*, not by raw rate.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_SCAN = "PortScan"
_DOS = "DoS Hulk"
_BRUTE = "SSH-Patator"
_EXFIL = "Infiltration"
_BOT = "Bot"


def _z(vec: np.ndarray, names: Sequence[str], key: str) -> float:
    try:
        return float(vec[names.index(key)])
    except (ValueError, IndexError):
        return 0.0


def infer_family(state_vec: np.ndarray, feat_names: Sequence[str]) -> tuple[str, str]:
    """-> (cic_family, confidence in {"Medium","Low"}).  BENIGN if nothing stands out."""
    v = np.asarray(state_vec, dtype=float).ravel()
    g = lambda k: _z(v, feat_names, k)  # noqa: E731

    churn = g("failed_conn_ratio") + g("rst_rate") + g("fin_rate")
    concentrated = -g("dstip_entropy") - g("unique_dst")      # traffic aimed at one host
    fanned = g("unique_dst") + g("dstip_entropy") + g("fan_out")

    scores = {
        # scan: many distinct dst *ports* per source, SYN-heavy, tiny flows,
        # port-scan entropy up; NOT aimed at a single host's volume
        _SCAN: (
            0.9 * g("unique_dst_ports_per_src") + 1.0 * g("unique_dst_ports")
            + 1.0 * g("port_scan_entropy") + 0.8 * g("port_scan_seq_ratio")
            + 0.8 * g("syn_rate") + 0.6 * fanned
            - 0.6 * g("byte_count") - 0.5 * g("flow_duration_mean")
        ),
        # volumetric DoS: lots of flows at ONE target, heavy connection churn,
        # positive volume deltas
        _DOS: (
            1.0 * g("flow_count") + 0.9 * g("d_packet_rate") + 0.9 * g("d_byte_rate")
            + 0.9 * churn + 1.0 * concentrated + 0.5 * g("packet_rate")
            - 0.5 * g("unique_dst_ports")
        ),
        # brute force: modest flow volume, high reset/fin churn, few dst ports,
        # aimed at one host, small payloads
        _BRUTE: (
            1.0 * churn + 0.8 * concentrated + 0.6 * g("failed_conn_ratio")
            - 0.9 * g("unique_dst_ports") - 0.8 * g("flow_count")
            - 0.6 * g("byte_count") - 0.5 * g("d_byte_rate")
        ),
        # exfil / infiltration: few, long, byte-heavy flows outbound
        _EXFIL: (
            1.2 * g("byte_count") + 1.0 * g("fwd_bwd_ratio") + 0.8 * g("d_byte_rate")
            + 0.7 * g("flow_duration_mean")
            - 1.0 * g("flow_count") - 0.8 * g("packet_rate") - 0.6 * churn
        ),
        # botnet C2: low-volume, periodic, few endpoints
        _BOT: (
            0.8 * g("udp_ratio") + 0.7 * g("time_since_prev_window")
            - 0.9 * g("flow_count") - 0.8 * g("byte_count") - 0.6 * churn
            - 0.5 * g("unique_dst")
        ),
    }
    fam = max(scores, key=scores.get)
    ordered = sorted(scores.values())
    top, runner = ordered[-1], ordered[-2]
    # deliberately strict: single-window family ID is ambiguous, so only fire on
    # a blatant, well-separated signal; otherwise the caller keeps the neutral
    # progression-state-derived wording.
    if top < 7.0 or (top - runner) < 2.5:
        return "BENIGN", "Low"
    conf = "Medium" if (top - runner) >= 5.0 and top >= 12.0 else "Low"
    return fam, conf
