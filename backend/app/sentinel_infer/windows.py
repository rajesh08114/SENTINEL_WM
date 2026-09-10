"""10-second network-state aggregation. Vendored from
research/sentinel_wm/state_windows.py - the aggregation must stay byte-for-byte
equivalent or the scaler + model see different features than they were trained on.
Stripped: the parquet round-trip, the flow_augment source, verbose prints.
`build_state_windows` takes an in-memory flows DataFrame and returns a DataFrame.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema as C
from . import attack_stages as A


def _assign_windows(df: pd.DataFrame, w: C.WindowConfig) -> pd.DataFrame:
    df = df.copy()
    parts = []
    for day, g in df.groupby("day", sort=False):
        t0 = g[C.ORDER_COL].min()
        rel = (g[C.ORDER_COL].values - t0)
        if w.stride_seconds >= w.window_seconds:
            widx = np.floor(rel / w.window_seconds).astype(np.int64)
            gg = g.assign(window_index=widx, window_start=t0 + widx * w.window_seconds)
            parts.append(gg)
        else:
            reps = []
            first = np.floor((rel - w.window_seconds) / w.stride_seconds).astype(np.int64) + 1
            first = np.clip(first, 0, None)
            last = np.floor(rel / w.stride_seconds).astype(np.int64)
            for k, (f_i, l_i) in enumerate(zip(first, last)):
                for wi in range(int(f_i), int(l_i) + 1):
                    reps.append((k, wi))
            if not reps:
                continue
            ridx = np.array([r[0] for r in reps])
            widx = np.array([r[1] for r in reps])
            gg = g.iloc[ridx].assign(
                window_index=widx, window_start=t0 + widx * w.stride_seconds)
            parts.append(gg)
    return pd.concat(parts, ignore_index=True)


_FLAG = {
    "flag_true_syn": "syn", "flag_true_ack": "ack", "flag_true_rst": "rst",
    "flag_true_fin": "fin", "flag_true_psh": "psh", "flag_true_urg": "urg",
}


def _agg_windows(fw: pd.DataFrame, w: C.WindowConfig) -> pd.DataFrame:
    W = float(w.window_seconds)
    g = fw.groupby(["day", "window_index"], sort=True)

    out = pd.DataFrame(index=g.size().index)
    out["window_start"] = g["window_start"].first()

    out["flow_count"] = g.size()
    tot_pkts = g["Total Fwd Packets"].sum() + g["Total Backward Packets"].sum()
    tot_bytes = (g["Total Length of Fwd Packets"].sum()
                 + g["Total Length of Bwd Packets"].sum())
    out["packet_count"] = tot_pkts
    out["byte_count"] = tot_bytes
    out["packet_rate"] = tot_pkts / W
    out["byte_rate"] = tot_bytes / W
    out["fwd_bwd_ratio"] = (g["Total Fwd Packets"].sum()
                            / (g["Total Backward Packets"].sum() + 1.0))

    out["unique_src"] = g["Source IP"].nunique()
    out["unique_dst"] = g["Destination IP"].nunique()
    pair = fw["Source IP"].astype(str) + ">" + fw["Destination IP"].astype(str)
    fw = fw.assign(_pair=pair)
    gp = fw.groupby(["day", "window_index"], sort=True)
    out["unique_pairs"] = gp["_pair"].nunique()
    out["unique_dst_ports"] = gp["Destination Port"].nunique()
    fo = (fw.groupby(["day", "window_index", "Source IP"])["Destination IP"]
            .nunique().groupby(level=[0, 1]).max())
    fi = (fw.groupby(["day", "window_index", "Destination IP"])["Source IP"]
            .nunique().groupby(level=[0, 1]).max())
    out["fan_out"] = fo
    out["fan_in"] = fi
    out["failed_conn_ratio"] = (gp["flag_true_rst"].apply(lambda s: (s > 0).sum())
                                / out["flow_count"])

    for col, name in _FLAG.items():
        if col in fw.columns:
            out[f"{name}_rate"] = g[col].sum() / W
        else:
            out[f"{name}_rate"] = 0.0

    out["flow_duration_mean"] = g["Flow Duration"].mean() / 1e6
    out["iat_mean"] = g["Flow IAT Mean"].mean()
    out["iat_std"] = g["Flow IAT Mean"].std().fillna(0.0)
    out["burstiness"] = (out["iat_std"] / (out["iat_mean"] + 1e-6))

    def _mean(col, default=0.0):
        return g[col].mean() if col in fw.columns else pd.Series(default, index=out.index)

    out["ttl_mean"] = _mean("pkt_ttl_mean")
    out["ttl_var"] = (g["pkt_ttl_mean"].var().fillna(0.0)
                      if "pkt_ttl_mean" in fw.columns else 0.0)
    out["tcp_window_mean"] = _mean("pkt_win_mean")
    out["tcp_window_std"] = (g["pkt_win_mean"].std().fillna(0.0)
                             if "pkt_win_mean" in fw.columns else 0.0)
    out["payload_var_mean"] = _mean("pkt_payload_var")
    out["payload_nonzero_ratio"] = _mean("pkt_payload_nonzero_ratio")
    out["payload_skew_mean"] = _mean("pkt_payload_skew")
    out["payload_kurt_mean"] = _mean("pkt_payload_kurtosis")
    out["retransmission_rate"] = (g["retransmission_count"].sum() / W
                                  if "retransmission_count" in fw.columns else 0.0)
    out["frag_present_ratio"] = (
        _mean("frag_more_flag_present") + _mean("frag_dont_flag_present"))

    out["port_scan_entropy"] = _mean("port_scan_entropy")
    out["port_scan_seq_ratio"] = (g["port_scan_sequential_ratio"].max()
                                  if "port_scan_sequential_ratio" in fw.columns else 0.0)
    out["port_scan_max_run"] = (g["port_scan_max_sequential_run"].max()
                                if "port_scan_max_sequential_run" in fw.columns else 0.0)
    out["unique_dst_ports_per_src"] = (g["unique_dst_ports_per_src"].max()
                                       if "unique_dst_ports_per_src" in fw.columns else 0.0)

    out["tcp_ratio"] = g["Protocol"].apply(lambda s: (s == 6).mean())
    out["udp_ratio"] = g["Protocol"].apply(lambda s: (s == 17).mean())

    def _shannon(series_vals):
        v = np.asarray(series_vals, float)
        v = v[v > 0]
        if v.size == 0:
            return 0.0
        p = v / v.sum()
        return float(-(p * np.log2(p)).sum())

    out["dstport_entropy"] = gp["Destination Port"].apply(
        lambda s: _shannon(s.value_counts().values))
    out["dstip_entropy"] = gp["Destination IP"].apply(
        lambda s: _shannon(s.value_counts().values))
    out["srcip_entropy"] = gp["Source IP"].apply(
        lambda s: _shannon(s.value_counts().values))
    out["flowsize_entropy"] = g.apply(lambda d: _shannon(np.histogram(
        np.log1p(d["Total Length of Fwd Packets"].to_numpy(float)
                 + d["Total Length of Bwd Packets"].to_numpy(float)),
        bins=16)[0]))

    out["is_synthetic"] = np.int8(0)
    out["attack_flows"] = g["is_attack"].sum()
    out["attack_ratio"] = out["attack_flows"] / out["flow_count"]
    dom = (fw[fw["is_attack"] == 1]
           .groupby(["day", "window_index"])["attack_family"]
           .agg(lambda s: s.value_counts().idxmax()))
    out["dominant_family"] = dom
    out["dominant_family"] = out["dominant_family"].fillna("BENIGN")
    return out.reset_index()


def _derive_progression(sw: pd.DataFrame, w: C.WindowConfig) -> pd.DataFrame:
    sw = sw.sort_values(["day", "window_index"]).reset_index(drop=True)
    raw = ((sw["attack_flows"] >= w.min_attack_flows)
           & (sw["attack_ratio"] >= w.min_attack_ratio)).astype(np.int8)
    r = int(getattr(w, "label_smooth_windows", 0))
    if r > 0:
        sm = raw.copy().to_numpy()
        for _day, g in sw.groupby("day", sort=False):
            v = raw.to_numpy()[g.index]
            o = v.copy()
            for i in range(len(v)):
                lo, hi = max(0, i - r), min(len(v), i + r + 1)
                o[i] = 1 if v[lo:hi].mean() >= 0.5 else 0
            sm[g.index] = o
        sw["is_attack"] = sm.astype(np.int8)
    else:
        sw["is_attack"] = raw
    states = np.array(["NORMAL"] * len(sw), dtype=object)

    for day, g in sw.groupby("day", sort=False):
        idx = g.index.to_numpy()
        atk = g["is_attack"].to_numpy()
        fam = g["dominant_family"].to_numpy()
        wi = g["window_index"].to_numpy()

        episodes: List[Tuple[int, int]] = []
        i, n = 0, len(atk)
        while i < n:
            if atk[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < n:
                if atk[j + 1] == 1:
                    j += 1
                elif (j + 1 + w.episode_gap_windows < n
                      and atk[j + 1: j + 2 + w.episode_gap_windows].max() == 0
                      and (j + 2 + w.episode_gap_windows) < n
                      and atk[j + 2 + w.episode_gap_windows] == 1
                      and (wi[j + 2 + w.episode_gap_windows] - wi[j]) <= w.episode_gap_windows + 1):
                    j += 1
                else:
                    break
            episodes.append((i, j))
            i = j + 1

        for (s, e) in episodes:
            for k in range(1, w.pre_attack_span + 1):
                p = s - k
                if p < 0 or atk[p] == 1:
                    break
                if wi[s] - wi[p] != k:
                    break
                states[idx[p]] = "PRE_ATTACK"
            states[idx[s]] = "ONSET"
            prev_fam = fam[s]
            for m in range(s + 1, e + 1):
                if atk[m] == 0:
                    states[idx[m]] = "CONTINUATION"
                    continue
                if fam[m] != prev_fam and fam[m] != "BENIGN":
                    states[idx[m]] = "CONTINUATION"
                    prev_fam = fam[m]
                else:
                    states[idx[m]] = "ACTIVE"
            if e > s and sw.loc[idx[e], "attack_ratio"] < 0.5 * sw.loc[idx[s:e + 1], "attack_ratio"].max():
                states[idx[e]] = "CONTINUATION"

    sw["progression_state"] = states
    sw["progression_idx"] = sw["progression_state"].map(C.STATE_TO_IDX).astype(np.int64)
    sw["y_attack"] = sw["progression_state"].isin(
        ["ONSET", "ACTIVE", "CONTINUATION"]).astype(np.int8)
    return sw


def _attach_attack_stage(sw: pd.DataFrame) -> pd.DataFrame:
    sw = sw.sort_values(["day", "window_index"]).reset_index(drop=True)
    rows = []
    for _, g in sw.groupby("day", sort=False):
        rows.extend(A.label_frame_stages(
            g["progression_state"].tolist(),
            g["dominant_family"].tolist(),
            g["attack_ratio"].tolist()))
    stg = pd.DataFrame(rows, index=sw.index).rename(columns={
        "mitre_tactic": "attck_tactic", "technique_ids": "attck_techniques",
        "kill_chain_phase": "attck_phase", "confidence": "attck_confidence",
        "rationale": "attck_rationale", "family_transition": "attck_family_transition",
    })
    stg["attck_techniques"] = stg["attck_techniques"].apply(
        lambda x: ";".join(x) if isinstance(x, (list, tuple)) else (x or ""))
    keep = ["attck_tactic", "attck_techniques", "attck_phase",
            "attck_confidence", "attck_rationale", "attck_family_transition"]
    return pd.concat([sw, stg[keep]], axis=1)


_BASE_FEATURE_COLS: List[str] = [
    "flow_count", "packet_count", "byte_count", "packet_rate", "byte_rate",
    "fwd_bwd_ratio",
    "unique_src", "unique_dst", "unique_pairs", "unique_dst_ports",
    "fan_out", "fan_in", "failed_conn_ratio",
    "syn_rate", "ack_rate", "rst_rate", "fin_rate", "psh_rate", "urg_rate",
    "flow_duration_mean", "iat_mean", "iat_std", "burstiness",
    "ttl_mean", "ttl_var", "tcp_window_mean", "tcp_window_std",
    "payload_var_mean", "payload_nonzero_ratio", "payload_skew_mean",
    "payload_kurt_mean", "retransmission_rate", "frag_present_ratio",
    "port_scan_entropy", "port_scan_seq_ratio", "port_scan_max_run",
    "unique_dst_ports_per_src",
    "tcp_ratio", "udp_ratio",
    "time_since_prev_window", "window_index_in_day",
]
_ENTROPY_FEATURE_COLS: List[str] = [
    "dstport_entropy", "dstip_entropy", "srcip_entropy", "flowsize_entropy",
]
_DELTA_SOURCE = ["packet_rate", "byte_rate", "syn_rate", "rst_rate",
                 "unique_dst", "unique_dst_ports", "fan_out", "failed_conn_ratio"]
_DELTA_FEATURE_COLS: List[str] = [f"d_{c}" for c in _DELTA_SOURCE]

STATE_FEATURE_COLS: List[str] = (_BASE_FEATURE_COLS + _ENTROPY_FEATURE_COLS
                                 + _DELTA_FEATURE_COLS)


def build_state_windows(flows: pd.DataFrame,
                        cfg: Optional[C.InferConfig] = None) -> pd.DataFrame:
    cfg = cfg or C.CONFIG
    w = cfg.window
    fw = _assign_windows(flows, w)
    sw = _agg_windows(fw, w)
    sw = _derive_progression(sw, w)
    sw = _attach_attack_stage(sw)

    sw = sw.sort_values(["day", "window_index"]).reset_index(drop=True)
    sw["time_since_prev_window"] = (
        sw.groupby("day")["window_start"].diff().fillna(w.window_seconds))
    sw["window_index_in_day"] = sw.groupby("day").cumcount()

    if getattr(w, "add_derived_features", True):
        for c in _DELTA_SOURCE:
            if c in sw.columns:
                sw[f"d_{c}"] = sw.groupby("day")[c].diff().fillna(0.0)
        feat_cols = STATE_FEATURE_COLS
    else:
        feat_cols = _BASE_FEATURE_COLS

    for c in feat_cols:
        if c not in sw.columns:
            sw[c] = 0.0
        sw[c] = pd.to_numeric(sw[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return sw
