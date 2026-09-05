#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  state_windows.py   (PHASE 2 - network state construction)
# -----------------------------------------------------------------------------
# Turns the tidy flow table into the central object of the whole system:
#
#     S_t  =  f( all flows whose flow_start_epoch falls in window [t, t+W) )
#
# One row per (day, window_index).  ~50-dim numeric state vector + labels:
#     * progression_state   (NORMAL/PRE_ATTACK/ONSET/ACTIVE/CONTINUATION)  <- primary target
#     * is_attack           (0/1)                                          <- binary target
#     * attack_ratio, dominant_family                                     <- for ATT&CK layer
#     * ATT&CK assessment fields (tactic / techniques / phase / confidence)
#
# All aggregation is vectorised with pandas groupby; the only python loop is the
# episode / progression-state pass which is O(n_windows).
# =============================================================================
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sentinel_wm import config as C
from sentinel_wm import attack_stages as A


# -----------------------------------------------------------------------------
# 1. assign every flow to a (day, window_index)
# -----------------------------------------------------------------------------
def _assign_windows(df: pd.DataFrame, w: C.WindowConfig) -> pd.DataFrame:
    df = df.copy()
    parts = []
    for day, g in df.groupby("day", sort=False):
        t0 = g[C.ORDER_COL].min()
        rel = (g[C.ORDER_COL].values - t0)
        if w.stride_seconds >= w.window_seconds:
            # non-overlapping fast path
            widx = np.floor(rel / w.window_seconds).astype(np.int64)
            gg = g.assign(window_index=widx, window_start=t0 + widx * w.window_seconds)
            parts.append(gg)
        else:
            # overlapping windows: a flow can belong to several. Replicate rows.
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
                window_index=widx,
                window_start=t0 + widx * w.stride_seconds)
            parts.append(gg)
    return pd.concat(parts, ignore_index=True)


# -----------------------------------------------------------------------------
# 2. aggregate flows in a window  ->  S_t
# -----------------------------------------------------------------------------
_FLAG = {  # true-flag column -> short name used in the state vector
    "flag_true_syn": "syn", "flag_true_ack": "ack", "flag_true_rst": "rst",
    "flag_true_fin": "fin", "flag_true_psh": "psh", "flag_true_urg": "urg",
}


def _agg_windows(fw: pd.DataFrame, w: C.WindowConfig) -> pd.DataFrame:
    W = float(w.window_seconds)
    g = fw.groupby(["day", "window_index"], sort=True)

    out = pd.DataFrame(index=g.size().index)
    out["window_start"] = g["window_start"].first()

    # ---- volume / rate ------------------------------------------------------
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

    # ---- connection dynamics --------------------------------------------- --
    out["unique_src"] = g["Source IP"].nunique()
    out["unique_dst"] = g["Destination IP"].nunique()
    pair = fw["Source IP"].astype(str) + ">" + fw["Destination IP"].astype(str)
    fw = fw.assign(_pair=pair)
    gp = fw.groupby(["day", "window_index"], sort=True)
    out["unique_pairs"] = gp["_pair"].nunique()
    out["unique_dst_ports"] = gp["Destination Port"].nunique()
    # fan-out / fan-in
    fo = (fw.groupby(["day", "window_index", "Source IP"])["Destination IP"]
            .nunique().groupby(level=[0, 1]).max())
    fi = (fw.groupby(["day", "window_index", "Destination IP"])["Source IP"]
            .nunique().groupby(level=[0, 1]).max())
    out["fan_out"] = fo
    out["fan_in"] = fi
    # failed-connection proxy: flows carrying a RST / total flows
    out["failed_conn_ratio"] = (gp["flag_true_rst"].apply(lambda s: (s > 0).sum())
                                / out["flow_count"])

    # ---- TCP flag rates (per second) ----------------------------------- --
    for col, name in _FLAG.items():
        if col in fw.columns:
            out[f"{name}_rate"] = g[col].sum() / W
        else:
            out[f"{name}_rate"] = 0.0

    # ---- timing ------------------------------------------------------------ -
    out["flow_duration_mean"] = g["Flow Duration"].mean() / 1e6  # us -> s
    out["iat_mean"] = g["Flow IAT Mean"].mean()
    out["iat_std"] = g["Flow IAT Mean"].std().fillna(0.0)
    out["burstiness"] = (out["iat_std"] / (out["iat_mean"] + 1e-6))

    # ---- packet layer (Tier 2, window-aggregated) -----------------------
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

    # ---- scanning signatures (Tier 2 / behaviour) ---------------------------
    out["port_scan_entropy"] = _mean("port_scan_entropy")
    out["port_scan_seq_ratio"] = (g["port_scan_sequential_ratio"].max()
                                  if "port_scan_sequential_ratio" in fw.columns else 0.0)
    out["port_scan_max_run"] = (g["port_scan_max_sequential_run"].max()
                                if "port_scan_max_sequential_run" in fw.columns else 0.0)
    out["unique_dst_ports_per_src"] = (g["unique_dst_ports_per_src"].max()
                                       if "unique_dst_ports_per_src" in fw.columns else 0.0)

    # ---- protocol mix ----------------------------------------------------- -
    out["tcp_ratio"] = g["Protocol"].apply(lambda s: (s == 6).mean())
    out["udp_ratio"] = g["Protocol"].apply(lambda s: (s == 17).mean())

    # ---- labels (not model input) ----------------------------------------- -
    out["attack_flows"] = g["is_attack"].sum()
    out["attack_ratio"] = out["attack_flows"] / out["flow_count"]
    dom = (fw[fw["is_attack"] == 1]
           .groupby(["day", "window_index"])["attack_family"]
           .agg(lambda s: s.value_counts().idxmax()))
    out["dominant_family"] = dom
    out["dominant_family"] = out["dominant_family"].fillna("BENIGN")

    out = out.reset_index()
    return out


# -----------------------------------------------------------------------------
# 3. progression-state derivation  (episodes -> NORMAL/PRE/ONSET/ACTIVE/CONT.)
# -----------------------------------------------------------------------------
def _derive_progression(sw: pd.DataFrame, w: C.WindowConfig) -> pd.DataFrame:
    sw = sw.sort_values(["day", "window_index"]).reset_index(drop=True)
    sw["is_attack"] = ((sw["attack_flows"] >= w.min_attack_flows)
                       & (sw["attack_ratio"] >= w.min_attack_ratio)).astype(np.int8)
    states = np.array(["NORMAL"] * len(sw), dtype=object)

    for day, g in sw.groupby("day", sort=False):
        idx = g.index.to_numpy()
        atk = g["is_attack"].to_numpy()
        fam = g["dominant_family"].to_numpy()
        wi = g["window_index"].to_numpy()

        # ---- build episodes: runs of attack windows, bridging small gaps ---
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
                    j += 1  # bridge a <=gap benign hole
                else:
                    break
            episodes.append((i, j))
            i = j + 1

        # ---- assign states -------------------------------------------------
        for (s, e) in episodes:
            # PRE_ATTACK: up to `pre_attack_span` benign windows before onset,
            # only if they are contiguous in time with the onset window.
            for k in range(1, w.pre_attack_span + 1):
                p = s - k
                if p < 0 or atk[p] == 1:
                    break
                if wi[s] - wi[p] != k:      # time gap -> stop
                    break
                states[idx[p]] = "PRE_ATTACK"
            # ONSET
            states[idx[s]] = "ONSET"
            prev_fam = fam[s]
            for m in range(s + 1, e + 1):
                if atk[m] == 0:
                    states[idx[m]] = "CONTINUATION"      # bridged hole
                    continue
                if fam[m] != prev_fam and fam[m] != "BENIGN":
                    states[idx[m]] = "CONTINUATION"
                    prev_fam = fam[m]
                else:
                    states[idx[m]] = "ACTIVE"
            # trailing edge: last window of a long episode with falling ratio
            if e > s and sw.loc[idx[e], "attack_ratio"] < 0.5 * sw.loc[idx[s:e+1], "attack_ratio"].max():
                states[idx[e]] = "CONTINUATION"

    sw["progression_state"] = states
    sw["progression_idx"] = sw["progression_state"].map(C.STATE_TO_IDX).astype(np.int64)
    # binary forecasting target = "is this an attack window"
    sw["y_attack"] = sw["progression_state"].isin(
        ["ONSET", "ACTIVE", "CONTINUATION"]).astype(np.int8)
    return sw


# -----------------------------------------------------------------------------
# 4. attach ATT&CK assessment (layer A+B from attack_stages.py)
# -----------------------------------------------------------------------------
def _attach_attack_stage(sw: pd.DataFrame) -> pd.DataFrame:
    sw = sw.sort_values(["day", "window_index"]).reset_index(drop=True)
    rows = []
    for _, g in sw.groupby("day", sort=False):
        rows.extend(A.label_frame_stages(
            g["progression_state"].tolist(),
            g["dominant_family"].tolist(),
            g["attack_ratio"].tolist()))
    stg = pd.DataFrame(rows, index=sw.index)
    stg = stg.rename(columns={
        "mitre_tactic": "attck_tactic",
        "technique_ids": "attck_techniques",
        "kill_chain_phase": "attck_phase",
        "confidence": "attck_confidence",
        "rationale": "attck_rationale",
        "family_transition": "attck_family_transition",
    })
    stg["attck_techniques"] = stg["attck_techniques"].apply(
        lambda x: ";".join(x) if isinstance(x, (list, tuple)) else (x or ""))
    keep = ["attck_tactic", "attck_techniques", "attck_phase",
            "attck_confidence", "attck_rationale", "attck_family_transition"]
    return pd.concat([sw, stg[keep]], axis=1)


# -----------------------------------------------------------------------------
# 5. metadata derived from time (safe model inputs) + public API
# -----------------------------------------------------------------------------
STATE_FEATURE_COLS: List[str] = [
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


def build_state_windows(flows: Optional[pd.DataFrame] = None,
                        cfg: Optional[C.Config] = None,
                        out_path: Optional[str] = None,
                        verbose: bool = True) -> pd.DataFrame:
    cfg = cfg or C.CONFIG
    if flows is None:
        from sentinel_wm import preprocessing
        flows = preprocessing.load_clean()

    w = cfg.window
    if verbose:
        print(f"[windows] W={w.window_seconds}s stride={w.stride_seconds}s "
              f"pre_attack_span={w.pre_attack_span}")

    fw = _assign_windows(flows, w)
    sw = _agg_windows(fw, w)
    sw = _derive_progression(sw, w)
    sw = _attach_attack_stage(sw)

    # time-derived safe features
    sw = sw.sort_values(["day", "window_index"]).reset_index(drop=True)
    sw["time_since_prev_window"] = (
        sw.groupby("day")["window_start"].diff().fillna(w.window_seconds))
    sw["window_index_in_day"] = sw.groupby("day").cumcount()

    for c in STATE_FEATURE_COLS:
        if c not in sw.columns:
            sw[c] = 0.0
        sw[c] = pd.to_numeric(sw[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    out_path = out_path or C.STATE_WINDOWS_PARQUET
    sw.to_parquet(out_path, index=False)

    if verbose:
        print(f"[windows] {len(sw):,} state windows -> {out_path}")
        print("[windows] progression_state:\n"
              + sw["progression_state"].value_counts().to_string().replace("\n", "\n    "))
        print("[windows] y_attack balance: "
              f"{sw['y_attack'].mean():.3f} positive")
        print("[windows] ATT&CK phase x confidence:")
        print(sw.groupby(["attck_phase", "attck_confidence"]).size()
              .to_string().replace("\n", "\n    "))
    return sw


def load_state_windows(path: Optional[str] = None) -> pd.DataFrame:
    path = path or C.STATE_WINDOWS_PARQUET
    import os
    if not os.path.exists(path):
        return build_state_windows()
    return pd.read_parquet(path)


if __name__ == "__main__":
    build_state_windows()
