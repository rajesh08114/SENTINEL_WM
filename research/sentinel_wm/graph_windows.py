#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  graph_windows.py   -  per-window host interaction graphs
# -----------------------------------------------------------------------------
# The state-vector pipeline reduces each 10 s window to 41 scalars. A Graph
# Attention Network needs the structure those scalars throw away: WHICH host
# talked to WHICH host. This module rebuilds, for every (day, window_index)
# already in `state_windows.parquet`, a small directed host graph:
#
#   nodes  = the busiest hosts in the window (subnet-typed), padded to N_max
#   node features (14)  = in/out flow-count, bytes, packets, SYN, RST,
#                         fan-out, distinct dst ports, is_internal, degree
#   adj [N_max, N_max]  = directed edge weight = log1p(flow count src->dst)
#
# Output: artifacts/graph_windows.npz aligned ROW-FOR-ROW with
#         state_windows.parquet (same day/window_index order), plus a
#         train-only node-feature RobustScaler. The GAT dataset in `gat.py`
#         slices L consecutive windows per anchor on the fly (cheap).
# =============================================================================
from __future__ import annotations

import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sentinel_wm import config as C
from sentinel_wm.state_windows import _assign_windows, load_state_windows

GRAPH_NPZ = os.path.join(C.ARTIFACTS, "graph_windows.npz")
GRAPH_SCALER = os.path.join(C.ARTIFACTS, "graph_node_scaler.pkl")

N_MAX = 32          # hosts kept per window (by total degree)
N_NODE_FEAT = 14

_INTERNAL_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                      "172.19.", "172.2", "172.30.", "172.31.")


def _is_internal(ip: str) -> int:
    return int(str(ip).startswith(_INTERNAL_PREFIXES))


# -----------------------------------------------------------------------------
def _window_graph(fw: pd.DataFrame) -> tuple:
    """fw = flows of ONE window. -> (node_feat [N_MAX,F], adj [N_MAX,N_MAX], mask)."""
    src = fw["Source IP"].to_numpy(str)
    dst = fw["Destination IP"].to_numpy(str)
    byt = (fw["Total Length of Fwd Packets"].to_numpy(float)
           + fw["Total Length of Bwd Packets"].to_numpy(float))
    pkt = (fw["Total Fwd Packets"].to_numpy(float)
           + fw["Total Backward Packets"].to_numpy(float))
    syn = fw["flag_true_syn"].to_numpy(float) if "flag_true_syn" in fw else np.zeros(len(fw))
    rst = fw["flag_true_rst"].to_numpy(float) if "flag_true_rst" in fw else np.zeros(len(fw))
    dpt = fw["Destination Port"].to_numpy(float)

    # rank hosts by degree (how many flows touch them)
    deg: Dict[str, int] = {}
    for a, b in zip(src, dst):
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    hosts = [h for h, _ in sorted(deg.items(), key=lambda kv: -kv[1])][:N_MAX]
    idx = {h: i for i, h in enumerate(hosts)}
    n = len(hosts)

    feat = np.zeros((N_MAX, N_NODE_FEAT), np.float32)
    adj = np.zeros((N_MAX, N_MAX), np.float32)
    mask = np.zeros(N_MAX, np.float32)
    mask[:n] = 1.0
    dports_out = [set() for _ in range(N_MAX)]
    peers_out = [set() for _ in range(N_MAX)]

    for k in range(len(fw)):
        a, b = src[k], dst[k]
        ia = idx.get(a); ib = idx.get(b)
        if ia is None or ib is None:
            continue
        adj[ia, ib] += 1.0
        # ia = source side
        feat[ia, 0] += 1                     # out flows
        feat[ia, 2] += byt[k]               # out bytes
        feat[ia, 4] += pkt[k]               # out packets
        feat[ia, 6] += syn[k]              # out SYN
        feat[ia, 8] += rst[k]             # out RST
        dports_out[ia].add(int(dpt[k])); peers_out[ia].add(b)
        # ib = destination side
        feat[ib, 1] += 1                     # in flows
        feat[ib, 3] += byt[k]              # in bytes
        feat[ib, 5] += pkt[k]              # in packets
        feat[ib, 7] += syn[k]             # in SYN
        feat[ib, 9] += rst[k]            # in RST

    for i, h in enumerate(hosts):
        feat[i, 10] = len(dports_out[i])     # distinct dst ports (fan on ports)
        feat[i, 11] = len(peers_out[i])      # fan-out (distinct peers)
        feat[i, 12] = _is_internal(h)
        feat[i, 13] = deg[h]                 # total degree
    # log-compress the heavy-tailed counts
    feat[:, :12] = np.log1p(np.clip(feat[:, :12], 0, None))
    feat[:, 13] = np.log1p(feat[:, 13])
    np.log1p(adj, out=adj)
    return feat, adj, mask


# -----------------------------------------------------------------------------
def build_graph_windows(flows: Optional[pd.DataFrame] = None,
                        cfg: Optional[C.Config] = None,
                        verbose: bool = True) -> Dict:
    cfg = cfg or C.CONFIG
    swf = load_state_windows()
    # real per-window split (NOT DAY_SCHEDULE - under `stratified` a "train day"
    # like Wednesday also holds val/test windows, which must NOT enter the
    # node-feature scaler fit).
    from sentinel_wm.sequences import assign_split
    win_split = assign_split(swf, cfg.split, cfg.window, verbose=False).to_numpy()
    sw_cols = ["day", "window_index"] + (
        ["is_synthetic"] if "is_synthetic" in swf.columns else [])
    sw = swf[sw_cols].copy()
    sw["_split"] = win_split
    if flows is None:
        from sentinel_wm import preprocessing
        aug = C.CLEAN_FLOWS_AUG_PARQUET
        if getattr(cfg.window, "flow_augment", False) and os.path.exists(aug):
            flows = pd.read_parquet(aug)
        else:
            flows = preprocessing.load_clean()

    fw = _assign_windows(flows, cfg.window)
    keep = ["day", "window_index", "Source IP", "Destination IP",
            "Destination Port", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "flag_true_syn", "flag_true_rst"]
    fw = fw[[c for c in keep if c in fw.columns]]

    order = list(sw[["day", "window_index"]].itertuples(index=False, name=None))
    row_of = {(d, int(w)): i for i, (d, w) in enumerate(order)}
    W = len(order)
    node_feat = np.zeros((W, N_MAX, N_NODE_FEAT), np.float32)
    adj = np.zeros((W, N_MAX, N_MAX), np.float32)
    node_mask = np.zeros((W, N_MAX), np.float32)

    grp = fw.groupby(["day", "window_index"], sort=False)
    done = 0
    for (d, w), g in grp:
        r = row_of.get((d, int(w)))
        if r is None:
            continue
        f, a, m = _window_graph(g)
        node_feat[r], adj[r], node_mask[r] = f, a, m
        done += 1
        if verbose and done % 2000 == 0:
            print(f"  graph windows {done}/{W}")

    # node-feature scaling fit on REAL TRAIN windows only (leakage-safe)
    from sklearn.preprocessing import RobustScaler
    is_train = (sw["_split"].to_numpy() == "train")
    if "is_synthetic" in sw.columns:                # never fit the scaler on synthetic
        is_train &= (sw["is_synthetic"].to_numpy() == 0)
    if is_train.sum() == 0:
        is_train = np.ones(W, bool)
    if verbose:
        print(f"[graph] node scaler fit on {int(is_train.sum())}/{W} real-train windows")
    scaler = RobustScaler()
    flat = node_feat[is_train][node_mask[is_train] > 0]
    scaler.fit(flat if len(flat) else node_feat.reshape(-1, N_NODE_FEAT))
    shp = node_feat.shape
    node_feat = scaler.transform(node_feat.reshape(-1, N_NODE_FEAT)).reshape(shp).astype(np.float32)
    node_feat *= node_mask[..., None]        # keep pad rows at 0

    days = sw["day"].to_numpy(str)
    wins = sw["window_index"].to_numpy(np.int64)
    np.savez_compressed(GRAPH_NPZ, node_feat=node_feat, adj=adj,
                        node_mask=node_mask, day=days, window_index=wins,
                        n_max=np.int64(N_MAX), n_node_feat=np.int64(N_NODE_FEAT))
    with open(GRAPH_SCALER, "wb") as fh:
        pickle.dump(scaler, fh)
    if verbose:
        print(f"[graph] {done}/{W} windows populated | "
              f"node_feat {node_feat.shape} adj {adj.shape} -> {GRAPH_NPZ}")
        print(f"[graph] mean nodes/window = {node_mask.sum(1).mean():.1f} "
              f"(cap {N_MAX})")
    return dict(node_feat=node_feat, adj=adj, node_mask=node_mask,
               day=days, window_index=wins)


def load_graph_windows(path: str = None) -> Dict:
    path = path or GRAPH_NPZ
    if not os.path.exists(path):
        return build_graph_windows()
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build_graph_windows()
