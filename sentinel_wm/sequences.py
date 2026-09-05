#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  sequences.py   (PHASE 2c - sequence dataset + splits)
# -----------------------------------------------------------------------------
# Consumes state_windows.parquet and produces the tensors the models train on:
#
#   X       [N, L, F]   history of L state vectors  (RobustScaler-normalised)
#   dt      [N, L]      time_since_prev_window for each history step
#   x_next  [N, F]      the (scaled) state vector at t+1        -> STN target
#   y_atk   [N, K]      binary attack label at t+1 .. t+K       -> attack head
#   y_prog  [N, K]      progression-state index at t+1 .. t+K   -> progression head
#   y_now   [N]         attack label of the current window t    (baseline nowcast)
#   split   [N]         "train" | "val" | "test"
#   meta    frame       day, window_index, dominant_family, attck_* for eval
#
# Splitting (config.SplitConfig):
#   auto  -> "day"   if >1 day present in the data
#         -> "block" (block-interleaved) otherwise, because a single CIC-IDS-2017
#            day puts whole attack episodes on one side of a chronological cut
#            (verified on Wednesday: DoS episodes are all in the first 60%).
#   Scaler is ALWAYS fit on the training split only (proposal 7.3).
# =============================================================================
from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sentinel_wm import config as C
from sentinel_wm.state_windows import STATE_FEATURE_COLS


# -----------------------------------------------------------------------------
# split assignment  (operates on the per-window frame, before sequencing)
# -----------------------------------------------------------------------------
def assign_split(sw: pd.DataFrame, sp: C.SplitConfig, w: C.WindowConfig,
                 verbose: bool = True) -> pd.Series:
    days = sorted(sw["day"].unique())
    mode = sp.mode
    if mode == "auto":
        mode = "day" if len([d for d in days if d != "Unknown"]) > 1 else "block"
    if verbose:
        print(f"[split] mode={mode}  days={days}")

    out = pd.Series("train", index=sw.index, dtype=object)

    if mode == "day":
        for key, (day, _o, split) in C.DAY_SCHEDULE.items():
            out[sw["day"] == day] = split
        # any day not in the schedule -> train
        return out

    if mode == "chronological":
        f_tr, f_va, _ = sp.chrono_fracs
        for day, g in sw.groupby("day"):
            q = g["window_index"].rank(pct=True)
            out[g.index[q <= f_tr]] = "train"
            out[g.index[(q > f_tr) & (q <= f_tr + f_va)]] = "val"
            out[g.index[q > f_tr + f_va]] = "test"
        return out

    if mode == "family":
        fam = sw["dominant_family"]
        out[:] = "train"
        out[fam.isin(sp.family_val)] = "val"
        out[fam.isin(sp.family_test)] = "test"
        # benign windows follow the split of their temporal neighbourhood
        for day, g in sw.groupby("day"):
            lab = out.loc[g.index].replace("train", np.nan).ffill().bfill()
            benign = g.index[sw.loc[g.index, "dominant_family"] == "BENIGN"]
            out.loc[benign] = lab.loc[benign].fillna("train")
        return out

    # ---- block-interleaved (default for a single day) --------------------
    assign = list(sp.block_assignment)
    block_w = max(1, int(sp.block_minutes * 60 / w.window_seconds))
    for day, g in sw.groupby("day"):
        blk = (g["window_index"] - g["window_index"].min()) // block_w
        out.loc[g.index] = [assign[int(b) % len(assign)] for b in blk]
    return out


# -----------------------------------------------------------------------------
# sequence windowing
# -----------------------------------------------------------------------------
def _make_sequences(sw_day: pd.DataFrame, L: int, K: int, feat_cols) -> Dict:
    sw_day = sw_day.sort_values("window_index").reset_index(drop=True)
    F = sw_day[feat_cols].to_numpy(np.float32)
    dt = sw_day["time_since_prev_window"].to_numpy(np.float32)
    yA = sw_day["y_attack"].to_numpy(np.int64)
    yP = sw_day["progression_idx"].to_numpy(np.int64)
    wi = sw_day["window_index"].to_numpy(np.int64)
    split = sw_day["split"].to_numpy(object)

    n = len(sw_day)
    X, DT, XN, YA, YP, YNOW, SPL, IDX = [], [], [], [], [], [], [], []
    for t in range(L - 1, n - K):
        # contiguity guard: history + horizon must be consecutive windows
        if wi[t] - wi[t - L + 1] != L - 1:
            continue
        if wi[t + K] - wi[t] != K:
            continue
        X.append(F[t - L + 1: t + 1])
        DT.append(dt[t - L + 1: t + 1])
        XN.append(F[t + 1])
        YA.append(yA[t + 1: t + 1 + K])
        YP.append(yP[t + 1: t + 1 + K])
        YNOW.append(yA[t])
        SPL.append(split[t])            # split of the anchor window t
        IDX.append(wi[t])
    if not X:
        return {}
    return dict(
        X=np.stack(X), dt=np.stack(DT), x_next=np.stack(XN),
        y_atk=np.stack(YA), y_prog=np.stack(YP),
        y_now=np.asarray(YNOW, np.int64),
        split=np.asarray(SPL, object), window_index=np.asarray(IDX, np.int64))


def build_sequences(sw: Optional[pd.DataFrame] = None,
                    cfg: Optional[C.Config] = None,
                    out_path: Optional[str] = None,
                    verbose: bool = True) -> Dict:
    cfg = cfg or C.CONFIG
    if sw is None:
        from sentinel_wm.state_windows import load_state_windows
        sw = load_state_windows()

    sw = sw.copy()
    sw["split"] = assign_split(sw, cfg.split, cfg.window, verbose=verbose)

    L, K = cfg.sequence.history, cfg.sequence.horizon
    feat_cols = [c for c in STATE_FEATURE_COLS if c in sw.columns]

    per_day = []
    for day, g in sw.groupby("day", sort=False):
        d = _make_sequences(g, L, K, feat_cols)
        if d:
            d["day"] = np.array([day] * len(d["X"]), dtype=object)
            per_day.append(d)
    if not per_day:
        raise RuntimeError("no sequences built - not enough contiguous windows; "
                           "lower sequence.history or window.stride_seconds")

    cat = {k: np.concatenate([d[k] for d in per_day], axis=0)
           for k in per_day[0] if k != "day"}
    cat["day"] = np.concatenate([d["day"] for d in per_day], axis=0)

    # ---- fit RobustScaler on TRAIN sequences only ----------------------
    from sklearn.preprocessing import RobustScaler
    tr = cat["split"] == "train"
    if tr.sum() == 0:
        raise RuntimeError("training split is empty - check SplitConfig")
    scaler = RobustScaler()
    flat_train = cat["X"][tr].reshape(-1, cat["X"].shape[-1])
    scaler.fit(flat_train)

    def _scale(a3):
        s = a3.shape
        return scaler.transform(a3.reshape(-1, s[-1])).reshape(s).astype(np.float32)

    cat["X"] = _scale(cat["X"])
    cat["x_next"] = scaler.transform(cat["x_next"]).astype(np.float32)
    # log1p the raw dt then leave unscaled (positional encoder handles it)
    cat["dt"] = np.log1p(np.clip(cat["dt"], 0, None)).astype(np.float32)

    cat["feature_names"] = np.array(feat_cols, dtype=object)
    cat["L"] = np.int64(L)
    cat["K"] = np.int64(K)

    out_path = out_path or C.SEQUENCE_NPZ
    np.savez_compressed(out_path, **cat)
    with open(C.SCALER_PKL, "wb") as fh:
        pickle.dump({"scaler": scaler, "feature_names": feat_cols}, fh)

    if verbose:
        for s in ("train", "val", "test"):
            m = cat["split"] == s
            if m.sum():
                print(f"[seq] {s:5s}: {m.sum():6d} seqs | "
                      f"y_atk(any k)+ = {cat['y_atk'][m].max(1).mean():.3f} | "
                      f"k1+ = {cat['y_atk'][m][:,0].mean():.3f}")
        print(f"[seq] X shape {cat['X'].shape} -> {out_path}")
        print(f"[seq] scaler -> {C.SCALER_PKL}")
    return cat


def load_sequences(path: Optional[str] = None) -> Dict:
    path = path or C.SEQUENCE_NPZ
    if not os.path.exists(path):
        return build_sequences()
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def split_arrays(seq: Dict, split: str) -> Dict:
    m = seq["split"] == split
    keys = ["X", "dt", "x_next", "y_atk", "y_prog", "y_now",
            "window_index", "day"]
    return {k: seq[k][m] for k in keys if k in seq}


if __name__ == "__main__":
    build_sequences()
