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
#   auto -> "stratified" (LEAKAGE-SAFE family stratification):
#     * benign windows          -> contiguous per-day 60/20/20 backbone
#     * a whole attack EPISODE goes to ONE split, rotating per family
#       (train-favoured) so families with >= 3 episodes span all 3 splits
#     * long episodes (>= 3*(L+K) windows) are cut 60/20/20 internally
#     Half-episode cuts are NOT used: they would force every sequence inside the
#     episode to straddle a split boundary and be purged, draining the positives.
#   Synthetic (flow-augmented) windows are ALWAYS train (rule #1, all modes).
#   Also: "block" | "day" | "family" | "episode_chrono" | "chronological".
#   RobustScaler is fit on the REAL training split only (proposal 7.3).
#
# LEAKAGE GUARD (all modes): a sequence touches windows [t-L+1 .. t+K]. If those
# are not all one split it is dropped (split="ignore") - otherwise a train anchor
# would learn val/test window labels through its horizon target, and a val/test
# anchor would be scored on a window some train anchor already trained on.
# `SequenceConfig.purge_boundary_sequences` (default True). `build_sequences`
# then asserts every retained sequence is span-pure.
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
# shared: episode scan (run of attack windows, bridging <= episode_gap_windows)
# -----------------------------------------------------------------------------
def _iter_episodes(atk: np.ndarray, wi: np.ndarray, gap: int = 0):
    """yield (start_i, end_i) index pairs over a day frame sorted by window_index."""
    i, n = 0, len(atk)
    while i < n:
        if atk[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if atk[j + 1] == 1:
                j += 1
            elif gap and (wi[min(j + 1 + gap, n - 1)] - wi[j] <= gap + 1) \
                    and atk[j + 1: j + 2 + gap].max() == 0 \
                    and j + 2 + gap <= n - 1 and atk[j + 2 + gap] == 1:
                j += 1
            else:
                break
        yield i, j
        i = j + 1


def _block_labels(g: pd.DataFrame, sp: C.SplitConfig, w: C.WindowConfig) -> np.ndarray:
    assign = list(sp.block_assignment)
    block_w = max(1, int(sp.block_minutes * 60 / w.window_seconds))
    blk = (g["window_index"] - g["window_index"].min()) // block_w
    return np.array([assign[int(b) % len(assign)] for b in blk], object)


def _contiguous_labels(g: pd.DataFrame, fracs) -> np.ndarray:
    """per-day chronological 60/20/20 by window rank - only 2 internal split
    boundaries per day, so the leakage-purge (see _make_sequences) throws away
    far fewer benign sequences than the interleaved-block backbone."""
    f_tr, f_va, _ = fracs
    q = g["window_index"].rank(pct=True).to_numpy()
    return np.where(q <= f_tr, "train",
                    np.where(q <= f_tr + f_va, "val", "test")).astype(object)


# -----------------------------------------------------------------------------
# split assignment  (operates on the per-window frame, before sequencing)
# -----------------------------------------------------------------------------
def assign_split(sw: pd.DataFrame, sp: C.SplitConfig, w: C.WindowConfig,
                 verbose: bool = True) -> pd.Series:
    days = sorted(sw["day"].unique())
    mode = sp.mode
    if mode == "auto":
        mode = "stratified"     # every family in all 3 splits, leakage-safe
    if verbose:
        print(f"[split] mode={mode}  days={days}")

    out = pd.Series("train", index=sw.index, dtype=object)

    # ---- RULE #1 (all modes): synthetic windows are ALWAYS train -----------
    syn = sw["is_synthetic"].to_numpy() == 1 if "is_synthetic" in sw.columns \
        else np.zeros(len(sw), bool)
    real_idx = sw.index[~syn]
    real = sw.loc[real_idx]
    if syn.any():
        out[sw.index[syn]] = "train"
        # decide the real windows with the same logic, then splice back
        sub = assign_split(real, sp, w, verbose=False)   # real has is_synthetic all 0
        out.loc[real_idx] = sub.values
        return out

    if mode == "stratified":
        # Leakage-safe family stratification. An attack EPISODE (a burst) is the
        # atomic unit: assigning half an episode to train and half to val would
        # force every sequence inside it to straddle the cut and be purged
        # (`_make_sequences` span-purity), draining the positive class. So:
        #   * benign windows          -> contiguous per-day 60/20/20 backbone
        #   * short episodes          -> the WHOLE episode goes to one split,
        #                                rotating per family (train-favoured) so a
        #                                family with >= 3 episodes spans all splits
        #   * long episodes (>= 3*(L+K)) -> internal 60/20/20 chronological cut
        #                                (one lone burst - DDoS, Hulk - still
        #                                reaches every split; span-purity trims
        #                                its ~L+K boundary sequences)
        # Families made of a single short burst (e.g. DoS GoldenEye) end up in
        # only 1-2 splits: that is a real property of CIC-IDS-2017, reported by
        # the per-family metrics, not a bug.
        from collections import Counter
        f_tr, f_va, _ = sp.stratified_fracs
        purge = int(getattr(sp, "stratified_purge_windows", 0))
        benign_mode = getattr(sp, "stratified_benign", "contiguous")
        L_, K_ = C.CONFIG.sequence.history, C.CONFIG.sequence.horizon
        long_ep = int(getattr(sp, "stratified_long_episode_windows", 3 * (L_ + K_)))
        rot = tuple(getattr(sp, "stratified_episode_rotation",
                            ("train", "val", "train", "test")))

        day_frames = {day: g.sort_values("window_index")
                      for day, g in sw.groupby("day", sort=False)}
        # 1) benign / default backbone
        for day, g in day_frames.items():
            gi = g.index.to_numpy()
            out.loc[gi] = (_block_labels(g, sp, w) if benign_mode == "block"
                           else _contiguous_labels(g, sp.stratified_fracs))
        # 2) gather episodes per family, in time order
        fam_eps: Dict[str, list] = {}
        for day, g in day_frames.items():
            gi = g.index.to_numpy()
            wi = g["window_index"].to_numpy()
            atk = g["y_attack"].to_numpy()
            fam = g["dominant_family"].to_numpy()
            for s, e in _iter_episodes(atk, wi, w.episode_gap_windows):
                names = [f for f in fam[s:e + 1] if f not in ("BENIGN", "?")]
                dom = Counter(names).most_common(1)[0][0] if names else "BENIGN"
                fam_eps.setdefault(dom, []).append(
                    (days.index(day) if day in days else 0, int(wi[s]), gi[s:e + 1]))
        # 3) assign each episode
        for dom, eps in fam_eps.items():
            eps.sort(key=lambda t: (t[0], t[1]))
            ri = 0
            for (_d, _w0, idxs) in eps:
                nep = len(idxs)
                if nep >= long_ep:
                    n_tr = max(1, int(round(nep * f_tr)))
                    n_va = max(1, int(round(nep * f_va)))
                    lab = np.array(["test"] * nep, dtype=object)
                    lab[:n_tr] = "train"
                    lab[n_tr:n_tr + n_va] = "val"
                    if purge:
                        for cut in (n_tr, n_tr + n_va):
                            lab[max(0, cut - purge):cut] = "ignore"
                    out.loc[idxs] = lab
                else:
                    out.loc[idxs] = rot[ri % len(rot)]
                    ri += 1
        return out

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

    if mode == "episode_chrono":
        # per day: split every attack EPISODE's window range chronologically so
        # onsets land in val/test too -> Mean Lead Time becomes a live metric.
        # Also carry a fraction of benign windows into val/test so FPR stays
        # measurable.
        f_tr, f_va, _ = sp.chrono_fracs
        out[:] = "train"
        rng = np.random.default_rng(1337)
        for day, g in sw.groupby("day"):
            g = g.sort_values("window_index")
            atk = g["y_attack"].to_numpy()
            wi = g["window_index"].to_numpy()
            gi = g.index.to_numpy()
            n = len(atk)
            for i, j in _iter_episodes(atk, wi, w.episode_gap_windows):
                span = wi[j] - wi[i] + 1
                lo_pre = max(0, i - 6)               # include the run-up
                for k in range(lo_pre, min(j + 7, n)):   # + a short tail
                    frac = (wi[k] - wi[lo_pre]) / max(span + (i - lo_pre), 1)
                    out.loc[gi[k]] = ("train" if frac <= f_tr
                                      else "val" if frac <= f_tr + f_va else "test")
                i = j + 1
            # sprinkle benign windows into val/test at the same proportions
            benign = gi[atk == 0]
            pick = rng.random(len(benign)) < getattr(sp, "episode_chrono_benign_frac", 0.25)
            r2 = rng.random(pick.sum())
            lab = np.where(r2 < f_va / (f_va + (1 - f_tr - f_va) + 1e-9), "val", "test")
            out.loc[benign[pick]] = lab
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
def _make_sequences(sw_day: pd.DataFrame, L: int, K: int, feat_cols,
                    purge_boundary: bool = True) -> Dict:
    sw_day = sw_day.sort_values("window_index").reset_index(drop=True)
    F = sw_day[feat_cols].to_numpy(np.float32)
    dt = sw_day["time_since_prev_window"].to_numpy(np.float32)
    yA = sw_day["y_attack"].to_numpy(np.int64)
    yP = sw_day["progression_idx"].to_numpy(np.int64)
    wi = sw_day["window_index"].to_numpy(np.int64)
    split = sw_day["split"].to_numpy(object)
    fam = (sw_day["dominant_family"].astype(object).to_numpy()
           if "dominant_family" in sw_day.columns
           else np.array(["?"] * len(sw_day), object))
    isyn = (sw_day["is_synthetic"].to_numpy(np.int8)
            if "is_synthetic" in sw_day.columns
            else np.zeros(len(sw_day), np.int8))

    n = len(sw_day)
    n_purged = 0
    X, DT, XN, YA, YP, YNOW, SPL, IDX, FAM, ISY = ([] for _ in range(10))
    for t in range(L - 1, n - K):
        # contiguity guard: history + horizon must be consecutive windows
        if wi[t] - wi[t - L + 1] != L - 1:
            continue
        if wi[t + K] - wi[t] != K:
            continue
        # LEAKAGE GUARD: the sequence reads/forecasts windows [t-L+1 .. t+K].
        # If they are not all one split, a train anchor would train on val/test
        # labels via its horizon and a val/test anchor would be scored on
        # windows a train anchor already saw. Drop it (-> "ignore").
        syn = int(isyn[t - L + 1: t + 1 + K].max())
        if syn:
            spl = "train"                       # synthetic is always train (rule #1)
        else:
            span = set(split[t - L + 1: t + K + 1].tolist())
            if len(span) == 1:
                spl = span.pop()
            elif purge_boundary:
                spl = "ignore"
                n_purged += 1
            else:
                spl = split[t]                  # legacy: anchor split only (leaky)
        X.append(F[t - L + 1: t + 1])
        DT.append(dt[t - L + 1: t + 1])
        XN.append(F[t + 1])
        YA.append(yA[t + 1: t + 1 + K])
        YP.append(yP[t + 1: t + 1 + K])
        YNOW.append(yA[t])
        SPL.append(spl)
        IDX.append(wi[t])
        # family this anchor is forecasting: first non-benign in [t .. t+K], else BENIGN
        win = fam[t: t + 1 + K]
        nb = [f for f in win if f not in ("BENIGN", "?")]
        FAM.append(nb[0] if nb else "BENIGN")
        ISY.append(int(isyn[t - L + 1: t + 1 + K].max()))
    if not X:
        return {}
    return dict(
        X=np.stack(X), dt=np.stack(DT), x_next=np.stack(XN),
        y_atk=np.stack(YA), y_prog=np.stack(YP),
        y_now=np.asarray(YNOW, np.int64),
        split=np.asarray(SPL, object), window_index=np.asarray(IDX, np.int64),
        dominant_family=np.asarray(FAM, object),
        is_synthetic=np.asarray(ISY, np.int8),
        _n_purged=n_purged)


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

    purge_boundary = bool(getattr(cfg.sequence, "purge_boundary_sequences", True))
    per_day = []
    n_purged = 0
    for day, g in sw.groupby("day", sort=False):
        d = _make_sequences(g, L, K, feat_cols, purge_boundary=purge_boundary)
        if d:
            n_purged += int(d.pop("_n_purged", 0))
            d["day"] = np.array([day] * len(d["X"]), dtype=object)
            per_day.append(d)
    if not per_day:
        raise RuntimeError("no sequences built - not enough contiguous windows; "
                           "lower sequence.history or window.stride_seconds")

    cat = {k: np.concatenate([d[k] for d in per_day], axis=0)
           for k in per_day[0] if k != "day"}
    cat["day"] = np.concatenate([d["day"] for d in per_day], axis=0)

    # ---- INDEPENDENT leakage assertion --------------------------------
    # every window a retained (non-ignore, non-synthetic) sequence reads or
    # forecasts, [anchor-L+1 .. anchor+K], must carry that sequence's split.
    if purge_boundary:
        wsplit = sw.set_index(["day", "window_index"])["split"].to_dict()
        keep = (cat["split"] != "ignore")
        if "is_synthetic" in cat:
            keep &= (cat["is_synthetic"] == 0)
        aday = cat["day"][keep]
        awi = cat["window_index"][keep].astype(np.int64)
        aspl = cat["split"][keep]
        bad = 0
        for off in range(-(L - 1), K + 1):
            got = np.array([wsplit.get((d, int(x) + off), s)
                            for d, x, s in zip(aday, awi, aspl)], object)
            bad += int((got != aspl).sum())
        if bad:
            raise RuntimeError(
                f"leakage: {bad} span-window/split mismatches across retained "
                f"sequences - purge logic is broken")
        if verbose:
            print(f"[seq] leakage check OK: {keep.sum()} sequences, span-pure "
                  f"({n_purged} boundary sequences purged -> 'ignore')")

    # ---- fit RobustScaler on REAL TRAIN sequences only ----------------
    from sklearn.preprocessing import RobustScaler
    tr = cat["split"] == "train"
    if tr.sum() == 0:
        raise RuntimeError("training split is empty - check SplitConfig")
    real_tr = tr & (cat.get("is_synthetic", np.zeros(len(tr), np.int8)) == 0)
    if real_tr.sum() == 0:                       # all-synthetic edge case
        real_tr = tr
    scaler = RobustScaler()
    flat_train = cat["X"][real_tr].reshape(-1, cat["X"].shape[-1])
    scaler.fit(flat_train)

    # leakage guard: no synthetic anchor may land in val/test
    if "is_synthetic" in cat:
        bad = int(((cat["is_synthetic"] == 1) & (cat["split"] != "train")).sum())
        if bad:
            raise RuntimeError(f"{bad} synthetic sequences leaked into val/test")

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
        for s in ("train", "val", "test", "ignore"):
            m = cat["split"] == s
            if m.sum():
                print(f"[seq] {s:6s}: {m.sum():6d} seqs | "
                      f"y_atk(any k)+ = {cat['y_atk'][m].max(1).mean():.3f} | "
                      f"k1+ = {cat['y_atk'][m][:,0].mean():.3f}")
        if "is_synthetic" in cat and cat["is_synthetic"].sum():
            n_syn = int(((cat["is_synthetic"] == 1) & tr).sum())
            print(f"[seq] train = {int(real_tr.sum())} real + {n_syn} synthetic")
        print(f"[seq] X shape {cat['X'].shape} -> {out_path}")
        print(f"[seq] scaler (real train only) -> {C.SCALER_PKL}")
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
            "window_index", "day", "dominant_family", "is_synthetic"]
    return {k: seq[k][m] for k in keys if k in seq}


if __name__ == "__main__":
    build_sequences()
