#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  flow_augment.py   -  TRAIN-ONLY flow-level data augmentation
# -----------------------------------------------------------------------------
# The proposal (section 7.5) asks for flow-level augmentation of the scarce
# attack episodes. Tensor-space jitter (augment.py) only perturbs already-built
# sequences; this module synthesises *genuinely new* attack episodes from the
# raw flow table so the state-window aggregation produces new, realistic
# positive windows:
#
#   (a) timing jitter    x (1 +/- flow_aug_jitter_pct) on every IAT / duration /
#                        active-idle column of the episode's attack flows
#   (b) flow dropout     drop flow_aug_dropout_frac of the episode's attack
#                        flows, then re-aggregate -> a lower-ratio positive (or
#                        a legitimate hard negative; not forced positive)
#   (c) port shuffle     a consistent permutation of the episode's distinct
#                        Destination Port values, kept in [1, 65535]
#   (d) cross-day        for rare families, prepend a benign window stretch
#       transplant       sampled from a DIFFERENT training day so the family is
#                        seen in a new benign environment
#
# Provenance & leakage safety:
#   * every synthetic flow gets  is_synthetic = 1
#   * every synthetic episode lands on a dedicated day  "__aug_<family>__"
#   * sequences.assign_split rule #1 pins is_synthetic==1 windows to TRAIN, and
#     val/test membership is decided only over real windows -> a flow_augment
#     True vs False run has byte-identical val/test (day, window_index) sets.
#
# Output: artifacts/clean_flows_aug.parquet = concat(real flows, synthetic).
# state_windows.build_state_windows / graph_windows.build_graph_windows load it
# automatically when WindowConfig.flow_augment is True and it exists.
# =============================================================================
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sentinel_wm import config as C

# families with too few real flows to ever form a learnable window -> never augment
_UNLEARNABLE = {"Heartbleed", "Infiltration", "Web Attack SQL Injection"}

# columns scaled by the timing-jitter transform (all in microseconds / counts)
_JITTER_COLS = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


# -----------------------------------------------------------------------------
def _bucket_windows(flows: pd.DataFrame, w: C.WindowConfig) -> pd.Series:
    """per-day floor((t - t0) / window_seconds) - same rule as _assign_windows."""
    def _wi(s: pd.Series) -> pd.Series:
        return np.floor((s - s.min()) / w.window_seconds).astype(np.int64)
    return flows.groupby("day")[C.ORDER_COL].transform(_wi)


def _runs(wis, gap: int):
    """maximal runs of window indices, bridging holes of <= gap empty windows."""
    wis = np.sort(np.unique(np.asarray(wis, np.int64)))
    if len(wis) == 0:
        return []
    runs, s, prev = [], int(wis[0]), int(wis[0])
    for x in wis[1:]:
        x = int(x)
        if x - prev <= gap + 1:
            prev = x
        else:
            runs.append((s, prev))
            s = prev = x
    runs.append((s, prev))
    return runs


# -----------------------------------------------------------------------------
def flow_split_labels(flows: pd.DataFrame,
                      cfg: Optional[C.Config] = None) -> pd.Series:
    """train / val / test / ignore label per FLOW, obtained by building the real
    state windows, running sequences.assign_split, and mapping the per-window
    label back onto the flows by (day, window_index)."""
    cfg = cfg or C.CONFIG
    from sentinel_wm.state_windows import build_state_windows
    from sentinel_wm.sequences import assign_split

    real = flows
    if "is_synthetic" in flows.columns:
        real = flows[flows["is_synthetic"] == 0]
    scratch = os.path.join(C.ARTIFACTS, "_sw_realsplit.parquet")
    sw = build_state_windows(flows=real.drop(columns=[c for c in ("is_synthetic",)
                                                      if c in real.columns]),
                             cfg=cfg, out_path=scratch, verbose=False)
    sw["split"] = assign_split(sw, cfg.split, cfg.window, verbose=False).values
    key = sw.set_index(["day", "window_index"])["split"].to_dict()
    try:
        os.remove(scratch)
    except OSError:
        pass
    wi = _bucket_windows(flows, cfg.window).to_numpy()
    day = flows["day"].to_numpy(object)
    lab = [key.get((d, int(k)), "train") for d, k in zip(day, wi)]
    return pd.Series(lab, index=flows.index, dtype=object)


# -----------------------------------------------------------------------------
def _benign_pad(flows: pd.DataFrame, exclude_day: str, need: int,
                rng: np.random.Generator) -> Optional[pd.DataFrame]:
    """a contiguous `need`-window benign slice from a different real day, taken
    only from TRAIN windows (never val/test - that would be a leak into the
    train-pinned synthetic day)."""
    train_only = flows[flows["_split"] == "train"] if "_split" in flows.columns else flows
    cand = [d for d in train_only["day"].unique()
            if d != exclude_day and not str(d).startswith("__aug_")]
    rng.shuffle(cand)
    for d in cand:
        g = train_only[train_only["day"] == d]
        if g.empty:
            continue
        # windows that are NOT wholly benign-and-train are off limits
        blocked = set(int(x) for x in g.loc[g["is_attack"] == 1, "_wi"].unique())
        gwins = set(int(x) for x in g["_wi"].unique())
        wmin, wmax = min(gwins), max(gwins)
        starts = [x for x in range(wmin, wmax - need)
                  if all((x + k) in gwins and (x + k) not in blocked
                         for k in range(need))]
        if starts:
            st = int(rng.choice(starts))
            return g[(g["_wi"] >= st) & (g["_wi"] < st + need)].copy()
    return None


def _make_variant(sl: pd.DataFrame, fam: str, rng: np.random.Generator,
                  w: C.WindowConfig) -> Optional[pd.DataFrame]:
    """one augmented copy of an episode context slice (benign context kept,
    attack flows of `fam` perturbed). Timestamps rebased to start at 0."""
    sl = sl.copy()
    is_atk = sl["is_attack"].to_numpy() == 1
    # keep only THIS family's attack flows as attack; drop stray other-family ones
    other = is_atk & (sl["attack_family"].to_numpy(object) != fam)
    if other.any():
        sl = sl[~other].copy()
        is_atk = sl["is_attack"].to_numpy() == 1
    if int(is_atk.sum()) < 2:
        return None

    p = float(w.flow_aug_jitter_pct)
    ops = {"jitter"}
    if rng.random() < 0.6:
        ops.add("dropout")
    if w.flow_aug_port_shuffle and rng.random() < 0.7:
        ops.add("ports")

    # (a) per-flow timing jitter on the attack flows
    for c in _JITTER_COLS:
        if c in sl.columns:
            col = sl[c].to_numpy(float).copy()
            fac = 1.0 + rng.uniform(-p, p, size=len(sl))
            col[is_atk] = np.clip(col[is_atk] * fac[is_atk], 0.0, None)
            sl[c] = col

    # (c) consistent destination-port permutation over the attack flows
    if "ports" in ops and "Destination Port" in sl.columns:
        dp = sl["Destination Port"].to_numpy().copy()
        uniq = np.unique(dp[is_atk])
        if len(uniq) > 1:
            perm = rng.permutation(uniq)
            mp = {int(a): int(np.clip(b, 1, 65535)) for a, b in zip(uniq, perm)}
            for i in np.where(is_atk)[0]:
                dp[i] = mp.get(int(dp[i]), int(dp[i]))
            sl["Destination Port"] = dp

    # (b) flow dropout on the attack flows (keep >= 2)
    if "dropout" in ops:
        atk_idx = np.where(is_atk)[0]
        ndrop = min(len(atk_idx) - 2,
                    int(round(w.flow_aug_dropout_frac * len(atk_idx))))
        if ndrop > 0:
            drop = rng.choice(atk_idx, size=ndrop, replace=False)
            keep = np.ones(len(sl), bool)
            keep[drop] = False
            sl = sl.iloc[keep].copy()
    if int((sl["is_attack"].to_numpy() == 1).sum()) < 2:
        return None

    # rebase timestamps to start at 0 (caller offsets onto the aug-day cursor)
    ep = sl[C.ORDER_COL].to_numpy(float)
    rel = (ep - ep.min()) * (1.0 + rng.uniform(-p, p))    # global tempo jitter
    sl = sl.sort_values(C.ORDER_COL).copy()
    sl[C.ORDER_COL] = np.sort(rel)
    sl["day"] = f"__aug_{fam}__"
    if "source_day" in sl.columns:
        sl["source_day"] = f"__aug_{fam}__"
    sl["is_synthetic"] = np.int8(1)
    return sl


# -----------------------------------------------------------------------------
def build_augmented_flows(flows: Optional[pd.DataFrame] = None,
                          cfg: Optional[C.Config] = None,
                          out_path: Optional[str] = None,
                          verbose: bool = True) -> Dict:
    cfg = cfg or C.CONFIG
    w = cfg.window
    L, K = cfg.sequence.history, cfg.sequence.horizon
    rng = np.random.default_rng(int(w.flow_aug_seed))
    out_path = out_path or C.CLEAN_FLOWS_AUG_PARQUET

    if flows is None:
        from sentinel_wm import preprocessing
        flows = preprocessing.load_clean()
    flows = flows.copy()
    if "is_synthetic" not in flows.columns:
        flows["is_synthetic"] = np.int8(0)

    fams = tuple(w.flow_aug_families) or tuple(sorted(
        set(flows.loc[flows["is_attack"] == 1, "attack_family"].unique())
        - _UNLEARNABLE))
    if verbose:
        print(f"[flowaug] target families: {list(fams)}")
        print(f"[flowaug] labelling flow-level split (mode={cfg.split.mode}) ...")

    flows["_split"] = flow_split_labels(flows, cfg).values
    flows["_wi"] = _bucket_windows(flows, w).values

    train_atk = flows[(flows["_split"] == "train") & (flows["is_attack"] == 1)]
    real_pos_windows = int(train_atk.groupby(["day", "attack_family", "_wi"]).ngroups)
    cap = int(w.flow_aug_cap_frac * real_pos_windows) if w.flow_aug_cap_frac else 0

    pad_b, pad_a = L + 3, K + 3
    cursor: Dict[str, float] = {}
    blocks = []
    n_syn_pos_windows = 0
    n_episodes = 0

    # collect every candidate (day, family, episode) up front, then emit variants
    # ROUND-ROBIN by family so the cap is spread evenly (not eaten by whichever
    # family groupby happens to visit first).
    by_family: Dict[str, list] = {}
    for (day, fam), gf in train_atk.groupby(["day", "attack_family"], sort=False):
        if fam not in fams:
            continue
        day_flows = flows[flows["day"] == day]
        rare = int(gf["_wi"].nunique()) < int(w.flow_aug_transplant_max_windows)
        for (rs, re) in _runs(gf["_wi"].to_numpy(), w.episode_gap_windows):
            lo, hi = rs - pad_b, re + pad_a
            sl = day_flows[(day_flows["_wi"] >= lo) & (day_flows["_wi"] <= hi)]
            if sl.empty or int((sl["is_attack"].to_numpy() == 1).sum()) < 2:
                continue
            n_episodes += 1
            by_family.setdefault(fam, []).append((day, sl, rare, int(re - rs + 1)))

    fam_cycle = [f for f in fams if f in by_family]
    max_v = int(w.flow_aug_max_variants)
    for v in range(max_v):
        if cap and n_syn_pos_windows >= cap:
            break
        for fam in fam_cycle:
            for (day, sl, rare, ep_windows) in by_family[fam]:
                if cap and n_syn_pos_windows >= cap:
                    break
                var = _make_variant(sl, fam, rng, w)
                if var is None:
                    continue
                # (d) cross-day transplant: prepend a benign stretch from another
                # day for rare families on ~half the variants
                if rare and w.flow_aug_transplant and (v % 2 == 1):
                    bp = _benign_pad(flows, day, L + 3, rng)
                    if bp is not None and not bp.empty:
                        bp = bp.copy()
                        bep = bp[C.ORDER_COL].to_numpy(float)
                        bp[C.ORDER_COL] = np.sort(bep - bep.min())
                        gap = w.window_seconds
                        var[C.ORDER_COL] = (var[C.ORDER_COL].to_numpy(float)
                                            + float(bp[C.ORDER_COL].max()) + gap)
                        bp["day"] = f"__aug_{fam}__"
                        if "source_day" in bp.columns:
                            bp["source_day"] = f"__aug_{fam}__"
                        bp["is_synthetic"] = np.int8(1)
                        var = pd.concat([bp, var], ignore_index=True)

                aug_day = f"__aug_{fam}__"
                base = cursor.get(aug_day, 0.0)
                var[C.ORDER_COL] = var[C.ORDER_COL].to_numpy(float) + base
                span = float(var[C.ORDER_COL].max() - base)
                cursor[aug_day] = base + span + 120.0 * w.window_seconds  # big gap
                # approx positive-window contribution for the cap
                n_syn_pos_windows += ep_windows
                blocks.append(var)

    drop_tmp = ["_split", "_wi"]
    real_out = flows.drop(columns=drop_tmp)
    if not blocks:
        if verbose:
            print("[flowaug] no synthetic episodes produced; writing real flows only")
        combined = real_out
    else:
        syn = pd.concat(blocks, ignore_index=True).drop(
            columns=[c for c in drop_tmp if c in blocks[0].columns], errors="ignore")
        combined = pd.concat([real_out, syn], ignore_index=True)

    combined = combined.sort_values(["day", C.ORDER_COL],
                                    kind="mergesort").reset_index(drop=True)
    combined.to_parquet(out_path, index=False)

    n_syn_flows = int((combined["is_synthetic"] == 1).sum())
    n_syn_atk = int(((combined["is_synthetic"] == 1) & (combined["is_attack"] == 1)).sum())
    stats = dict(real_flows=int(len(real_out)), synthetic_flows=n_syn_flows,
                 synthetic_attack_flows=n_syn_atk, episodes_used=n_episodes,
                 approx_synth_pos_windows=int(n_syn_pos_windows),
                 real_train_pos_windows=real_pos_windows, cap=cap,
                 aug_days=sorted(str(d) for d in combined["day"].unique()
                                 if str(d).startswith("__aug_")),
                 out_path=out_path)
    if verbose:
        print(f"[flowaug] {n_syn_flows:,} synthetic flows "
              f"({n_syn_atk:,} attack) over {len(stats['aug_days'])} aug-days, "
              f"~{n_syn_pos_windows} new positive windows "
              f"(cap {cap} / real {real_pos_windows})")
        print(f"[flowaug] -> {out_path}  ({len(combined):,} rows total)")
    return stats


if __name__ == "__main__":
    build_augmented_flows()
