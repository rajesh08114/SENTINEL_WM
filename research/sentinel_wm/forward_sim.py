#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  forward_sim.py   (PHASE 5 - infiltration prediction engine)
# -----------------------------------------------------------------------------
# K-step Monte-Carlo forward simulation from a current traffic snapshot
# (proposal 6.6). For one anchor window it returns, per horizon step k:
#     * P(attack at t+k)  with 95% CI          (attack head over MC latents)
#     * predicted progression state + full distribution
#     * ATT&CK-aligned phase + confidence      (attack_stages.assess_forecast)
#     * top driving features                   (explain.top_features, optional)
#
# Load order:
#     model   <- artifacts/world_model.pt
#     data    <- artifacts/sequences.npz   (use --split test / an index)
#             OR a raw unified CSV via --csv  (runs the full pipeline first)
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from sentinel_wm import config as C
from sentinel_wm import attack_stages as A
from sentinel_wm.models import build_model


# -----------------------------------------------------------------------------
def load_checkpoint(device: str = "cpu"):
    ckpt = torch.load(C.WORLD_MODEL_PT, map_location=device, weights_only=False)
    cfg = C.CONFIG
    cfg.sequence.horizon = ckpt["sequence"]["K"]
    model = build_model(ckpt["config"]["n_features"], cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# -----------------------------------------------------------------------------
def simulate_anchor(model, x: np.ndarray, dt: np.ndarray,
                    ckpt: Dict, meta: Optional[Dict] = None,
                    M_samples: int = 50, device: str = "cpu",
                    feature_vec: Optional[np.ndarray] = None,
                    explainer=None) -> Dict:
    """x: [L, F]   dt: [L]   -> one forecast dict."""
    K = ckpt["sequence"]["K"]
    W = C.CONFIG.window.window_seconds
    xb = torch.as_tensor(x[None], dtype=torch.float32, device=device)
    dtb = torch.as_tensor(dt[None], dtype=torch.float32, device=device)
    roll = model.rollout(xb, dtb, K=K, M=M_samples)

    thr = ckpt.get("alert_threshold", 0.7)
    meta = meta or {}
    fam_hint = meta.get("dominant_family_hint", "BENIGN")
    prev_fam = meta.get("prev_family")

    horizon = []
    first_alert_k = None
    for k in range(K):
        p = float(roll["attack_prob"][0, k])
        state_probs = roll["prog_prob"][0, k].tolist()
        stage = A.assess_forecast(state_probs, horizon_k=k + 1,
                                  dominant_family_hint=fam_hint,
                                  prev_family=prev_fam, attack_prob=p)
        if first_alert_k is None and p >= thr:
            first_alert_k = k + 1
        horizon.append(dict(
            k=k + 1, horizon_seconds=(k + 1) * W,
            attack_prob=p,
            attack_ci=[float(roll["attack_ci_lo"][0, k]),
                       float(roll["attack_ci_hi"][0, k])],
            attack_std=float(roll["attack_std"][0, k]),
            progression_state=C.IDX_TO_STATE[int(roll["prog_state"][0, k])],
            progression_dist={C.IDX_TO_STATE[i]: round(v, 4)
                              for i, v in enumerate(state_probs)},
            attck=stage.to_dict()))

    result = dict(
        meta=meta,
        alert_threshold=thr,
        alert=bool(first_alert_k is not None),
        lead_time_seconds=(0 if first_alert_k is None
                           else (K - first_alert_k + 1) * W),
        first_alert_k=first_alert_k,
        max_attack_prob=float(roll["attack_prob"][0].max()),
        horizon=horizon,
    )
    if explainer is not None and feature_vec is not None:
        try:
            result["driving_features"] = explainer(x, dt, feature_vec)
        except Exception as e:                       # pragma: no cover
            result["driving_features"] = {"error": str(e)}
    return result


# -----------------------------------------------------------------------------
def simulate_split(split: str = "test", limit: Optional[int] = None,
                   device: str = "cpu", with_explain: bool = False) -> List[Dict]:
    from sentinel_wm.sequences import load_sequences
    model, ckpt = load_checkpoint(device)
    seq = load_sequences()
    m = seq["split"] == split
    idx = np.where(m)[0]
    if limit:
        idx = idx[:limit]

    explainer = None
    if with_explain:
        try:
            from sentinel_wm import explain
            explainer = explain.make_world_model_explainer(model, seq, device)
        except Exception as e:
            print(f"[sim] explainer unavailable: {e}")

    # family context of the PRESENT window - the best hint for "if this turns
    # into an attack, what kind" (attack_stages needs it to name an ATT&CK phase)
    fam_lookup = {}
    try:
        import pandas as pd
        sw = pd.read_parquet(C.STATE_WINDOWS_PARQUET)
        fam_lookup = {(str(r.day), int(r.window_index)): str(r.dominant_family)
                      for r in sw.itertuples()}
    except Exception:
        pass

    feats = list(seq["feature_names"])
    out = []
    for i in idx:
        day_i, wi_i = str(seq["day"][i]), int(seq["window_index"][i])
        fam_hint = fam_lookup.get((day_i, wi_i), "BENIGN")
        meta = dict(split=split, day=day_i, window_index=wi_i,
                    y_now=int(seq["y_now"][i]),
                    dominant_family_hint=fam_hint,
                    y_atk_true=seq["y_atk"][i].tolist())
        res = simulate_anchor(
            model, seq["X"][i], seq["dt"][i], ckpt, meta,
            M_samples=C.CONFIG.train.mc_samples, device=device,
            feature_vec=seq["X"][i][-1], explainer=explainer)
        out.append(res)
    return out


# -----------------------------------------------------------------------------
def print_timeline(results: List[Dict], max_rows: int = 40):
    W = C.CONFIG.window.window_seconds
    shown = [r for r in results if r["alert"]] or results
    print(f"\n{'win':>6} {'now':>3} | " + " ".join(f"+{(k+1)*W:>3}s" for k in range(len(results[0]['horizon']))) + "  | phase@peak (conf)")
    print("-" * 78)
    for r in shown[:max_rows]:
        probs = " ".join(f"{h['attack_prob']:>5.2f}" for h in r["horizon"])
        peak = max(r["horizon"], key=lambda h: h["attack_prob"])
        tag = "ALERT" if r["alert"] else "     "
        print(f"{r['meta'].get('window_index','?'):>6} "
              f"{r['meta'].get('y_now','?'):>3} | {probs}  | "
              f"{peak['attck']['kill_chain_phase']:<16} "
              f"{peak['attck']['confidence']:<7} {tag}")


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="K-step forward simulation")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--index", type=int, default=None,
                   help="single anchor: row index within the split")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--explain", action="store_true")
    p.add_argument("--out", default=os.path.join(C.REPORT_DIR, "forward_sim.json"))
    args = p.parse_args()

    results = simulate_split(args.split, args.limit, args.device, args.explain)
    if args.index is not None:
        results = [results[args.index]]
        print(json.dumps(results[0], indent=2))
    else:
        print_timeline(results)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    n_alert = sum(r["alert"] for r in results)
    print(f"\n[sim] {len(results)} anchors, {n_alert} alerts -> {args.out}")


if __name__ == "__main__":
    main()
