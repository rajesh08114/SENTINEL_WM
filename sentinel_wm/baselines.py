#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  baselines.py   (PHASE 3 - comparison floor)
# -----------------------------------------------------------------------------
# The proposal (10.2 / 12 Tier-1) requires a NON-temporal baseline trained on
# the SAME features, so the world model's temporal advantage is measurable.
#
# Baselines here:
#   * Logistic Regression  (the mandated primary baseline)
#   * Random Forest         (stronger tree baseline)
#   * "persistence"          (predict A_{t+k} = A_t) - trivial reference
#
# Input  : the CURRENT state window S_t only  (the last row of each sequence's
#          history) -> no temporal modelling by construction.
# Target : A_{t+k} for k = 1..K  (one classifier per horizon).
# Output : artifacts/baselines/{model}_metrics.json  +  fitted estimators.
# =============================================================================
from __future__ import annotations

import json
import os
import pickle
from typing import Dict

import numpy as np

from sentinel_wm import config as C
from sentinel_wm import metrics as M
from sentinel_wm.sequences import load_sequences


def _flatten_current(seq: Dict) -> np.ndarray:
    """S_t = last window of the (already scaled) history sequence."""
    return seq["X"][:, -1, :]


def _fit_one(name, est_factory, Xtr, Ytr, Xva, Yva, Xte, Yte,
             window_seconds, target_fpr):
    K = Ytr.shape[1]
    per_horizon, estimators, val_probs, test_probs = [], [], [], []
    for k in range(K):
        ytr = Ytr[:, k]
        est = est_factory()
        if len(np.unique(ytr)) < 2:
            # degenerate horizon (no positives in train) - constant predictor
            p_va = np.full(len(Xva), ytr.mean(), float)
            p_te = np.full(len(Xte), ytr.mean(), float)
            est = ("constant", float(ytr.mean()))
        else:
            est.fit(Xtr, ytr)
            p_va = est.predict_proba(Xva)[:, 1]
            p_te = est.predict_proba(Xte)[:, 1]
        estimators.append(est)
        val_probs.append(p_va)
        test_probs.append(p_te)

    val_probs = np.stack(val_probs, 1)
    test_probs = np.stack(test_probs, 1)

    # calibrate a single alert threshold on VAL (max over horizons), FPR target
    thr = M.calibrate_threshold(Yva.max(1), val_probs.max(1), target_fpr)

    per_horizon = M.horizon_table(Yte, test_probs, window_seconds, thr)
    any_k = M.binary_scores(Yte.max(1), test_probs.max(1), thr)
    cal = M.expected_calibration_error(Yte[:, 0], test_probs[:, 0])
    return dict(
        model=name, threshold=thr,
        any_horizon=any_k, per_horizon=per_horizon,
        brier_k1=M.brier_score(Yte[:, 0], test_probs[:, 0]),
        ece_k1=cal["ece"]), estimators, test_probs


def run_baselines(cfg: C.Config = None, verbose: bool = True) -> Dict:
    cfg = cfg or C.CONFIG
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    seq = load_sequences()
    W = int(seq["L"]) and cfg.window.window_seconds
    K = int(seq["K"])

    def part(split):
        m = seq["split"] == split
        return _flatten_current({"X": seq["X"][m]}), seq["y_atk"][m], seq["window_index"][m], seq["y_now"][m]

    Xtr, Ytr, _, _ = part("train")
    Xva, Yva, _, _ = part("val")
    Xte, Yte, WIte, NOWte = part("test")
    if verbose:
        print(f"[baseline] train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
              f"F={Xtr.shape[1]} K={K}")

    factories = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=cfg.train.seed),
    }

    results = {}
    for name, fac in factories.items():
        res, ests, test_probs = _fit_one(
            name, fac, Xtr, Ytr, Xva, Yva, Xte, Yte,
            cfg.window.window_seconds, cfg.train.target_fpr)
        lt = M.lead_time(WIte, NOWte, test_probs.max(1),
                         cfg.window.window_seconds, res["threshold"], K)
        res["lead_time"] = lt
        results[name] = res
        with open(os.path.join(C.BASELINE_DIR, f"{name}.pkl"), "wb") as fh:
            pickle.dump(ests, fh)
        if verbose:
            print(M.summarise(f"[{name}] any-k", res["any_horizon"]))
            print(f"           MLT={lt['mean_lead_time_s']:.1f}s  "
                  f"detect={lt['detection_rate']:.2f}  "
                  f"FA={lt['false_alarm_rate']:.3f}")

    # persistence reference
    pers_pred = np.repeat(NOWte[:, None], K, axis=1).astype(float)
    pr = M.horizon_table(Yte, pers_pred, cfg.window.window_seconds, 0.5)
    results["persistence"] = dict(model="persistence", threshold=0.5,
                                  any_horizon=M.binary_scores(Yte.max(1), pers_pred.max(1), 0.5),
                                  per_horizon=pr)

    out = os.path.join(C.BASELINE_DIR, "baseline_metrics.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    if verbose:
        print(f"[baseline] metrics -> {out}")
    return results


if __name__ == "__main__":
    run_baselines()
