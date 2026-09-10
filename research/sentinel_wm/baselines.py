#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  baselines.py   (PHASE 3 - classical ML comparison zoo)
# -----------------------------------------------------------------------------
# The proposal (10.2 / 12 Tier-1) requires NON-temporal baselines trained on the
# SAME features so the world model's temporal advantage is measurable. This
# module now trains a whole zoo of classical models, in two input regimes:
#
#   input="window"    S_t only            [N, F]      -> "no temporal context"
#   input="sequence"  flatten S_{t-L+1..t} [N, L*F]   -> "same info as the WM"
#
# Models (sklearn always available; xgboost / lightgbm used if importable):
#   logistic_regression   random_forest   extra_trees   hist_gradient_boosting
#   gradient_boosting      mlp_sklearn     linear_svc    knn   gaussian_nb
#   xgboost                lightgbm
#   + a non-trained `persistence` reference (A_{t+k} = A_t)
#
# Each model = K independent binary classifiers (one per horizon step). A single
# alert threshold is calibrated on validation to FPR <= target. Fitted
# estimators + a meta sidecar are written to research/models/classical/ so the
# future serving app can load them; metrics go to artifacts/baselines/.
# =============================================================================
from __future__ import annotations

import json
import os
import pickle
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from sentinel_wm import config as C
from sentinel_wm import metrics as M
from sentinel_wm.sequences import load_sequences

# optional gradient-boosting libraries
try:
    import xgboost as _xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    import lightgbm as _lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


# -----------------------------------------------------------------------------
# model registry
# -----------------------------------------------------------------------------
def _factories(seed: int) -> Dict[str, Callable]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    f: Dict[str, Callable] = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5, solver="lbfgs"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.07, max_iter=300,
            l2_regularization=1.0, random_state=seed,
            early_stopping=True, validation_fraction=0.1),   # balanced via sample_weight
        "mlp_sklearn": lambda: MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256,
            learning_rate_init=8e-4, max_iter=80, early_stopping=True,
            n_iter_no_change=8, random_state=seed),
        "linear_svc": lambda: LinearSVC(
            class_weight="balanced", C=0.5, max_iter=4000),   # _proba() sigmoids
        "knn": lambda: KNeighborsClassifier(n_neighbors=25, weights="distance",
                                            n_jobs=-1),
        "gaussian_nb": lambda: GaussianNB(),
    }
    if _HAS_XGB:
        f["xgboost"] = lambda: _xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=seed)
    if _HAS_LGB:
        f["lightgbm"] = lambda: _lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, force_col_wise=True,
            n_jobs=1, random_state=seed, verbose=-1)   # n_jobs=1: Win stability
    return f


# window input only (no value / too slow flattened)
_WINDOW_ONLY = {"knn", "gaussian_nb", "linear_svc", "extra_trees"}
# also fit on the flat L*F sequence -> a temporal-aware classical baseline
_DEFAULT_SEQUENCE = {"logistic_regression", "random_forest",
                     "hist_gradient_boosting", "mlp_sklearn",
                     "xgboost", "lightgbm"}


# -----------------------------------------------------------------------------
# feature builders
# -----------------------------------------------------------------------------
def _X_window(seq: Dict, mask: np.ndarray) -> np.ndarray:
    return seq["X"][mask][:, -1, :]                       # [N, F]


def _X_sequence(seq: Dict, mask: np.ndarray) -> np.ndarray:
    x = seq["X"][mask]                                    # [N, L, F]
    return x.reshape(x.shape[0], -1)                      # [N, L*F]


def _proba(est, X) -> np.ndarray:
    if isinstance(est, tuple):                            # ("constant", p)
        return np.full(len(X), est[1], float)
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel()
    d = est.decision_function(X)                          # LinearSVC path
    return 1.0 / (1.0 + np.exp(-d))


# -----------------------------------------------------------------------------
# fit one model  (K per-horizon classifiers)
# -----------------------------------------------------------------------------
def _fit_model(name: str, factory: Callable, input_kind: str,
               parts: Dict, cfg: C.Config) -> Tuple[Dict, list, np.ndarray]:
    Xtr, Ytr = parts["train"]
    Xva, Yva = parts["val"]
    Xte, Yte = parts["test"]
    K = Ytr.shape[1]

    ests, va_p, te_p = [], [], []
    t0 = time.perf_counter()
    for k in range(K):
        ytr = Ytr[:, k]
        if len(np.unique(ytr)) < 2:
            est = ("constant", float(ytr.mean()))
        else:
            est = factory()
            # only weight the estimators that DON'T already take class_weight
            # (LR/RF/ET/SVC/LGBM set it in the factory; HGB/XGB/GNB/kNN don't)
            has_cw = "class_weight" in getattr(est, "get_params", lambda: {})()
            if has_cw:
                est.fit(Xtr, ytr)
            else:
                pos = max(ytr.mean(), 1e-6)
                sw = np.where(ytr == 1, 0.5 / pos, 0.5 / (1 - pos))
                try:
                    est.fit(Xtr, ytr, sample_weight=sw)
                except TypeError:
                    est.fit(Xtr, ytr)
        ests.append(est)
        va_p.append(_proba(est, Xva))
        te_p.append(_proba(est, Xte))
    fit_s = time.perf_counter() - t0

    va_p = np.stack(va_p, 1)
    te_p = np.stack(te_p, 1)
    thr = M.calibrate_threshold(Yva.max(1), va_p.max(1), cfg.train.target_fpr)

    per_h = M.horizon_table(Yte, te_p, cfg.window.window_seconds, thr)
    any_k = M.binary_scores(Yte.max(1), te_p.max(1), thr)
    cal = M.expected_calibration_error(Yte[:, 0], te_p[:, 0])
    res = dict(model=name, input_kind=input_kind, threshold=float(thr),
               any_horizon=any_k, per_horizon=per_h,
               brier_k1=M.brier_score(Yte[:, 0], te_p[:, 0]),
               ece_k1=cal["ece"], fit_seconds=round(fit_s, 2),
               n_features=int(Xtr.shape[1]))
    return res, ests, te_p


# -----------------------------------------------------------------------------
# public entry point
# -----------------------------------------------------------------------------
def run_baselines(cfg: C.Config = None, verbose: bool = True,
                  models: List[str] = None,
                  sequence_input: bool = True) -> Dict:
    cfg = cfg or C.CONFIG
    seq = load_sequences()
    K = int(seq["K"])
    facs = _factories(cfg.train.seed)
    names = models or list(facs)

    def mask(s):
        return seq["split"] == s

    wi_te = seq["window_index"][mask("test")]
    now_te = seq["y_now"][mask("test")]
    y_te = seq["y_atk"][mask("test")]

    os.makedirs(C.BASELINE_DIR, exist_ok=True)
    model_dir = os.path.join(C.research_dir(), "models", "classical")
    os.makedirs(model_dir, exist_ok=True)

    if verbose:
        print(f"[baseline] models={names}")
        print(f"[baseline] xgboost={_HAS_XGB} lightgbm={_HAS_LGB} | "
              f"train/val/test = {mask('train').sum()}/{mask('val').sum()}/"
              f"{mask('test').sum()}  K={K}")

    results: Dict[str, Dict] = {}
    for name in names:
        if name not in facs:
            print(f"[baseline] skip unknown '{name}'"); continue
        kinds = ["window"]
        if sequence_input and name in _DEFAULT_SEQUENCE and name not in _WINDOW_ONLY:
            kinds.append("sequence")
        for kind in kinds:
            build = _X_window if kind == "window" else _X_sequence
            parts = {s: (build(seq, mask(s)), seq["y_atk"][mask(s)])
                     for s in ("train", "val", "test")}
            try:
                res, ests, te_p = _fit_model(name, facs[name], kind, parts, cfg)
            except Exception as e:                        # keep the zoo going
                print(f"[baseline] {name}/{kind} FAILED: {e}")
                continue
            lt = M.lead_time(wi_te, now_te, te_p.max(1),
                             cfg.window.window_seconds, res["threshold"], K)
            res["lead_time"] = lt
            tag = name if kind == "window" else f"{name}__seq"
            results[tag] = res

            with open(os.path.join(model_dir, f"{tag}.pkl"), "wb") as fh:
                pickle.dump(ests, fh)
            with open(os.path.join(model_dir, f"{tag}.meta.json"), "w") as fh:
                json.dump(dict(name=tag, family="classical", input_kind=kind,
                               horizon=K, n_features=res["n_features"],
                               threshold=res["threshold"],
                               feature_names=list(seq["feature_names"]),
                               metrics={"f1": res["any_horizon"]["f1"],
                                        "auroc": res["any_horizon"]["auroc"],
                                        "mean_lead_time_s": lt["mean_lead_time_s"]}),
                          fh, indent=2)
            if verbose:
                print(M.summarise(f"[{tag:26s}]", res["any_horizon"])
                      + f"  MLT={lt['mean_lead_time_s']:.0f}s "
                        f"det={lt['detection_rate']:.2f} ({res['fit_seconds']}s)")

    # ---- persistence reference ------------------------------------------
    pers = np.repeat(now_te[:, None], K, axis=1).astype(float)
    results["persistence"] = dict(
        model="persistence", input_kind="none", threshold=0.5,
        any_horizon=M.binary_scores(y_te.max(1), pers.max(1), 0.5),
        per_horizon=M.horizon_table(y_te, pers, cfg.window.window_seconds, 0.5),
        lead_time=M.lead_time(wi_te, now_te, pers.max(1),
                              cfg.window.window_seconds, 0.5, K))

    out = os.path.join(C.BASELINE_DIR, "baseline_metrics.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    if verbose:
        print(f"[baseline] {len(results)} model variants -> {out}")
        print(f"[baseline] fitted estimators -> {model_dir}")
    return results


if __name__ == "__main__":
    run_baselines()
