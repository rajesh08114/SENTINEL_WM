#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  explain.py   (PHASE 6 - explainability, proposal 6.8)
# -----------------------------------------------------------------------------
# Three complementary mechanisms, all applied post-inference:
#
#   1. SHAP feature attribution
#        - baselines : shap.LinearExplainer / TreeExplainer   (exact-ish)
#        - world model: shap.GradientExplainer on a (flattened-sequence ->
#          P(attack in next K)) wrapper.
#        If `shap` is not installed we fall back to a gradient x input
#        saliency that returns the same ranked-feature structure.
#
#   2. Temporal attention saliency
#        - final transformer block attention, summed over query positions,
#          gives per-history-window importance ("the burst 4 windows ago").
#
#   3. Gradient x input on the state sequence
#        - |d P(attack)/d S_{t-k}| . |S_{t-k}|  per (window, feature).
#
# Nothing here is required for training; it only reads a trained checkpoint.
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from sentinel_wm import config as C
from sentinel_wm.models import build_model

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# -----------------------------------------------------------------------------
# attention saliency
# -----------------------------------------------------------------------------
def attention_saliency(model, x: np.ndarray, dt: np.ndarray,
                       device: str = "cpu") -> List[float]:
    xb = torch.as_tensor(x[None], dtype=torch.float32, device=device)
    dtb = torch.as_tensor(dt[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        enc = model.encoder(xb, dtb)
    attn = enc["attn"]                       # [1, L, L]  (query, key)
    if attn is None:
        return [0.0] * x.shape[0]
    per_key = attn[0].sum(0)                 # importance of each history window
    per_key = (per_key / per_key.sum()).cpu().numpy()
    return [float(v) for v in per_key]


# -----------------------------------------------------------------------------
# gradient x input
# -----------------------------------------------------------------------------
def gradient_x_input(model, x: np.ndarray, dt: np.ndarray,
                     feature_names: List[str], device: str = "cpu",
                     horizon_reduce: str = "max") -> Dict:
    xb = torch.tensor(np.asarray(x[None]), dtype=torch.float32, device=device,
                      requires_grad=True)
    dtb = torch.as_tensor(dt[None], dtype=torch.float32, device=device)
    # cuDNN's fused RNN backward is only allowed while the module is in training
    # mode; the model is eval() here for deterministic attributions, so run this
    # one grad pass through the native (non-cuDNN) RNN kernel instead.
    with torch.backends.cudnn.flags(enabled=False):
        out = model(xb, dtb)
        logits = out["attack_logits_k"][0]                # [K]
        score = logits.max() if horizon_reduce == "max" else logits.mean()
        model.zero_grad()
        score.backward()
    g = xb.grad[0].detach().cpu().numpy()                 # [L, F]
    sal = np.abs(g) * np.abs(x)                           # [L, F]
    feat_imp = sal.sum(0)
    feat_imp = feat_imp / (feat_imp.sum() + 1e-12)
    order = np.argsort(feat_imp)[::-1]
    return dict(
        per_window=[float(v) for v in sal.sum(1) / (sal.sum() + 1e-12)],
        top_features=[dict(feature=feature_names[i],
                           importance=float(feat_imp[i]),
                           signed=float(np.sign(g[-1, i])))
                      for i in order[:10]])


# -----------------------------------------------------------------------------
# SHAP - world model
# -----------------------------------------------------------------------------
def _flatten_wrapper(model, L: int, F: int, dt_ref: np.ndarray, device: str):
    dt_t = torch.as_tensor(dt_ref[None], dtype=torch.float32, device=device)

    def f(flat: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(flat.reshape(-1, L, F), dtype=torch.float32,
                            device=device)
        d = dt_t.expand(x.shape[0], L)
        with torch.no_grad():
            logit = model(x, d)["attack_logits_k"].max(1).values
        return torch.sigmoid(logit).cpu().numpy()
    return f


def shap_world_model(model, seq: Dict, device: str = "cpu",
                     n_background: int = 64, n_explain: int = 64) -> Dict:
    feats = list(seq["feature_names"])
    L, Fdim = int(seq["L"]), len(feats)
    tr = seq["split"] == "train"
    te = seq["split"] == "test"
    Xtr = seq["X"][tr].reshape(tr.sum(), -1)
    Xte = seq["X"][te].reshape(te.sum(), -1)
    rng = np.random.default_rng(0)
    bg = Xtr[rng.choice(len(Xtr), min(n_background, len(Xtr)), replace=False)]
    ex = Xte[rng.choice(len(Xte), min(n_explain, len(Xte)), replace=False)]
    dt_ref = seq["dt"][tr].mean(0)
    f = _flatten_wrapper(model, L, Fdim, dt_ref, device)

    if _HAS_SHAP:
        expl = shap.KernelExplainer(f, bg)
        sv = np.asarray(expl.shap_values(ex, nsamples=200))
    else:
        # finite-difference fallback (same output shape as KernelExplainer)
        base = f(bg).mean()
        sv = np.zeros_like(ex)
        eps = 1e-2
        for j in range(ex.shape[1]):
            xp = ex.copy(); xp[:, j] += eps
            xm = ex.copy(); xm[:, j] -= eps
            sv[:, j] = (f(xp) - f(xm)) / (2 * eps)
        sv = sv * (ex - bg.mean(0))

    sv3 = sv.reshape(-1, L, Fdim)
    per_feature = np.abs(sv3).mean(0).sum(0)
    per_feature = per_feature / (per_feature.sum() + 1e-12)
    order = np.argsort(per_feature)[::-1]
    return dict(
        method="shap.KernelExplainer" if _HAS_SHAP else "finite-difference-fallback",
        top_features=[dict(feature=feats[i], mean_abs_shap=float(per_feature[i]))
                      for i in order[:15]],
        per_window=[float(v) for v in np.abs(sv3).mean(0).sum(1)
                    / (np.abs(sv3).mean(0).sum() + 1e-12)])


# -----------------------------------------------------------------------------
# SHAP - baselines
# -----------------------------------------------------------------------------
def shap_baseline(model_name: str = "logistic_regression",
                  horizon_k: int = 1) -> Dict:
    import pickle
    from sentinel_wm.sequences import load_sequences
    from sentinel_wm.state_windows import STATE_FEATURE_COLS
    seq = load_sequences()
    feats = [c for c in STATE_FEATURE_COLS if c in list(seq["feature_names"])]
    with open(os.path.join(C.BASELINE_DIR, f"{model_name}.pkl"), "rb") as fh:
        ests = pickle.load(fh)
    est = ests[horizon_k - 1]
    te = seq["split"] == "test"
    Xte = seq["X"][te][:, -1, :]

    if isinstance(est, tuple):
        return dict(method="constant-predictor", top_features=[])
    if _HAS_SHAP:
        try:
            expl = (shap.LinearExplainer(est, Xte) if "Logistic" in type(est).__name__
                    else shap.TreeExplainer(est))
            sv = np.asarray(expl.shap_values(Xte))
            if sv.ndim == 3:
                sv = sv[1]
        except Exception:
            sv = None
    else:
        sv = None
    if sv is None:
        coef = getattr(est, "coef_", None)
        imp = (np.abs(coef).ravel() if coef is not None
               else getattr(est, "feature_importances_", np.ones(len(feats))))
    else:
        imp = np.abs(sv).mean(0)
    imp = imp / (imp.sum() + 1e-12)
    order = np.argsort(imp)[::-1]
    return dict(method="shap" if sv is not None else "coef/importance",
                model=model_name, horizon_k=horizon_k,
                top_features=[dict(feature=feats[i], importance=float(imp[i]))
                              for i in order[:15]])


# -----------------------------------------------------------------------------
# per-anchor explainer factory (used by forward_sim.py)
# -----------------------------------------------------------------------------
def make_world_model_explainer(model, seq: Dict, device: str = "cpu") -> Callable:
    feats = list(seq["feature_names"])

    def explain_anchor(x: np.ndarray, dt: np.ndarray, _feature_vec) -> Dict:
        gi = gradient_x_input(model, x, dt, feats, device)
        att = attention_saliency(model, x, dt, device)
        return dict(top_features=gi["top_features"],
                    temporal_saliency_gradient=gi["per_window"],
                    temporal_saliency_attention=att)
    return explain_anchor


# -----------------------------------------------------------------------------
def run_all(device: str = "cpu") -> Dict:
    import torch as _t
    from sentinel_wm.sequences import load_sequences
    ckpt = _t.load(C.WORLD_MODEL_PT, map_location=device, weights_only=False)
    seq = load_sequences()
    cfg = C.CONFIG
    cfg.sequence.horizon = ckpt["sequence"]["K"]
    model = build_model(ckpt["config"]["n_features"], cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    report = dict(
        has_shap=_HAS_SHAP,
        world_model_shap=shap_world_model(model, seq, device),
        baseline_lr_shap=shap_baseline("logistic_regression", 1),
        baseline_rf_shap=shap_baseline("random_forest", 1),
    )
    # one worked example
    te = np.where(seq["split"] == "test")[0]
    i = int(te[np.argmax(seq["y_atk"][te].max(1))])
    report["example_anchor"] = dict(
        window_index=int(seq["window_index"][i]),
        **gradient_x_input(model, seq["X"][i], seq["dt"][i],
                           list(seq["feature_names"]), device),
        attention=attention_saliency(model, seq["X"][i], seq["dt"][i], device))

    out = os.path.join(C.REPORT_DIR, "explainability.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[explain] shap_installed={_HAS_SHAP}")
    print("[explain] world-model top features:")
    for r in report["world_model_shap"]["top_features"][:8]:
        print(f"    {r['feature']:26s} {r['mean_abs_shap']:.4f}")
    print(f"[explain] -> {out}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    run_all(p.parse_args().device)
