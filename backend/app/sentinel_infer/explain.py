"""Per-anchor explanations. Vendored from research/sentinel_wm/explain.py
(the SHAP / global paths are NOT vendored - they need the sequences.npz corpus).
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch


def attention_saliency(model, x: np.ndarray, dt: np.ndarray,
                       device: str = "cpu") -> List[float]:
    xb = torch.as_tensor(x[None], dtype=torch.float32, device=device)
    dtb = torch.as_tensor(dt[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        enc = model.encoder(xb, dtb)
    attn = enc["attn"]
    if attn is None:
        return [0.0] * x.shape[0]
    per_key = attn[0].sum(0)
    per_key = (per_key / per_key.sum()).cpu().numpy()
    return [float(v) for v in per_key]


def gradient_x_input(model, x: np.ndarray, dt: np.ndarray,
                     feature_names: List[str], device: str = "cpu",
                     horizon_reduce: str = "max") -> Dict:
    xb = torch.tensor(np.asarray(x[None]), dtype=torch.float32, device=device,
                      requires_grad=True)
    dtb = torch.as_tensor(dt[None], dtype=torch.float32, device=device)
    # cuDNN's fused RNN backward is only legal in training mode; run this one
    # grad pass through the native RNN kernel instead (model stays in eval()).
    with torch.backends.cudnn.flags(enabled=False):
        out = model(xb, dtb)
        logits = out["attack_logits_k"][0]
        score = logits.max() if horizon_reduce == "max" else logits.mean()
        model.zero_grad()
        score.backward()
    g = xb.grad[0].detach().cpu().numpy()
    sal = np.abs(g) * np.abs(x)
    feat_imp = sal.sum(0)
    feat_imp = feat_imp / (feat_imp.sum() + 1e-12)
    order = np.argsort(feat_imp)[::-1]
    return dict(
        per_window=[float(v) for v in sal.sum(1) / (sal.sum() + 1e-12)],
        top_features=[dict(feature=feature_names[i],
                           importance=float(feat_imp[i]),
                           signed=float(np.sign(g[-1, i])))
                      for i in order[:10]])


def make_explainer(model, seq: Dict, device: str = "cpu") -> Callable:
    feats = list(seq["feature_names"])

    def explain_anchor(x: np.ndarray, dt: np.ndarray, _feature_vec) -> Dict:
        gi = gradient_x_input(model, x, dt, feats, device)
        att = attention_saliency(model, x, dt, device)
        return dict(top_features=gi["top_features"],
                    temporal_saliency_gradient=gi["per_window"],
                    temporal_saliency_attention=att)
    return explain_anchor
