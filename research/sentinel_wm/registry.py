#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  registry.py   -  one loader for every saved model
# -----------------------------------------------------------------------------
# The benchmark trains ~20 model variants and drops them under research/models/.
# This module gives the future FastAPI service (and the notebooks) a single
# uniform way to enumerate and call them:
#
#   from sentinel_wm.registry import build_registry, load_predictor
#   build_registry()                       # -> research/models/registry.json
#   p = load_predictor("SENTINEL-WM")
#   out = p.predict(X, dt)                  # X [N,L,F] scaled, dt [N,L] log1p
#   out["attack_prob_k"]     # [N, K]
#   out["progression_k"]     # [N, K]  (argmax state index) or None
#
# Every predictor consumes the SAME scaled sequence tensor produced by
# sentinel_wm.sequences (+ artifacts/state_scaler.pkl). The GAT predictor also
# needs per-window graphs; pass `graph=(day_array, window_index_array)` and it
# looks them up in artifacts/graph_windows.npz.
# =============================================================================
from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from sentinel_wm import config as C

def _models_dir() -> str:
    # a deploy bundle (MODEL_DIR/models/) wins over the research tree (runs/models/)
    b = os.path.join(C.model_dir(), "models")
    if os.path.isdir(b):
        return b
    return os.path.join(C.research_dir(), "models")


def _registry_json() -> str:
    return os.path.join(_models_dir(), "registry.json")


def _classical_dir() -> str:
    return os.path.join(_models_dir(), "classical")


def _nn_dir() -> str:
    return os.path.join(_models_dir(), "nn")


# back-compat module constants (prefer the functions for a live value)
REGISTRY_JSON = _registry_json()
CLASSICAL_DIR = _classical_dir()
NN_DIR = _nn_dir()


# -----------------------------------------------------------------------------
# predictors
# -----------------------------------------------------------------------------
class _ClassicalPredictor:
    def __init__(self, name, pkl_path, meta):
        with open(pkl_path, "rb") as fh:
            self.ests = pickle.load(fh)
        self.name = name
        self.input_kind = meta.get("input_kind", "window")
        self.threshold = meta.get("threshold", 0.5)

    def predict(self, X, dt=None, graph=None) -> Dict:
        from sentinel_wm.baselines import _proba
        Xin = X[:, -1, :] if self.input_kind == "window" else X.reshape(len(X), -1)
        probs = np.stack([_proba(e, Xin) for e in self.ests], 1)
        return dict(attack_prob_k=probs, progression_k=None,
                    threshold=self.threshold)


class _TorchPredictor:
    def __init__(self, name, pt_path, device="cpu"):
        import torch
        self.name, self.device = name, device
        ck = torch.load(pt_path, map_location=device, weights_only=False)
        self.threshold = ck.get("alert_threshold", 0.5)
        self.kind = ck.get("kind")
        if self.kind == "gat":
            from sentinel_wm.gat import GATForecaster, _GraphCtx
            ex = ck.get("extra", {})
            self._ctx = _GraphCtx(ck["sequence"]["L"])
            self.model = GATForecaster(ex.get("n_node_feat", self._ctx.Fn),
                                       ck["sequence"]["L"], ck["sequence"]["K"],
                                       len(C.PROGRESSION_STATES),
                                       d_model=ex.get("d_model", 96))
        elif self.kind in ("mlp", "lstm", "gru", "tcn"):
            from sentinel_wm.nn_zoo import build_nn_model
            self.model = build_nn_model(self.kind, ck["n_features"],
                                        ck["sequence"]["L"])
        else:                                   # world model
            from sentinel_wm.models import build_model
            cfg = C.CONFIG
            cfg.sequence.horizon = ck["sequence"]["K"]
            self.model = build_model(ck["config"]["n_features"], cfg)
            self._snaps = ck.get("snapshots", [])
            self._self_ens = bool(ck.get("self_ensemble", False))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(device)

    def predict(self, X, dt, graph=None) -> Dict:
        import torch
        with torch.no_grad():
            if self.kind == "gat":
                from torch.utils.data import DataLoader
                from sentinel_wm.gat import _GraphDataset
                assert graph is not None, "GAT predictor needs graph=(day, window_index)"
                day, wi = graph
                ds = _GraphDataset(self._ctx, day, wi,
                                   np.zeros((len(wi), 1)), np.zeros((len(wi), 1)))
                pa, pp = [], []
                for nf, aj, mk, _a, _p in DataLoader(ds, batch_size=256):
                    o = self.model(nf.to(self.device), aj.to(self.device),
                                   mk.to(self.device))
                    pa.append(torch.sigmoid(o["attack_logits_k"]).cpu().numpy())
                    pp.append(o["prog_logits_k"].argmax(-1).cpu().numpy())
                return dict(attack_prob_k=np.concatenate(pa),
                            progression_k=np.concatenate(pp),
                            threshold=self.threshold)
            if self.kind not in ("mlp", "lstm", "gru", "tcn"):   # world model
                from sentinel_wm.models import wm_predict
                probs, prog = wm_predict(self.model, X, dt, self.device,
                                         snapshots=getattr(self, "_snaps", None),
                                         self_ensemble=getattr(self, "_self_ens", False))
                return dict(attack_prob_k=probs, progression_k=prog,
                            threshold=self.threshold)
            x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            d = torch.as_tensor(dt, dtype=torch.float32, device=self.device)
            pa, pp = [], []
            for i in range(0, len(x), 512):
                o = self.model(x[i:i + 512], d[i:i + 512])
                pa.append(torch.sigmoid(o["attack_logits_k"]).cpu().numpy())
                pp.append(o["prog_logits_k"].argmax(-1).cpu().numpy())
        return dict(attack_prob_k=np.concatenate(pa),
                    progression_k=np.concatenate(pp), threshold=self.threshold)


# -----------------------------------------------------------------------------
def build_registry(verbose: bool = True, base_dir: str = None) -> Dict:
    """Scan a `<base>/models/{classical,nn}` tree and write its `registry.json`.
    `base_dir` defaults to the RESEARCH output dir (`runs/`); `bundle.py` passes
    the bundle dir. The read helpers (`load_predictor`) prefer the bundle, but
    the writer must not, or a stale bundle would capture a fresh research run."""
    reg: Dict[str, Dict] = {}
    base_dir = base_dir or C.research_dir()
    classical_dir = os.path.join(base_dir, "models", "classical")
    nn_dir = os.path.join(base_dir, "models", "nn")
    registry_json = os.path.join(base_dir, "models", "registry.json")
    # paths in registry.json are relative to base_dir; load_predictor resolves
    # against C.model_dir(), C.ROOT, and the registry's own dir.
    _base = base_dir
    wm_pt = os.path.join(base_dir, "models", "world_model.pt")
    if not os.path.exists(wm_pt):
        wm_pt = C.bundled("world_model.pt") or C.WORLD_MODEL_PT

    for meta_path in sorted(glob.glob(os.path.join(classical_dir, "*.meta.json"))):
        meta = json.load(open(meta_path))
        name = meta["name"]
        reg[name] = dict(family="classical", framework="sklearn",
                         kind=meta.get("input_kind"),
                         path=os.path.relpath(meta_path[:-10] + ".pkl", _base),
                         input=dict(tensor="window" if meta.get("input_kind") == "window"
                                    else "sequence_flat", K=meta.get("horizon")),
                         threshold=meta.get("threshold"),
                         metrics=meta.get("metrics", {}))

    for meta_path in sorted(glob.glob(os.path.join(nn_dir, "*.meta.json"))):
        meta = json.load(open(meta_path))
        name = meta["name"]
        reg[name] = dict(family=meta.get("family", "nn"), framework="torch",
                         kind=meta.get("kind"),
                         path=os.path.relpath(meta_path[:-10] + ".pt", _base),
                         input=dict(tensor="graph_sequence" if meta.get("kind") == "gat"
                                    else "sequence"),
                         threshold=meta.get("threshold"),
                         params=meta.get("params"),
                         metrics=meta.get("metrics", {}))

    if os.path.exists(wm_pt):
        import torch
        ck = torch.load(wm_pt, map_location="cpu", weights_only=False)
        wm = os.path.join(C.REPORT_DIR, "world_model_metrics.json")
        met = json.load(open(wm))["test"]["any_horizon"] if os.path.exists(wm) else {}
        ek = ck.get("config", {}).get("model", {}).get("encoder", "gru")
        reg["SENTINEL-WM"] = dict(
            family="world_model", framework="torch",
            kind=f"{ek}+STN (self-ensemble, {len(ck.get('snapshots', []))} snap)",
            path=os.path.relpath(wm_pt, _base),
            input=dict(tensor="sequence", L=ck["sequence"]["L"], K=ck["sequence"]["K"]),
            threshold=ck.get("alert_threshold"),
            snapshots=ck.get("snapshots", []),
            self_ensemble=bool(ck.get("self_ensemble", False)),
            metrics={"f1": met.get("f1"), "auroc": met.get("auroc")})
        # the deployed system = WM self-ensemble + strong members, val-tuned blend
        reg["SENTINEL-WM-system"] = dict(
            family="system", framework="composite",
            kind="WM self-ensemble + [xgboost__seq, random_forest__seq, "
                 "hist_gradient_boosting__seq, gat] (val-tuned blend)",
            components=["SENTINEL-WM"] + list(getattr(C.CONFIG.train, "system_members", [])),
            note="composed at inference by benchmark._score_world_model / "
                 "registry.load_system_predictor")

    os.makedirs(os.path.dirname(registry_json), exist_ok=True)
    json.dump(reg, open(registry_json, "w"), indent=2, default=float)
    if verbose:
        print(f"[registry] {len(reg)} models -> {registry_json}")
        for n, r in reg.items():
            print(f"   {n:28s} {r['family']:12s} {r['framework']:8s} "
                  f"f1={r.get('metrics', {}).get('f1', '?')}")
    return reg


class _SystemPredictor:
    """The deployed SENTINEL-WM system: WM self-ensemble blended with the strong
    members (weight fixed at build time from the last benchmark)."""

    def __init__(self, members, wm_weight: float, device: str = "cpu"):
        self.wm = load_predictor("SENTINEL-WM", device)
        self.members = [load_predictor(m, device) for m in members]
        self.w = wm_weight

    def predict(self, X, dt, graph=None) -> Dict:
        wp = self.wm.predict(X, dt, graph)["attack_prob_k"]
        import numpy as np
        mps = []
        for m in self.members:
            try:
                mps.append(m.predict(X, dt, graph)["attack_prob_k"])
            except Exception:
                pass
        blend = wp if not mps else self.w * wp + (1 - self.w) * np.mean(mps, 0)
        return dict(attack_prob_k=blend, progression_k=None)


def load_system_predictor(device: str = "cpu", wm_weight: float = 0.55):
    return _SystemPredictor(list(getattr(C.CONFIG.train, "system_members", [])),
                            wm_weight, device)


def list_models() -> List[str]:
    rj = _registry_json()
    if not os.path.exists(rj):
        build_registry(verbose=False)
    return list(json.load(open(rj)).keys())


def load_predictor(name: str, device: str = "cpu"):
    rj = _registry_json()
    if not os.path.exists(rj):
        build_registry(verbose=False)
    reg = json.load(open(rj))
    if name not in reg:
        raise KeyError(f"{name!r} not in registry. have: {list(reg)}")
    r = reg[name]
    # `path` is relative to the registry's base dir (bundle, runs/, or ROOT).
    rj_base = os.path.dirname(os.path.dirname(rj))         # <base>/models/registry.json -> <base>
    path = next((c for c in (os.path.join(C.model_dir(), r["path"]),
                             os.path.join(rj_base, r["path"]),
                             os.path.join(C.research_dir(), r["path"]),
                             os.path.join(C.ROOT, r["path"]))
                 if os.path.exists(c)), os.path.join(C.ROOT, r["path"]))
    if r["framework"] == "sklearn":
        meta = json.load(open(path[:-4] + ".meta.json"))
        return _ClassicalPredictor(name, path, meta)
    return _TorchPredictor(name, path, device)


if __name__ == "__main__":
    build_registry()
