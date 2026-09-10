"""Load the SENTINEL-WM world model + scaler + explainer once, at startup."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.settings import settings          # noqa: F401  (sets SENTINEL_WM_MODEL_DIR)


class BundleNotFound(RuntimeError):
    pass


@dataclass
class Engine:
    model: Any
    ckpt: dict
    scaler: Any
    feat_cols: list[str]
    explainer: Callable | None
    L: int
    K: int
    n_features: int
    window_seconds: int
    states: list[str]
    alert_threshold: float
    device: str
    manifest: dict = field(default_factory=dict)
    # SENTINEL-WM (system) blend: the member predictors + val-tuned weight.
    # Empty -> the backend serves the raw world model only.
    members: list = field(default_factory=list)          # [(name, _TorchPredictor)]
    blend_weight: float = 1.0
    serve_mode: str = "world_model"                       # "world_model" | "system"


@lru_cache
def get_engine() -> Engine:
    """Build the singleton inference engine. Raises BundleNotFound with a clear
    message if `SENTINEL_WM_MODEL_DIR` has no `world_model.pt`."""
    bundle = Path(settings.bundle_dir)
    wm = bundle / "world_model.pt"
    if not wm.exists():
        raise BundleNotFound(
            f"no model bundle at {bundle} (missing world_model.pt). "
            f"Build it with:  cd research && python -m sentinel_wm.research all "
            f"&& sentinel-wm bundle {bundle}"
        )

    from sentinel_wm import config as C
    from sentinel_wm import forward_sim
    from sentinel_wm.state_windows import STATE_FEATURE_COLS

    device = settings.device
    model, ckpt = forward_sim.load_checkpoint(device)

    scaler_path = C.bundled("state_scaler.pkl") or C.SCALER_PKL
    with open(scaler_path, "rb") as fh:
        sc = pickle.load(fh)
    scaler = sc["scaler"]
    feat_cols = list(sc.get("feature_names") or STATE_FEATURE_COLS)

    explainer = None
    try:
        from sentinel_wm import explain
        explainer = explain.make_world_model_explainer(
            model, {"feature_names": np.array(feat_cols, dtype=object)}, device)
    except Exception as e:                                   # pragma: no cover
        print(f"[loader] explainer unavailable: {e}")

    manifest = {}
    mf = bundle / "bundle.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())

    L = int(ckpt["sequence"]["L"])
    K = int(ckpt["sequence"]["K"])
    F = int(ckpt["config"]["n_features"])

    # ---- SENTINEL-WM (system): load the blend members if present ----------
    members: list = []
    blend_w = float(manifest.get("system_blend_weight", 0.5))
    want_system = settings.serve_mode in ("system", "auto")
    if want_system:
        try:
            from sentinel_wm import registry
            for nm in manifest.get("system_members", ["tcn", "lstm", "gru"]):
                try:
                    members.append((nm, registry.load_predictor(nm, device)))
                except Exception as e:                       # pragma: no cover
                    print(f"[loader] system member {nm} unavailable: {e}")
        except Exception as e:                               # pragma: no cover
            print(f"[loader] registry unavailable: {e}")
    serve_mode = "system" if members else "world_model"
    if settings.serve_mode == "system" and not members:
        print("[loader] serve_mode=system requested but no member models in the "
              "bundle - falling back to world_model")

    eng = Engine(
        model=model, ckpt=ckpt, scaler=scaler, feat_cols=feat_cols,
        explainer=explainer, L=L, K=K, n_features=F,
        window_seconds=int(C.CONFIG.window.window_seconds),
        states=list(C.PROGRESSION_STATES),
        alert_threshold=float(manifest.get("system_threshold")
                              or ckpt.get("alert_threshold", 0.7)),
        device=device, manifest=manifest,
        members=members, blend_weight=blend_w if members else 1.0,
        serve_mode=serve_mode,
    )

    # warm pass so the first real request is not cold
    try:
        z = np.zeros((L, len(feat_cols)), np.float32)
        forward_sim.simulate_anchor(model, z, np.zeros(L, np.float32), ckpt,
                                    meta={}, M_samples=4, device=device)
    except Exception as e:                                   # pragma: no cover
        print(f"[loader] warm pass failed (non-fatal): {e}")

    print(f"[loader] engine ready: mode={eng.serve_mode} "
          f"members={[n for n, _ in members]} blend_w={eng.blend_weight:.2f} "
          f"L={L} K={K} F={F} device={device} bundle={bundle}")
    return eng
