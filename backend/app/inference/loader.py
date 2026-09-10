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
    eng = Engine(
        model=model, ckpt=ckpt, scaler=scaler, feat_cols=feat_cols,
        explainer=explainer, L=L, K=K, n_features=F,
        window_seconds=int(C.CONFIG.window.window_seconds),
        states=list(C.PROGRESSION_STATES),
        alert_threshold=float(ckpt.get("alert_threshold", 0.7)),
        device=device, manifest=manifest,
    )

    # warm pass so the first real request is not cold
    try:
        z = np.zeros((L, len(feat_cols)), np.float32)
        forward_sim.simulate_anchor(model, z, np.zeros(L, np.float32), ckpt,
                                    meta={}, M_samples=4, device=device)
    except Exception as e:                                   # pragma: no cover
        print(f"[loader] warm pass failed (non-fatal): {e}")

    print(f"[loader] engine ready: L={L} K={K} F={F} device={device} "
          f"bundle={bundle}")
    return eng
