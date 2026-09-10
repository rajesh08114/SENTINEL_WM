from __future__ import annotations

import time

from fastapi import APIRouter

from app.inference.loader import BundleContractError, BundleNotFound, get_engine
from app.sentinel_infer.schema import PROGRESSION_STATES
from app.schemas import HealthResponse, MetaResponse
from app.settings import settings

router = APIRouter(tags=["meta"])


def _live_agent_detail() -> dict:
    from app.agent.registry import REGISTRY
    from app.live.session import MANAGER
    active = [s for s in MANAGER.sessions.values() if s.state != "stopped"]
    return {
        "live": {"sessions": len(active), "max": MANAGER.max_sessions},
        "agent": {"connected": bool(REGISTRY.all()), "count": len(REGISTRY.all())},
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from app.main import _START
    extra = {**_live_agent_detail(), "uptime_s": round(time.time() - _START, 1)}
    try:
        eng = get_engine()
        return HealthResponse(
            status="ok", model_loaded=True,
            bundle={"loaded": True, "serve_mode": eng.serve_mode, "L": eng.L,
                    "K": eng.K, "n_features": eng.n_features},
            **extra)
    except (BundleNotFound, BundleContractError) as e:
        return HealthResponse(status="degraded", model_loaded=False, detail=str(e),
                              bundle={"loaded": False}, **extra)
    except Exception as e:                                   # pragma: no cover
        return HealthResponse(status="error", model_loaded=False, detail=str(e),
                              **extra)


@router.get("/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    eng = get_engine()
    return MetaResponse(
        history_windows=eng.L, horizon_steps=eng.K, feature_dim=eng.n_features,
        window_seconds=eng.window_seconds,
        progression_states=PROGRESSION_STATES, feature_names=eng.feat_cols,
        alert_threshold=eng.alert_threshold, device=eng.device,
        serve_mode=eng.serve_mode,
        blend_members=[n for n, _ in eng.members],
        blend_weight=round(eng.blend_weight, 3) if eng.members else None,
        bundle=eng.manifest,
    )


@router.get("/models")
def models() -> dict:
    """What the bundle contains (from bundle.json). The MVP serves
    `serve_mode`; the members / metrics are informational."""
    eng = get_engine()
    return {
        "active": "SENTINEL-WM (system)" if eng.members else "SENTINEL-WM",
        "serve_mode": eng.serve_mode,
        "blend_members": [n for n, _ in eng.members],
        "blend_weight": round(eng.blend_weight, 3) if eng.members else None,
        "metrics": eng.manifest.get("system_metrics"),
        "bundle": {k: eng.manifest.get(k) for k in
                   ("created", "git_sha", "encoder", "L", "K", "n_features",
                    "snapshots", "registry")},
    }
