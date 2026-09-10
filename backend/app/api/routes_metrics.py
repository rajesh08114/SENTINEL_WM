"""Optional Prometheus metrics (SENTINEL_METRICS=true, needs prometheus-client).

Kept dependency-light: if prometheus_client isn't installed the router still
mounts but /metrics returns a plain-JSON snapshot instead of the exposition
format.
"""
from __future__ import annotations

from fastapi import APIRouter, Response

router = APIRouter(tags=["metrics"])

try:
    from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                                   generate_latest)

    _HAVE_PROM = True
    FORECAST_LATENCY = Histogram(
        "sentinel_forecast_seconds", "simulate_anchor latency per window")
    LIVE_SESSIONS = Gauge("sentinel_live_sessions", "active live sessions")
    FLOWS_IN = Counter("sentinel_flows_in_total", "flow rows fed to live sessions")
except Exception:                                    # pragma: no cover
    _HAVE_PROM = False
    FORECAST_LATENCY = LIVE_SESSIONS = FLOWS_IN = None  # type: ignore


def observe_forecast(seconds: float) -> None:
    if _HAVE_PROM and FORECAST_LATENCY is not None:
        FORECAST_LATENCY.observe(seconds)


def add_flows(n: int) -> None:
    if _HAVE_PROM and FLOWS_IN is not None:
        FLOWS_IN.inc(n)


@router.get("/metrics")
def metrics() -> Response:
    from app.live.session import MANAGER
    active = sum(1 for s in MANAGER.sessions.values() if s.state != "stopped")
    if _HAVE_PROM:
        LIVE_SESSIONS.set(active)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    total_flows = sum(s.stats.get("flows_in", 0) for s in MANAGER.sessions.values())
    return Response(
        media_type="application/json",
        content=f'{{"live_sessions": {active}, "flows_in_total": {total_flows}}}',
    )
