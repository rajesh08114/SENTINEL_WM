"""SENTINEL-WM inference backend - FastAPI app factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.logging import RequestIdMiddleware, configure_logging
from app.settings import settings          # noqa: F401  (sets SENTINEL_WM_MODEL_DIR early)

configure_logging(settings.log_json)
_START = __import__("time").time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from app.jobs import store, worker
    from app.inference.loader import (BundleContractError, BundleNotFound,
                                      get_engine)
    from app.agent.registry import REGISTRY
    from app.live.session import MANAGER

    store.init_db()
    try:
        get_engine()                        # load + warm the engine from the bundle
        print("[startup] model bundle loaded")
    except (BundleNotFound, BundleContractError) as e:
        print(f"[startup] WARNING - {e}")   # /health reports 'degraded' until a bundle appears

    MANAGER.sessions.clear()                 # fresh registries per process
    REGISTRY._agents.clear()
    reaper = asyncio.create_task(MANAGER.reap_idle())
    try:
        yield
    finally:
        reaper.cancel()
        await MANAGER.shutdown()
        worker.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="SENTINEL-WM inference API",
        version=__version__,
        summary="Flow CSV / PCAP / telemetry -> attack-progression forecast "
                "with explanations and MITRE ATT&CK phase mapping.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    from app.api import (routes_agent, routes_forecast, routes_jobs, routes_live,
                         routes_meta, routes_rules, ws_stream)
    app.include_router(routes_meta.router)
    app.include_router(routes_forecast.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_live.router)
    app.include_router(routes_agent.router)
    app.include_router(routes_rules.router)
    app.include_router(ws_stream.router)
    if settings.metrics:
        from app.api import routes_metrics
        app.include_router(routes_metrics.router)
    return app



app = create_app()
