"""LiveSession — one real-time forecasting stream — and LiveManager, the registry.

A session owns a StreamingWindower (the same incremental 10s windowing +
simulate_anchor path a CSV upload uses), a bounded ring buffer of recent
forecasts for late subscribers, and a set of WebSocket subscribers it fans
forecasts out to.  Sources (synthetic generator, capture agent) call `feed()`.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any, Callable, Iterable, Optional

from app.live.errors import LiveCapacityError, LiveNotFound
from app.settings import settings

RING_MAX = 200


def _observe(flows: int, seconds: float, n_forecasts: int) -> None:
    """Best-effort metrics hook; no-op unless SENTINEL_METRICS + prometheus-client."""
    try:
        from app.api import routes_metrics as _m
        _m.add_flows(flows)
        for _ in range(n_forecasts):
            _m.observe_forecast(seconds / max(1, n_forecasts))
    except Exception:
        pass


class LiveSession:
    def __init__(self, sid: str, source_kind: str, params: dict,
                 *, windower: Any = None,
                 windower_factory: Optional[Callable[[str, dict], Any]] = None):
        self.id = sid
        self.source_kind = source_kind
        self.params = dict(params or {})
        self.state = "starting"          # starting|running|stopping|stopped|error
        self.error: Optional[str] = None
        self.stats = {"flows_in": 0, "windows": 0, "forecasts": 0, "alerts": 0}
        self.ring: deque = deque(maxlen=RING_MAX)
        self.subscribers: set = set()
        self.created_at = time.time()
        self.last_activity = self.created_at
        self._tasks: set[asyncio.Task] = set()
        self._source: Any = None            # set by the route that starts the source

        if windower is not None:
            self._win = windower
        elif windower_factory is not None:
            self._win = windower_factory(sid, self.params)
        else:                            # default: the real streaming windower
            from app.streaming.windower import StreamingWindower
            self._win = StreamingWindower(
                sid, family_hint=self.params.get("family_hint"),
                explain=bool(self.params.get("explain", True)))

    # -- ingest --------------------------------------------------------
    async def feed(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self.state in ("starting",):
            self.state = "running"
        try:
            n = self._win.add_flows(list(rows))
        except Exception as e:                 # BadUpload etc. - surface, keep going
            self.error = f"{type(e).__name__}: {e}"
            await self._broadcast({"type": "error", "detail": self.error})
            return
        self.stats["flows_in"] += int(n)
        self.last_activity = time.time()
        t0 = time.perf_counter()
        forecasts = await asyncio.to_thread(self._win.poll_ready)
        _observe(int(n), time.perf_counter() - t0, len(forecasts or []))
        await self._emit(forecasts)

    async def _emit(self, forecasts: Iterable[dict]) -> None:
        for fc in forecasts or []:
            if "matched_rules" not in fc:
                try:
                    from app.rules.engine import RULE_ENGINE
                    matches = RULE_ENGINE.evaluate_anchor(fc, fc.get("driving_features") or {})
                    fc["matched_rules"] = [m.model_dump() for m in matches]
                except Exception:
                    fc["matched_rules"] = []
            self.ring.append(fc)
            self.stats["forecasts"] += 1
            self.stats["windows"] += 1
            if fc.get("alert"):
                self.stats["alerts"] += 1
            await self._broadcast({"type": "forecast", **fc})


    # -- subscribers -------------------------------------------------
    async def subscribe(self, ws: Any) -> None:
        self.subscribers.add(ws)
        await ws.send_json({"type": "status", **self.info()})
        for fc in list(self.ring):
            await ws.send_json({"type": "forecast", **fc})

    def unsubscribe(self, ws: Any) -> None:
        self.subscribers.discard(ws)

    async def _broadcast(self, msg: dict) -> None:
        for ws in list(self.subscribers):
            try:
                await ws.send_json(msg)
            except Exception:
                self.subscribers.discard(ws)

    async def broadcast_status(self) -> None:
        await self._broadcast({"type": "status", **self.info()})

    # -- lifecycle -------------------------------------------------
    def attach_task(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        if self.state in ("stopping", "stopped"):
            return
        self.state = "stopping"
        src = self._source
        if src is not None and hasattr(src, "stop"):
            try:
                await src.stop()
            except Exception:
                pass
        for t in list(self._tasks):
            t.cancel()
        try:
            tail = await asyncio.to_thread(self._win.flush)
            await self._emit(tail)
        except Exception:
            pass
        self.state = "stopped"
        self.last_activity = time.time()
        await self._broadcast({"type": "status", **self.info()})
        await self._broadcast({"type": "bye"})
        for ws in list(self.subscribers):
            try:
                await ws.close()
            except Exception:
                pass
        self.subscribers.clear()

    def info(self) -> dict:
        return {
            "id": self.id, "source_kind": self.source_kind, "state": self.state,
            "error": self.error, "params": self.params, "stats": dict(self.stats),
            "created_at": self.created_at, "last_activity": self.last_activity,
            "subscribers": len(self.subscribers),
        }


class LiveManager:
    def __init__(self, max_sessions: Optional[int] = None):
        self.sessions: dict[str, LiveSession] = {}
        self.max_sessions = int(max_sessions or settings.live_max_sessions)

    def _active(self) -> list[LiveSession]:
        return [s for s in self.sessions.values() if s.state != "stopped"]

    def create(self, source_kind: str, params: dict, **kw) -> LiveSession:
        if len(self._active()) >= self.max_sessions:
            raise LiveCapacityError(
                f"max {self.max_sessions} concurrent live sessions reached")
        sid = uuid.uuid4().hex[:12]
        s = LiveSession(sid, source_kind, params, **kw)
        self.sessions[sid] = s
        return s

    def get(self, sid: str) -> LiveSession:
        try:
            return self.sessions[sid]
        except KeyError:
            raise LiveNotFound(sid)

    def list(self) -> list[dict]:
        return [s.info() for s in self.sessions.values()]

    async def remove(self, sid: str) -> None:
        s = self.get(sid)
        await s.stop()
        self.sessions.pop(sid, None)

    async def reap_idle(self, poll_s: float = 30.0) -> None:
        while True:
            try:
                await asyncio.sleep(poll_s)
            except asyncio.CancelledError:
                return
            now = time.time()
            for sid, s in list(self.sessions.items()):
                if s.state == "stopped":
                    self.sessions.pop(sid, None)
                elif (not s.subscribers
                      and now - s.last_activity > settings.live_idle_timeout_s):
                    try:
                        await self.remove(sid)
                    except Exception:
                        pass

    async def shutdown(self) -> None:
        for sid in list(self.sessions):
            try:
                await self.remove(sid)
            except Exception:
                pass


MANAGER = LiveManager()
