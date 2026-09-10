"""Sources that feed a LiveSession.

SyntheticSource drives the scenario generator on a wall-clock-paced asyncio
task.  AgentSource bridges a connected capture agent's flow batches into the
session (agent wiring lands in Phase 2; start/stop no-op when agent is None).
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional

from app.synth import scenarios as _s


class SyntheticSource:
    TICK = 0.25

    def __init__(self, session: Any, cfg: "_s.ScenarioConfig",
                 base_epoch: Optional[float] = None):
        self.session = session
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._stop = False
        # emitted flow_start_epoch values look like real unix time so the
        # windower / a mixed capture+synthetic session behave identically
        self.base_epoch = time.time() if base_epoch is None else base_epoch

    def request_stop(self) -> None:
        self._stop = True

    def _shift(self, rows: list[dict]) -> list[dict]:
        for r in rows:
            r["flow_start_epoch"] = round(self.base_epoch + r["flow_start_epoch"], 6)
        return rows

    async def run(self) -> None:
        start = time.monotonic()
        speed = self.cfg.speed
        emitted_until = 0.0                       # scenario seconds (relative)
        dur = self.cfg.duration_s
        try:
            while not self._stop:
                now_rel = (time.monotonic() - start) * speed
                if now_rel >= dur:
                    break
                if now_rel > emitted_until:
                    rows = _s.emit(emitted_until, now_rel, self.cfg, self._rng)
                    emitted_until = now_rel
                    if rows:
                        await self.session.feed(self._shift(rows))
                await asyncio.sleep(self.TICK)
            if emitted_until < dur and not self._stop:
                rows = _s.emit(emitted_until, dur, self.cfg, self._rng)
                if rows:
                    await self.session.feed(self._shift(rows))
        except asyncio.CancelledError:
            raise
        finally:
            await self.session.stop()


class AgentSource:
    def __init__(self, session: Any, agent: Optional[Any] = None,
                 iface: Optional[str] = None, bpf: Optional[str] = None):
        self.session = session
        self.agent = agent
        self.iface = iface
        self.bpf = bpf

    async def on_flows(self, rows: list[dict]) -> None:
        await self.session.feed(rows)

    async def start(self) -> None:
        if self.agent is not None:
            await self.agent.send_cmd(
                {"cmd": "start", "iface": self.iface, "bpf": self.bpf})

    async def stop(self) -> None:
        if self.agent is not None:
            await self.agent.send_cmd({"cmd": "stop"})
