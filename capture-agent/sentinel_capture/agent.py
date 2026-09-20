"""The agent: connect to the backend's WS /agent, advertise interfaces, and on
command sniff a NIC and stream assembled flow rows.

Server -> agent commands (JSON):  {"cmd":"start","iface":..,"bpf":..}
                                  {"cmd":"stop"}
                                  {"cmd":"interfaces"}
Agent  -> server messages:        {"type":"hello","name":..,"interfaces":[..]}
                                  {"type":"interfaces","interfaces":[..]}
                                  {"type":"started","iface":..}
                                  {"type":"flows","records":[..]}
                                  {"type":"stopped"}
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Optional

import websockets

from .flowmeter import FlowMeter
from .interfaces import list_interfaces, to_dicts

HARVEST_INTERVAL = 1.0


def _default_sniffer(iface: str, bpf: Optional[str], on_packet: Callable[[Any], None]):
    from scapy.sendrecv import AsyncSniffer
    return AsyncSniffer(iface=iface, filter=bpf or None, prn=on_packet, store=False)


class _Capture:
    def __init__(self, sniffer_factory):
        self._factory = sniffer_factory or _default_sniffer
        self.meter = FlowMeter()
        self._sniffer = None
        self._task: Optional[asyncio.Task] = None

    def start(self, iface: str, bpf: Optional[str], send_flows):
        self.stop()
        self.meter = FlowMeter()
        clean_bpf = bpf.strip() if bpf and bpf.strip() else None
        self._sniffer = self._factory(iface, clean_bpf, self.meter.add_packet)
        self._sniffer.start()
        self._task = asyncio.create_task(self._pump(send_flows))

    async def _pump(self, send_flows):
        try:
            while True:
                await asyncio.sleep(HARVEST_INTERVAL)
                rows = self.meter.harvest(time.time())
                if rows:
                    await send_flows(rows)
        except asyncio.CancelledError:
            raise

    def stop(self) -> list[dict]:
        if self._task:
            self._task.cancel()
            self._task = None
        if self._sniffer is not None:
            try:
                self._sniffer.stop()
            except Exception:
                pass
            self._sniffer = None
        return self.meter.flush_all()


async def _session(ws, name: str, sniffer_factory) -> None:
    async def send(obj: dict) -> None:
        await ws.send(json.dumps(obj))

    cap = _Capture(sniffer_factory)
    await send({"type": "hello", "name": name,
                "interfaces": to_dicts(list_interfaces())})
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            cmd = msg.get("cmd")
            if cmd == "start":
                try:
                    cap.start(msg.get("iface"), msg.get("bpf"),
                              lambda rows: send({"type": "flows", "records": rows}))
                    await send({"type": "started", "iface": msg.get("iface")})
                except Exception as e:
                    await send({"type": "error", "detail": f"Failed to start capture: {e}"})
            elif cmd == "stop":
                tail = cap.stop()
                if tail:
                    await send({"type": "flows", "records": tail})
                await send({"type": "stopped"})
            elif cmd == "interfaces":
                await send({"type": "interfaces",
                            "interfaces": to_dicts(list_interfaces())})
    finally:
        cap.stop()



async def run_agent(backend_ws_url: str, name: str, *,
                    connect: Callable[..., Any] = websockets.connect,
                    sniffer_factory=None, reconnect: bool = True) -> None:
    url = backend_ws_url.rstrip("/") + "/agent"
    backoff = 1.0
    while True:
        try:
            async with connect(url) as ws:
                backoff = 1.0
                await _session(ws, name, sniffer_factory)
        except asyncio.CancelledError:
            raise
        except Exception as e:                       # noqa: BLE001 - report + retry
            print(f"[agent] connection error: {e}")
        if not reconnect:
            return
        print(f"[agent] reconnecting in {backoff:.0f}s")
        await asyncio.sleep(backoff)
        backoff = min(30.0, backoff * 2)
