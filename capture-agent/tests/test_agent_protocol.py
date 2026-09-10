"""Agent protocol handling with a fake socket and a stub sniffer (no live capture)."""
from __future__ import annotations

import json

from sentinel_capture.agent import run_agent


class FakeWS:
    def __init__(self, incoming: list[dict]):
        self._incoming = [json.dumps(m) for m in incoming]
        self.sent: list[dict] = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        self._it = iter(self._incoming)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class StubSniffer:
    def __init__(self, *a, **k):
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


async def test_hello_and_interfaces_and_start_stop():
    ws = FakeWS([{"cmd": "interfaces"}, {"cmd": "start", "iface": "lo", "bpf": None},
                {"cmd": "stop"}])

    def connect(url):
        assert url.endswith("/agent")
        return ws

    made = []

    def sniffer_factory(iface, bpf, on_packet):
        s = StubSniffer()
        made.append((iface, s))
        return s

    await run_agent("ws://localhost:8000", "unit-host", connect=connect,
                    sniffer_factory=sniffer_factory, reconnect=False)

    types = [m["type"] for m in ws.sent]
    assert types[0] == "hello"
    assert ws.sent[0]["name"] == "unit-host"
    assert len(ws.sent[0]["interfaces"]) >= 1
    assert "interfaces" in types
    assert "started" in types and "stopped" in types
    assert made and made[0][0] == "lo" and made[0][1].started and made[0][1].stopped
