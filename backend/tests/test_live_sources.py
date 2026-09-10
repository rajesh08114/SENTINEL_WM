"""SyntheticSource / AgentSource behaviour."""
from __future__ import annotations

import asyncio

import pytest

from app.live.sources import AgentSource, SyntheticSource
from app.synth import scenarios as S


class FakeSession:
    def __init__(self):
        self.fed_rows = 0
        self.feeds = 0
        self.stopped = False

    async def feed(self, rows):
        self.fed_rows += len(rows)
        self.feeds += 1

    async def stop(self):
        self.stopped = True


async def test_synthetic_source_runs_to_completion():
    cfg = S.build_config("portscan", rate=20, duration_s=1, seed=3)
    sess = FakeSession()
    src = SyntheticSource(sess, cfg)
    await asyncio.wait_for(src.run(), timeout=4)
    assert sess.fed_rows > 0
    assert sess.stopped is True


async def test_request_stop_ends_run_promptly():
    cfg = S.build_config("dos_hulk", rate=20, duration_s=30, seed=1)
    sess = FakeSession()
    src = SyntheticSource(sess, cfg)
    task = asyncio.create_task(src.run())
    await asyncio.sleep(0.6)
    src.request_stop()
    await asyncio.wait_for(task, timeout=3)
    assert sess.stopped is True


async def test_agent_source_shell_is_safe_without_agent():
    sess = FakeSession()
    src = AgentSource(sess, agent=None, iface="lo", bpf=None)
    await src.start()
    await src.on_flows([{"a": 1}, {"a": 2}])
    await src.stop()
    assert sess.fed_rows == 2


async def test_agent_source_sends_commands_when_bound():
    sent = []

    class FakeAgent:
        async def send_cmd(self, d):
            sent.append(d)

    src = AgentSource(FakeSession(), agent=FakeAgent(), iface="eth0", bpf="host 1.2.3.4")
    await src.start()
    await src.stop()
    assert sent[0] == {"cmd": "start", "iface": "eth0", "bpf": "host 1.2.3.4"}
    assert sent[1] == {"cmd": "stop"}
