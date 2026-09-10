"""LiveSession + LiveManager unit tests (no bundle, no real windower)."""
from __future__ import annotations

import asyncio

import pytest

from app.live.errors import LiveCapacityError, LiveNotFound
from app.live.session import LiveManager, LiveSession


class FakeWindower:
    """add N rows; once >= threshold total, poll_ready yields one canned forecast."""

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.total = 0
        self._fired = False
        self.flushed = False

    def add_flows(self, records):
        self.total += len(records)
        return len(records)

    def poll_ready(self):
        if not self._fired and self.total >= self.threshold:
            self._fired = True
            return [{"alert": True, "first_alert_k": 1, "horizon": [{"k": 0}]}]
        return []

    def flush(self):
        self.flushed = True
        return []

    @property
    def stats(self):
        return {"buffered": 0, "total_received": self.total}


class FakeWS:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send_json(self, msg):
        self.sent.append(msg)

    async def close(self):
        self.closed = True


async def test_feed_produces_forecast_and_updates_stats():
    mgr = LiveManager(max_sessions=4)
    s = mgr.create("synthetic", {}, windower=FakeWindower(threshold=3))
    await s.feed([{"x": 1}, {"x": 2}])
    assert s.stats["forecasts"] == 0            # below threshold
    await s.feed([{"x": 3}])
    assert s.stats["forecasts"] == 1
    assert s.stats["alerts"] == 1
    assert len(s.ring) == 1
    assert s.state == "running"


async def test_subscriber_gets_status_then_ring_then_live():
    mgr = LiveManager()
    s = mgr.create("synthetic", {}, windower=FakeWindower(threshold=1))
    await s.feed([{"x": 1}])                    # one forecast now in the ring
    ws = FakeWS()
    await s.subscribe(ws)
    assert ws.sent[0]["type"] == "status"
    assert ws.sent[1]["type"] == "forecast"     # ring replay
    fw = s._win
    fw._fired = False
    fw.total = 0
    await s.feed([{"x": 1}])                    # a fresh live forecast
    assert ws.sent[-1]["type"] == "forecast"


async def test_manager_capacity_and_lookup():
    mgr = LiveManager(max_sessions=2)
    a = mgr.create("synthetic", {}, windower=FakeWindower())
    mgr.create("synthetic", {}, windower=FakeWindower())
    with pytest.raises(LiveCapacityError):
        mgr.create("synthetic", {}, windower=FakeWindower())
    with pytest.raises(LiveNotFound):
        mgr.get("nope")
    await mgr.remove(a.id)
    with pytest.raises(LiveNotFound):
        mgr.get(a.id)
    # slot freed
    mgr.create("synthetic", {}, windower=FakeWindower())


async def test_stop_flushes_cancels_and_notifies():
    mgr = LiveManager()
    s = mgr.create("synthetic", {}, windower=FakeWindower())
    ws = FakeWS()
    await s.subscribe(ws)

    async def forever():
        await asyncio.sleep(3600)

    t = asyncio.create_task(forever())
    s.attach_task(t)
    await s.stop()
    assert s.state == "stopped"
    assert s._win.flushed is True
    assert t.cancelled()
    assert ws.sent[-1]["type"] == "bye"
    assert ws.closed is True
