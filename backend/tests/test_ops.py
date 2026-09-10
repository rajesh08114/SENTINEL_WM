"""Request-id middleware, JSON logging, and the enriched /health payload."""
from __future__ import annotations

import json
import logging

from app.logging import RequestIdMiddleware, configure_logging, current_request_id


def test_request_id_roundtrips(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers.get("X-Request-ID") == "abc123"
    r2 = client.get("/health")
    assert r2.headers.get("X-Request-ID")            # generated when absent


def test_json_logging_emits_parseable_lines(capsys):
    configure_logging(json_mode=True)
    logging.getLogger("test").warning("hello ops")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    obj = json.loads(line)
    assert obj["level"] == "WARNING" and obj["msg"] == "hello ops"
    configure_logging(json_mode=False)              # restore for other tests


def test_health_has_live_agent_uptime_detail(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["bundle"]["loaded"] is True
    assert set(h["live"]) == {"sessions", "max"}
    assert set(h["agent"]) == {"connected", "count"}
    assert isinstance(h["uptime_s"], (int, float))
