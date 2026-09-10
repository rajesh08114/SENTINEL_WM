"""WS /agent registration + a capture live-session end to end (fake agent)."""
from __future__ import annotations

import random
import time

from app.synth import scenarios as S


def _capture_flows(span_s=140, rate=10, seed=1) -> list[dict]:
    cfg = S.build_config("portscan", rate=rate, duration_s=span_s, seed=seed)
    rows = S.emit(0.0, span_s, cfg, random.Random(seed))
    base = 1_700_000_000.0
    for r in rows:
        r["flow_start_epoch"] = round(base + r["flow_start_epoch"], 6)
    return rows


def test_agent_registers_and_status_lists_it(client):
    with client.websocket_connect("/agent") as aws:
        aws.send_json({"type": "hello", "name": "unit-host",
                       "interfaces": [{"name": "lo", "ipv4": "127.0.0.1"}]})
        assert aws.receive_json()["type"] == "ready"

        st = client.get("/agent/status").json()
        assert st["connected"] is True
        assert st["count"] == 1
        assert st["agents"][0]["name"] == "unit-host"
        assert st["agents"][0]["interfaces"][0]["name"] == "lo"


def test_capture_session_streams_flows_from_agent(client):
    with client.websocket_connect("/agent") as aws:
        aws.send_json({"type": "hello", "name": "h", "interfaces": [{"name": "lo"}]})
        assert aws.receive_json()["type"] == "ready"

        r = client.post("/live/sessions", json={"source": "capture", "iface": "lo"})
        assert r.status_code == 201, r.text
        sid = r.json()["id"]

        start_cmd = aws.receive_json()
        assert start_cmd == {"cmd": "start", "iface": "lo", "bpf": None}

        aws.send_json({"type": "flows", "records": _capture_flows()})

        deadline = time.time() + 25
        forecasts = 0
        while time.time() < deadline:
            info = client.get(f"/live/sessions/{sid}").json()
            forecasts = info["stats"]["forecasts"]
            if forecasts >= 1:
                break
            time.sleep(0.5)
        assert forecasts >= 1, "no forecast produced from agent flows"

        assert client.delete(f"/live/sessions/{sid}").status_code == 204
        stop_cmd = aws.receive_json()
        assert stop_cmd == {"cmd": "stop"}


def test_capture_without_agent_is_503(client):
    r = client.post("/live/sessions", json={"source": "capture", "iface": "lo"})
    assert r.status_code == 503
