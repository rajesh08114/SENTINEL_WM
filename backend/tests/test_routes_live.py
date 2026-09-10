"""/live/sessions REST + WS, end to end against the tiny random-weight bundle."""
from __future__ import annotations

import time


def test_create_list_get_delete_synthetic(client):
    r = client.post("/live/sessions", json={
        "source": "synthetic", "scenario": "portscan",
        "rate": 8, "duration_s": 30, "seed": 1, "speed": 5})
    assert r.status_code == 201, r.text
    info = r.json()
    sid = info["id"]
    assert info["source_kind"] == "synthetic"
    assert info["state"] in ("starting", "running")

    lst = client.get("/live/sessions").json()
    assert any(s["id"] == sid for s in lst)

    assert client.get(f"/live/sessions/{sid}").status_code == 200
    assert client.delete(f"/live/sessions/{sid}").status_code == 204
    assert client.get(f"/live/sessions/{sid}").status_code == 404


def test_bad_scenario_422_and_capture_503(client):
    assert client.post("/live/sessions",
                       json={"source": "synthetic", "scenario": "nope"}).status_code == 422
    assert client.post("/live/sessions",
                       json={"source": "capture", "iface": "lo"}).status_code == 503
    assert client.post("/live/sessions",
                       json={"source": "bogus"}).status_code == 422


def test_ws_streams_status_then_forecast(client):
    r = client.post("/live/sessions", json={
        "source": "synthetic", "scenario": "dos_hulk",
        "rate": 12, "duration_s": 600, "seed": 2, "speed": 40, "explain": False})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    got_status = got_forecast = False
    deadline = time.time() + 30
    with client.websocket_connect(f"/live/sessions/{sid}/stream") as ws:
        first = ws.receive_json()
        assert first["type"] == "status"
        got_status = True
        while time.time() < deadline and not got_forecast:
            msg = ws.receive_json()
            if msg["type"] == "forecast":
                got_forecast = True
                assert "horizon" in msg and len(msg["horizon"]) == 6
            elif msg["type"] == "bye":
                break
    client.delete(f"/live/sessions/{sid}")
    assert got_status and got_forecast, "no forecast frame arrived in time"
