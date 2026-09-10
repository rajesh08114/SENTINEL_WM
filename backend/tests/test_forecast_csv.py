"""End-to-end: CSV upload -> forecast JSON with the full per-horizon + ATT&CK
+ explanation contract. Numbers are random (throwaway bundle); structure is real."""
from __future__ import annotations


def test_meta(client):
    r = client.get("/meta")
    assert r.status_code == 200
    m = r.json()
    assert m["history_windows"] == 12 and m["horizon_steps"] == 6
    assert m["feature_dim"] == len(m["feature_names"])
    assert m["progression_states"][0] == "NORMAL"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_forecast_csv_contract(client, synth_csv_bytes):
    r = client.post("/forecast/csv",
                    files={"file": ("flows.csv", synth_csv_bytes, "text/csv")},
                    data={"explain": "true", "family_hint": "PortScan"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["meta"]["n_flows"] > 300
    assert body["meta"]["window_seconds"] == 10
    assert body["meta"]["family_hint"] == "PortScan"
    assert body["meta"]["n_anchors"] >= 1
    assert set(body["summary"]) >= {"n_alerts", "max_attack_prob", "phases"}

    a = body["anchors"][0]
    assert len(a["horizon"]) == 6
    assert "driving_features" in a and a["driving_features"] is not None
    for k, h in enumerate(a["horizon"], start=1):
        assert h["k"] == k
        assert h["horizon_seconds"] == k * 10
        assert 0.0 <= h["attack_prob"] <= 1.0
        assert len(h["attack_ci"]) == 2
        assert abs(sum(h["progression_dist"].values()) - 1.0) < 1e-3
        att = h["attck"]
        assert att["kill_chain_phase"]
        assert att["confidence"] in {"High", "Medium", "Low"}
        assert isinstance(att["technique_ids"], list)


def test_forecast_csv_bad_schema(client):
    r = client.post("/forecast/csv",
                    files={"file": ("x.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 422
    assert "missing required columns" in r.text or "flow_start_epoch" in r.text


def test_forecast_csv_no_explain(client, synth_csv_bytes):
    r = client.post("/forecast/csv",
                    files={"file": ("flows.csv", synth_csv_bytes, "text/csv")},
                    data={"explain": "false"})
    assert r.status_code == 200
    assert r.json()["anchors"][0].get("driving_features") is None
