"""POST /forecast/pcap — offline capture -> reassembled flows -> forecast."""
from __future__ import annotations

import os
import tempfile

import pytest

scapy = pytest.importorskip("scapy.all")
from scapy.all import IP, TCP, UDP, Ether, wrpcap  # noqa: E402


def _demo_pcap(seconds: float = 200.0, base: float = 1_700_000_000.0) -> bytes:
    """A benign-ish capture: many short web conversations over `seconds`, plus a
    late SYN-scan burst. Enough wall-clock span for >= 13 ten-second windows."""
    pkts = []
    n = int(seconds / 0.4)
    for i in range(n):
        t = base + i * 0.4
        c, s = "10.0.0.9", f"93.184.216.{(i % 5) + 10}"
        cp = 40000 + (i % 2000)
        for pk in (
            IP(src=c, dst=s) / TCP(sport=cp, dport=443, flags="S"),
            IP(src=s, dst=c) / TCP(sport=443, dport=cp, flags="SA"),
            IP(src=c, dst=s) / TCP(sport=cp, dport=443, flags="A") / (b"x" * 200),
            IP(src=s, dst=c) / TCP(sport=443, dport=cp, flags="PA") / (b"y" * 400),
            IP(src=c, dst=s) / TCP(sport=cp, dport=443, flags="FA"),
        ):
            p = Ether() / pk
            p.time = t
            pkts.append(p)
        if i % 7 == 0:
            u = Ether() / IP(src=c, dst="10.0.0.1") / UDP(sport=cp, dport=53) / (b"q" * 30)
            u.time = t + 0.05
            pkts.append(u)
    # late scan burst
    for j in range(120):
        p = Ether() / IP(src="10.0.0.66", dst="10.0.0.5") / TCP(
            sport=50000 + j, dport=1000 + j, flags="S")
        p.time = base + seconds - 20 + j * 0.1
        pkts.append(p)

    fh = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
    fh.close()
    try:
        wrpcap(fh.name, pkts)
        with open(fh.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(fh.name)


def test_forecast_pcap_contract(client):
    data = _demo_pcap()
    r = client.post(
        "/forecast/pcap",
        files={"file": ("demo.pcap", data, "application/vnd.tcpdump.pcap")},
        data={"explain": "false"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["n_flows"] > 100
    assert body["meta"]["window_seconds"] == 10
    assert body["meta"]["n_anchors"] >= 1
    a = body["anchors"][0]
    assert len(a["horizon"]) == 6
    for h in a["horizon"]:
        assert 0.0 <= h["attack_prob"] <= 1.0
        assert h["attck"]["kill_chain_phase"]


def test_forecast_pcap_bad_file(client):
    r = client.post(
        "/forecast/pcap",
        files={"file": ("x.pcap", b"not a pcap at all, just text", "application/octet-stream")},
    )
    assert r.status_code == 422


def test_forecast_pcap_too_large(client, monkeypatch):
    import app.api.routes_forecast as RF
    monkeypatch.setattr(RF.settings, "pcap_max_bytes", 10, raising=False)
    r = client.post(
        "/forecast/pcap",
        files={"file": ("big.pcap", b"0" * 5000, "application/octet-stream")},
    )
    assert r.status_code == 413
