from scapy.layers.inet import IP, TCP, UDP

from sentinel_capture.flowmeter import FlowMeter

REQUIRED = [
    "flow_start_epoch", "Source IP", "Destination IP", "Destination Port",
    "Protocol", "Flow Duration", "Flow IAT Mean", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]


def test_syn_scan_becomes_many_short_fwd_only_flows():
    m = FlowMeter()
    for i, dport in enumerate(range(1000, 1020)):
        m.add_packet(IP(src="10.0.0.66", dst="10.0.0.5") / TCP(sport=44000 + i,
                     dport=dport, flags="S"), ts=100.0 + i * 0.001)
    rows = m.harvest(now=200.0)                 # idle-timed-out
    assert len(rows) == 20
    for r in rows:
        for c in REQUIRED:
            assert c in r
        assert r["flag_true_syn"] == 1
        assert r["Total Backward Packets"] == 0
        assert r["Total Fwd Packets"] == 1
        assert r["Protocol"] == 6
        assert r["Flow Duration"] == 0.0


def test_full_tcp_conversation_is_one_bidirectional_flow():
    m = FlowMeter()
    c, s = "10.0.0.9", "93.184.216.34"
    cp, sp = 51000, 80
    seq = [
        (IP(src=c, dst=s) / TCP(sport=cp, dport=sp, flags="S"), 0.0),
        (IP(src=s, dst=c) / TCP(sport=sp, dport=cp, flags="SA"), 0.05),
        (IP(src=c, dst=s) / TCP(sport=cp, dport=sp, flags="A"), 0.06),
        (IP(src=c, dst=s) / TCP(sport=cp, dport=sp, flags="PA") / ("x" * 200), 0.10),
        (IP(src=s, dst=c) / TCP(sport=sp, dport=cp, flags="PA") / ("y" * 500), 0.20),
        (IP(src=c, dst=s) / TCP(sport=cp, dport=sp, flags="PA") / ("x" * 120), 0.30),
        (IP(src=s, dst=c) / TCP(sport=sp, dport=cp, flags="PA") / ("y" * 800), 0.40),
        (IP(src=c, dst=s) / TCP(sport=cp, dport=sp, flags="FA"), 0.50),
    ]
    for pkt, ts in seq:
        m.add_packet(pkt, ts=ts)
    assert m.pending() == 1
    rows = m.harvest(now=0.51)                  # FIN -> closed, harvested now
    assert len(rows) == 1
    r = rows[0]
    assert r["Source IP"] == c and r["Destination IP"] == s
    assert r["Destination Port"] == 80
    assert r["Total Fwd Packets"] == 5
    assert r["Total Backward Packets"] == 3
    assert r["Total Length of Fwd Packets"] > 0
    assert r["Total Length of Bwd Packets"] > 0
    assert r["flag_true_syn"] == 1 and r["flag_true_fin"] == 1
    assert 490_000 < r["Flow Duration"] < 510_000     # ~0.5 s in microseconds


def test_idle_eviction_waits_for_timeout():
    m = FlowMeter(idle_timeout=15.0)
    m.add_packet(IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=5000, dport=53), ts=0.0)
    assert m.harvest(now=10.0) == []
    assert m.pending() == 1
    rows = m.harvest(now=20.0)
    assert len(rows) == 1 and rows[0]["Protocol"] == 17


def test_udp_bidirectional_single_flow():
    m = FlowMeter()
    m.add_packet(IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=5000, dport=53), ts=0.0)
    m.add_packet(IP(src="10.0.0.2", dst="10.0.0.1") / UDP(sport=53, dport=5000), ts=0.1)
    rows = m.harvest(now=100.0)
    assert len(rows) == 1
    assert rows[0]["Total Fwd Packets"] == 1
    assert rows[0]["Total Backward Packets"] == 1
