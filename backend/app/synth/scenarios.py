"""Deterministic synthetic scenario generator.

`emit(t0, t1, cfg, rng)` returns the CICFlowMeter-schema flow rows whose
``flow_start_epoch`` falls in ``[t0, t1)`` -- a benign baseline always, plus
attack rows for whichever ramp phase is active at that time.  All randomness
comes from the caller-supplied ``random.Random`` so a seed fully reproduces a
run.  The rows feed straight into ``StreamingWindower.add_flows`` (which calls
``normalise_upload`` -> the ``Label``/``attack_family`` we set are display-only).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from app.settings import settings

# columns windows._agg_windows / normalise_upload index without a guard
REQUIRED = [
    "flow_start_epoch",
    "Source IP", "Destination IP", "Destination Port", "Protocol",
    "Flow Duration", "Flow IAT Mean",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
]

_FLAGS = ("fin", "syn", "rst", "psh", "ack", "urg")

_INTERNAL = [f"10.0.0.{i}" for i in range(10, 60)]
_EXTERNAL = ["93.184.216.34", "142.250.72.196", "151.101.1.140",
             "104.16.132.229", "13.107.42.14", "198.51.100.7"]
_BENIGN_DPORTS = [80, 443, 443, 443, 53, 22, 3389, 8080]

PHASES = ("benign", "pre_attack", "onset", "active", "continuation")
_DEFAULT_SCHEDULE = [(0.00, "benign"), (0.10, "pre_attack"), (0.25, "onset"),
                     (0.45, "active"), (0.80, "continuation")]
# attack-volume multiplier per phase (monotone: the intrusion establishes and
# sustains rather than spiking then fading)
_PHASE_GAIN = {"benign": 0.0, "pre_attack": 0.15, "onset": 0.5,
               "active": 0.9, "continuation": 1.0}


@dataclass
class ScenarioConfig:
    name: str
    rate: float = 40.0            # benign flows / second (scenario time)
    duration_s: float = 300.0     # scenario-time length
    seed: int = 0
    speed: float = 1.0            # scenario seconds advanced per wall second
    attacker_ip: str = "10.0.0.66"
    victim_ip: str = "10.0.0.20"
    c2_ip: str = "198.51.100.7"
    phase_schedule: list[tuple[float, str]] = field(
        default_factory=lambda: list(_DEFAULT_SCHEDULE))


# preset defaults; caller overrides rate/duration/seed/ips via build_config
SCENARIOS: dict[str, ScenarioConfig] = {
    "benign":     ScenarioConfig("benign"),
    "portscan":   ScenarioConfig("portscan"),
    "dos_hulk":   ScenarioConfig("dos_hulk", rate=50.0),
    "bruteforce": ScenarioConfig("bruteforce"),
    "botnet_c2":  ScenarioConfig("botnet_c2"),
    "exfil":      ScenarioConfig("exfil"),
}


def build_config(name: str, *, rate: Optional[float] = None,
                 duration_s: Optional[float] = None, seed: Optional[int] = None,
                 speed: Optional[float] = None,
                 attacker_ip: Optional[str] = None,
                 victim_ip: Optional[str] = None) -> ScenarioConfig:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(SCENARIOS)}")
    base = SCENARIOS[name]
    cfg = ScenarioConfig(
        name=name,
        rate=float(base.rate if rate is None else rate),
        duration_s=float(base.duration_s if duration_s is None else duration_s),
        seed=int(base.seed if seed is None else seed),
        speed=float(base.speed if speed is None else speed),
        attacker_ip=attacker_ip or base.attacker_ip,
        victim_ip=victim_ip or base.victim_ip,
        c2_ip=base.c2_ip,
        phase_schedule=list(base.phase_schedule),
    )
    cfg.rate = max(1.0, min(cfg.rate, float(settings.synth_max_rate)))
    cfg.duration_s = max(1.0, cfg.duration_s)
    cfg.speed = max(0.1, min(cfg.speed, 120.0))
    return cfg


def phase_at(cfg: ScenarioConfig, frac: float) -> str:
    frac = max(0.0, min(frac, 0.999999))
    label = "benign"
    for start, name in cfg.phase_schedule:
        if frac >= start:
            label = name
        else:
            break
    return label


# ---------------------------------------------------------------------------
def _flags(**on: bool) -> dict:
    return {f"flag_true_{f}": int(bool(on.get(f, False))) for f in _FLAGS}


def _row(epoch: float, src: str, dst: str, sport: int, dport: int, proto: int,
         fwd_pkts: int, bwd_pkts: int, fwd_bytes: int, bwd_bytes: int,
         dur_s: float, flags: dict, family: str, ttl: int = 64) -> dict:
    npk = max(1, fwd_pkts + bwd_pkts)
    dur_us = max(1.0, dur_s * 1_000_000.0)
    iat_mean = dur_us / max(1, npk - 1)
    row = {
        "flow_start_epoch": round(epoch, 6),
        "Source IP": src, "Destination IP": dst,
        "Source Port": int(sport), "Destination Port": int(dport),
        "Protocol": int(proto),
        "Flow Duration": round(dur_us, 1),
        "Flow IAT Mean": round(iat_mean, 3),
        "Total Fwd Packets": int(fwd_pkts),
        "Total Backward Packets": int(bwd_pkts),
        "Total Length of Fwd Packets": int(fwd_bytes),
        "Total Length of Bwd Packets": int(bwd_bytes),
        "Fwd IAT Total": round(dur_us * fwd_pkts / npk, 1),
        "Bwd IAT Total": round(dur_us * bwd_pkts / npk, 1),
        "pkt_len_mean": round((fwd_bytes + bwd_bytes) / npk, 2),
        "ttl_mean": ttl,
        "Label": "BENIGN" if family == "BENIGN" else family,
        "attack_family": family,
    }
    row.update(flags)
    return row


def _benign_flow(rng: random.Random, epoch: float) -> dict:
    src = rng.choice(_INTERNAL)
    if rng.random() < 0.7:
        dst = rng.choice(_EXTERNAL)
    else:
        dst = rng.choice([ip for ip in _INTERNAL if ip != src])
    dport = rng.choice(_BENIGN_DPORTS)
    proto = 17 if dport == 53 else 6
    fwd = rng.randint(2, 18)
    bwd = rng.randint(1, max(1, fwd))
    fwd_b = fwd * rng.randint(60, 700)
    bwd_b = bwd * rng.randint(80, 1400)
    dur = rng.uniform(0.005, 4.0)
    if proto == 6:
        fl = _flags(ack=True, psh=rng.random() < 0.6, syn=rng.random() < 0.25,
                    fin=rng.random() < 0.3)
    else:
        fl = _flags()
    return _row(epoch, src, dst, rng.randint(1024, 65000), dport, proto,
                fwd, bwd, fwd_b, bwd_b, dur, fl, "BENIGN",
                ttl=rng.choice([64, 128]))


def _attack_flows(rng: random.Random, b0: float, b1: float,
                  cfg: ScenarioConfig, gain: float) -> list[dict]:
    if gain <= 0:
        return []
    a, v = cfg.attacker_ip, cfg.victim_ip
    ts = lambda: rng.uniform(b0, b1)          # noqa: E731 - local shorthand
    out: list[dict] = []
    if cfg.name == "portscan":
        n = max(1, int(round(rng.uniform(8, 26) * gain * (b1 - b0) * 2)))
        for _ in range(n):
            dport = rng.randint(1, 65535)
            out.append(_row(ts(), a, v,
                            rng.randint(40000, 65000), dport, 6,
                            rng.randint(1, 2), 0, rng.randint(40, 120), 0,
                            rng.uniform(0.0001, 0.01),
                            _flags(syn=True), "PortScan"))
    elif cfg.name == "dos_hulk":
        n = max(1, int(round(rng.uniform(10, 30) * gain * (b1 - b0) * 2)))
        for _ in range(n):
            fwd = rng.randint(400, 1800)
            out.append(_row(ts(), a, v,
                            rng.randint(40000, 65000), 80, 6,
                            fwd, rng.randint(0, 5), fwd * rng.randint(200, 900),
                            rng.randint(0, 400), rng.uniform(0.2, 2.0),
                            _flags(syn=True, ack=True, psh=True), "DoS"))
    elif cfg.name == "bruteforce":
        n = max(1, int(round(rng.uniform(5, 16) * gain * (b1 - b0) * 2)))
        for _ in range(n):
            dport = rng.choice([22, 3389, 21])
            out.append(_row(ts(), a, v,
                            rng.randint(40000, 65000), dport, 6,
                            rng.randint(5, 15), rng.randint(3, 12),
                            rng.randint(200, 900), rng.randint(200, 700),
                            rng.uniform(0.05, 0.8),
                            _flags(syn=True, rst=True, fin=True, ack=True),
                            "BruteForce"))
    elif cfg.name == "botnet_c2":
        # periodic small beacons + occasional larger task pulls
        if int(b0) % 10 == 0 or rng.random() < 0.15 * gain:
            out.append(_row(ts(), a, cfg.c2_ip, rng.randint(40000, 65000),
                            rng.choice([443, 8080, 6667]), 6,
                            rng.randint(2, 6), rng.randint(2, 8),
                            rng.randint(120, 400), rng.randint(200, 1200),
                            rng.uniform(0.02, 0.5),
                            _flags(ack=True, psh=True), "Bot"))
        if rng.random() < 0.08 * gain:
            out.append(_row(ts(), cfg.c2_ip, a, rng.choice([443, 8080]),
                            rng.randint(40000, 65000), 6,
                            rng.randint(20, 80), rng.randint(40, 200),
                            rng.randint(2000, 9000), rng.randint(20000, 90000),
                            rng.uniform(0.5, 3.0),
                            _flags(ack=True, psh=True), "Bot"))
    elif cfg.name == "exfil":
        if rng.random() < 0.6 * gain + 0.05:
            fwd = rng.randint(2000, 9000)
            out.append(_row(ts(), a, rng.choice(_EXTERNAL),
                            rng.randint(40000, 65000), 443, 6,
                            fwd, rng.randint(50, 400),
                            fwd * rng.randint(600, 1400), rng.randint(2000, 40000),
                            rng.uniform(2.0, 8.0),
                            _flags(ack=True, psh=True), "Exfiltration"))
    return [r for r in out if b0 <= r["flow_start_epoch"] < b1]


def emit(t0: float, t1: float, cfg: ScenarioConfig,
         rng: random.Random) -> list[dict]:
    """Flow rows with flow_start_epoch in [t0, t1). Handles a slice spanning
    multiple ramp phases by bucketing internally."""
    if t1 <= t0:
        return []
    rows: list[dict] = []
    step = 0.5
    n_buckets = max(1, int(round((t1 - t0) / step)))
    bstep = (t1 - t0) / n_buckets
    for b in range(n_buckets):
        b0 = t0 + b * bstep
        b1 = t0 + (b + 1) * bstep
        mid = 0.5 * (b0 + b1)
        gain = _PHASE_GAIN[phase_at(cfg, mid / cfg.duration_s)]
        n_benign = _poisson(rng, cfg.rate * (b1 - b0))
        for _ in range(n_benign):
            rows.append(_benign_flow(rng, rng.uniform(b0, b1)))
        rows.extend(_attack_flows(rng, b0, b1, cfg, gain))
    rows.sort(key=lambda r: r["flow_start_epoch"])
    return rows


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm; fine for the small lambdas here."""
    if lam <= 0:
        return 0
    import math
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1
