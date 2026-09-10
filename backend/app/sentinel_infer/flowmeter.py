"""Packet -> bidirectional flow assembler.

Vendored verbatim from ``capture-agent/sentinel_capture/flowmeter.py`` (keep the
two in sync; ``tests/test_vendor_sync.py`` guards the shared contracts). Used by
``pcap.py`` for offline PCAP ingestion.

Not a full CICFlowMeter clone: it produces the columns the SENTINEL-WM model
actually consumes (the 11 required + TCP flag booleans + a couple of timing
fields).  Everything else the pipeline wants defaults to 0, exactly as an
uploaded CSV would.

    m = FlowMeter()
    m.add_packet(pkt, ts)        # per sniffed packet
    rows = m.harvest(time.time())  # closed / idle / over-long flows -> dict rows
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:                                    # scapy layers (import lazily-safe)
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except Exception:                       # pragma: no cover - scapy always present in practice
    IP = TCP = UDP = IPv6 = tuple()     # type: ignore

_FLAG_BITS = {"fin": 0x01, "syn": 0x02, "rst": 0x04,
              "psh": 0x08, "ack": 0x10, "urg": 0x20}


@dataclass
class FlowState:
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    first_ts: float
    last_ts: float
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_iat_total: float = 0.0
    bwd_iat_total: float = 0.0
    _fwd_last: Optional[float] = None
    _bwd_last: Optional[float] = None
    flags: dict = field(default_factory=lambda: {k: 0 for k in _FLAG_BITS})
    len_sum: int = 0
    len_cnt: int = 0
    closed: bool = False


def _endpoints(pkt: Any):
    """-> (proto, src, dst, sport, dport) or None if not IP/TCP/UDP."""
    if IP and IP in pkt:
        ipl = pkt[IP]
        src, dst = ipl.src, ipl.dst
    elif IPv6 and IPv6 in pkt:
        ipl = pkt[IPv6]
        src, dst = ipl.src, ipl.dst
    else:
        return None
    if TCP and TCP in pkt:
        l4 = pkt[TCP]
        return 6, src, dst, int(l4.sport), int(l4.dport)
    if UDP and UDP in pkt:
        l4 = pkt[UDP]
        return 17, src, dst, int(l4.sport), int(l4.dport)
    return None


class FlowMeter:
    def __init__(self, idle_timeout: float = 15.0, active_timeout: float = 120.0):
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self._flows: dict[tuple, FlowState] = {}

    def pending(self) -> int:
        return len(self._flows)

    def add_packet(self, pkt: Any, ts: Optional[float] = None) -> None:
        info = _endpoints(pkt)
        if info is None:
            return
        proto, src, dst, sport, dport = info
        if ts is None:
            ts = float(getattr(pkt, "time", 0.0)) or time.time()
        ea, eb = (src, sport), (dst, dport)
        key = (proto, *sorted((ea, eb)))

        st = self._flows.get(key)
        if st is None:
            st = FlowState(src=src, dst=dst, sport=sport, dport=dport,
                           proto=proto, first_ts=ts, last_ts=ts)
            self._flows[key] = st

        forward = (src, sport) == (st.src, st.sport)
        length = len(pkt)
        st.last_ts = ts
        st.len_sum += length
        st.len_cnt += 1
        if forward:
            st.fwd_pkts += 1
            st.fwd_bytes += length
            if st._fwd_last is not None:
                st.fwd_iat_total += ts - st._fwd_last
            st._fwd_last = ts
        else:
            st.bwd_pkts += 1
            st.bwd_bytes += length
            if st._bwd_last is not None:
                st.bwd_iat_total += ts - st._bwd_last
            st._bwd_last = ts

        if proto == 6 and TCP and TCP in pkt:
            bits = int(pkt[TCP].flags)
            for name, mask in _FLAG_BITS.items():
                if bits & mask:
                    st.flags[name] += 1
            if bits & _FLAG_BITS["rst"] or bits & _FLAG_BITS["fin"]:
                st.closed = True

    def harvest(self, now: Optional[float] = None) -> list[dict]:
        if now is None:
            now = time.time()
        done: list[tuple] = []
        for key, st in self._flows.items():
            if (st.closed
                    or now - st.last_ts > self.idle_timeout
                    or now - st.first_ts > self.active_timeout):
                done.append(key)
        rows = [self._row(self._flows.pop(k)) for k in done]
        rows.sort(key=lambda r: r["flow_start_epoch"])
        return rows

    def flush_all(self) -> list[dict]:
        rows = [self._row(st) for st in self._flows.values()]
        self._flows.clear()
        rows.sort(key=lambda r: r["flow_start_epoch"])
        return rows

    # ---------------------------------------------------------------
    @staticmethod
    def _row(st: FlowState) -> dict:
        total = max(1, st.fwd_pkts + st.bwd_pkts)
        dur_us = max(0.0, (st.last_ts - st.first_ts) * 1_000_000.0)
        row = {
            "flow_start_epoch": round(st.first_ts, 6),
            "Source IP": st.src, "Destination IP": st.dst,
            "Source Port": st.sport, "Destination Port": st.dport,
            "Protocol": st.proto,
            "Flow Duration": round(dur_us, 1),
            "Flow IAT Mean": round(dur_us / max(1, total - 1), 3),
            "Total Fwd Packets": st.fwd_pkts,
            "Total Backward Packets": st.bwd_pkts,
            "Total Length of Fwd Packets": st.fwd_bytes,
            "Total Length of Bwd Packets": st.bwd_bytes,
            "Fwd IAT Total": round(st.fwd_iat_total * 1_000_000.0, 1),
            "Bwd IAT Total": round(st.bwd_iat_total * 1_000_000.0, 1),
            "pkt_len_mean": round(st.len_sum / max(1, st.len_cnt), 2),
        }
        for name in _FLAG_BITS:
            row[f"flag_true_{name}"] = 1 if st.flags[name] else 0
        return row
