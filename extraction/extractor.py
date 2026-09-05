#!/usr/bin/env python3
# =============================================================================
# CIC-IDS2017 Unified Flow + Packet-Level Extractor
#
# There are TWO different targets and they are not the same thing. Trying to hit
# both with one code path is what makes these ports wrong. So this file has two
# explicit profiles:
#
#   PROFILE = "CICIDS2017"   reproduce the released *_ISCX.csv files
#                            (GeneratedLabelledFlows / MachineLearningCVE)
#   PROFILE = "STRICT_V4"    reproduce today's ahlashkari/CICFlowMeter master
#
# The released CSVs were produced by a build whose source was never published
# (the dataset files end in _ISCX.csv; every published build emits _Flow.csv).
# Rosay et al., "Network Intrusion Detection: A Comprehensive Analysis of
# CIC-IDS2017" (ICISSP 2022) reverse-engineered that build's behaviour against
# the PCAPs. Where their findings and the current master source disagree, the
# CICIDS2017 profile follows their findings and STRICT_V4 follows the source.
#
# Evidence tags used in the comments below:
#   [SRC]   read directly from current master Java source
#   [CSV]   observed in released CIC-IDS2017 CSV data
#   [PUB]   reported by Rosay et al. 2022 from dataset-wide analysis
#   [INF]   inferred; reconciles [SRC]+[CSV]+[PUB] but not directly verified
#
# Anything tagged [INF] is a hypothesis. Verify on one day before trusting it.
# Run `python cicflow_extractor.py --self-test` for a regression against a
# known-good released row.
# =============================================================================

import os
import gc
import math
import subprocess
from collections import defaultdict, Counter
from datetime import datetime
import zoneinfo

DATASET_TZ = zoneinfo.ZoneInfo("America/Halifax")

import numpy as np
import pandas as pd

PROFILE = "CICIDS2017"          # or "STRICT_V4"

# -----------------------------------------------------------------------------
PCAP_DIR   = "/content/drive/MyDrive/CICIDS2017/PCAPs"
OUTPUT_DIR = "/content/drive/MyDrive/SIH 2026-153/Data Sets/CIC-IDS2017/unified_features"
LOCAL_TMP_DIR  = "/content/tshark_tmp"
LOCAL_PCAP_TMP = "/content/tmp_pcap"

# CICFlowMeter.java: new FlowGenerator(true, 120000000L, 5000000L)        [SRC]
FLOW_TIMEOUT_US     = 120_000_000
ACTIVITY_TIMEOUT_US = 5_000_000
SUBFLOW_GAP_S       = 1.0
BULK_GAP_S          = 1.0
BULK_MIN_PACKETS    = 4

PROFILES = {
    # =====================================================================
    "CICIDS2017": {
        # --- flow termination -------------------------------------------
        # [PUB] The paper's 4-way-handshake walkthrough only reproduces if ANY
        # FIN closes the flow: FIN(A->B) closes flow 1; ACK(B->A) opens flow 2;
        # FIN(B->A) closes flow 2; the final ACK opens flow 3. That is exactly
        # the block sitting COMMENTED OUT in today's FlowGenerator.java. The
        # bwd+bwd==2 logic in current master is a later rewrite that produces
        # different flow boundaries, so it cannot reproduce the CSVs.
        "termination": "fin_any",
        # [PUB] "If a TCP communication is aborted by a 'RST' flag,
        # CICFlowMeter does not consider it as closing the flow."
        "rst_closes": False,
        # No fwdFIN/bwdFIN counters exist under fin_any, so no half-open guard.
        "half_close_guard": False,

        # --- feature quirks ---------------------------------------------
        # [CSV] A flow with zero backward packets carries -1, which can only
        # come from the field initialiser.
        "init_win_unset": -1,
        # [PUB] One of 4 duplicated features in the ISCX files. The released
        # CSVs are 85 columns WITH this duplicate. Do not remove it.
        "emit_dup_fwd_header_len": True,
        # [SRC]+[INF] Java 8 HashMap bucket order for the 8 flag keys.
        "flag_column_order": "hashmap",
        # [PUB] All six bulk columns are constant zero across the released
        # dataset (dropped as "always null" during feature selection).
        "bulk_mode": "zero",
        # [PUB] Bwd PSH Flags and Bwd URG Flags are likewise constant zero.
        "force_zero_bwd_psh_urg": True,
        # [PUB] "Due to misplaced parenthesis, the test is always true and so
        # the subflow count is increased for each new packet received."
        # [CSV] A 2-packet flow shows Subflow Fwd Packets == Total Fwd Packets,
        # which requires sfCount == 1, i.e. increment on every packet AFTER the
        # first. Current master's gap test would give sfCount == 0 -> 0.  [INF]
        "subflow_mode": "every_packet_after_first",
        # [SRC] DateFormatter with "dd/MM/yyyy hh:mm:ss" - lowercase hh is
        # 12-hour and there is no trailing 'a', which is precisely why
        # CIC-IDS2017 afternoon timestamps are ambiguous.
        "timestamp_fmt": "%d/%m/%Y %I:%M:%S",
    },
    # =====================================================================
    "STRICT_V4": {
        "termination": "fin_v4",         # [SRC] the bwd+bwd==2 branch
        "rst_closes": True,              # [SRC]
        "half_close_guard": True,        # [SRC]
        "init_win_unset": 0,             # [SRC] private int ... = 0
        "emit_dup_fwd_header_len": False,
        "flag_column_order": "canonical",
        "bulk_mode": "backward_only",    # [SRC] the byte[] == reference bug
        "force_zero_bwd_psh_urg": False,
        "subflow_mode": "gap",           # [SRC] (dt/1e6) > 1.0
        "timestamp_fmt": "%d/%m/%Y %I:%M:%S %p",
    },
}

COMMON_CFG = {
    # getDownUpRatio(): (double)(backward.size()/forward.size()) - the cast
    # happens after the integer division.
    "downup_int_div": True,
    # Flow Bytes/s and Flow Packets/s divide by duration with no zero guard, so
    # a zero-duration flow yields Infinity / NaN. Fwd and Bwd Packets/s DO
    # guard and return 0. This is the origin of the Infinity/NaN cells.
    "infinity_rates": True,
    # getIpv4Info() builds a BasicPacketInfo for every IPv4 packet, even when
    # neither hasHeader(tcp) nor hasHeader(udp) matches. Those carry
    # protocol=0, srcPort=0, dstPort=0, payloadBytes=0, headerBytes=0.
    "include_non_tcp_udp": True,
    "emit_true_flag_columns": True,
    "window_seconds": 5,
    # No CICFlowMeter equivalent - CICFlowMeter never evicts idle flows.
    "enable_idle_eviction": False,
    "idle_evict_seconds": 600.0,
    "max_active_flows": 0,
    "expiry_check_every": 200_000,
    "completed_flush_every": 200_000,
    "tshark_chunk_size": 200_000,
}

CFG = dict(PROFILES[PROFILE])
CFG.update(COMMON_CFG)
INIT_WIN_UNSET = CFG["init_win_unset"]


def set_profile(profile_name):
    global PROFILE, CFG, INIT_WIN_UNSET
    PROFILE = profile_name
    CFG.update(PROFILES[profile_name])
    for k, v in COMMON_CFG.items():
        CFG.setdefault(k, v)
    INIT_WIN_UNSET = CFG["init_win_unset"]
    return CFG


# =============================================================================
# SummaryStatistics - org.apache.commons.math3...SummaryStatistics
# =============================================================================
class SummaryStatistics:
    """
    getVariance() in Commons Math is the BIAS-CORRECTED SAMPLE variance
    (divide by n-1), and getStandardDeviation() is its square root, defined as
    0 when n == 1. Welford must use the old mean for one factor and the new
    mean for the other.
    """
    __slots__ = ("n", "total", "_mean", "m2", "_min", "_max")

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self._mean = 0.0
        self.m2 = 0.0
        self._min = 0.0
        self._max = 0.0

    def add(self, x):
        x = float(x)
        self.n += 1
        self.total += x
        if self.n == 1:
            self._min = self._max = x
        elif x < self._min:
            self._min = x
        elif x > self._max:
            self._max = x
        delta = x - self._mean
        self._mean += delta / self.n
        self.m2 += delta * (x - self._mean)

    def mean(self):     return self._mean if self.n else 0.0
    def sum(self):      return self.total if self.n else 0.0
    def variance(self): return self.m2 / (self.n - 1) if self.n > 1 else 0.0
    def std(self):      return math.sqrt(self.variance()) if self.n > 1 else 0.0
    def min(self):      return self._min if self.n else 0.0
    def max(self):      return self._max if self.n else 0.0


class RunningMoments:
    """Online mean/var/skew/kurtosis (Pebay). Matches scipy bias=True, Fisher."""
    __slots__ = ("n", "mean", "M2", "M3", "M4", "_min", "_max", "nonzero")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = self.M3 = self.M4 = 0.0
        self._min = self._max = 0.0
        self.nonzero = 0

    def add(self, x):
        x = float(x)
        n1 = self.n
        self.n = n = n1 + 1
        if n == 1:
            self._min = self._max = x
        elif x < self._min:
            self._min = x
        elif x > self._max:
            self._max = x
        if x != 0.0:
            self.nonzero += 1
        delta = x - self.mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self.mean += delta_n
        self.M4 += (term1 * delta_n2 * (n * n - 3 * n + 3)
                    + 6 * delta_n2 * self.M2 - 4 * delta_n * self.M3)
        self.M3 += term1 * delta_n * (n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

    def variance(self, ddof=1):
        return self.M2 / (self.n - ddof) if self.n > ddof else 0.0

    def std(self, ddof=1):
        return math.sqrt(self.variance(ddof))

    def skew(self):
        if self.n < 3 or self.M2 <= 0.0:
            return 0.0
        return math.sqrt(self.n) * self.M3 / (self.M2 ** 1.5)

    def kurtosis(self):
        if self.n < 4 or self.M2 <= 0.0:
            return 0.0
        return self.n * self.M4 / (self.M2 * self.M2) - 3.0

    def nonzero_ratio(self):
        return self.nonzero / self.n if self.n else 0.0

    def min(self): return self._min if self.n else 0.0
    def max(self): return self._max if self.n else 0.0


# =============================================================================
# tshark front end
# =============================================================================
TSHARK_FIELDS = [
    "frame.time_epoch", "frame.len",
    "ip.src", "ip.dst", "ip.proto", "ip.ttl",
    "ip.flags.mf", "ip.flags.df", "ip.frag_offset",
    "tcp.srcport", "tcp.dstport", "tcp.window_size_value",
    "tcp.len", "tcp.hdr_len", "tcp.seq", "tcp.flags",
    "tcp.analysis.retransmission",
    "udp.srcport", "udp.dstport", "udp.length",
]
UDP_HEADER_LEN = 8
FLAG_BITS = [("fin", 0x01), ("syn", 0x02), ("rst", 0x04), ("psh", 0x08),
             ("ack", 0x10), ("urg", 0x20), ("ece", 0x40), ("cwr", 0x80)]


def run_tshark_to_df_iter(pcap_path, chunksize):
    os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
    tmp_csv = os.path.join(LOCAL_TMP_DIR, os.path.basename(pcap_path) + ".csv")

    # ip.defragment:FALSE matters. jnetpcap does not reassemble, so a UDP
    # fragment after the first has no transport header and is booked as a
    # protocol-0 packet. With tshark's default reassembly it would be dissected
    # as UDP and land in a different flow.                              [PUB]
    cmd = ["tshark", "-r", pcap_path,
           "-o", "ip.defragment:FALSE",
           "-o", "tcp.desegment_tcp_streams:FALSE",
           "-T", "fields"]
    for f in TSHARK_FIELDS:
        cmd += ["-e", f]
    cmd += ["-E", "header=y", "-E", "separator=,", "-E", "quote=n",
            "-E", "occurrence=f"]

    try:
        with open(tmp_csv, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"[tshark] exit {r.returncode}:\n{r.stderr[-2000:]}")
        for chunk in pd.read_csv(tmp_csv, chunksize=chunksize, dtype=str,
                                 na_filter=False):
            yield chunk
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


def clean_chunk(chunk):
    """
    PacketReader.getIpv4Info():
        if (packet.hasHeader(ipv4)) {
            packetInfo = new BasicPacketInfo(generator);
            setSrc / setDst / setTimeStamp(timestampInMicros())
            if      (hasHeader(tcp)) { window, ports, proto=6, flags,
                                       payload=tcp.getPayloadLength(),
                                       header =tcp.getHeaderLength() }
            else if (hasHeader(udp)) { ports, proto=17,
                                       payload=udp.getPayloadLength(),
                                       header =udp.getHeaderLength()=8 }
            else                     { nothing set -> all defaults }
        }
    The else branch still returns a packet: ports 0, protocol 0, payload 0,
    header 0, every flag false. Dropping those rows removes an entire class of
    flows the released CSVs contain (protocol 0, port 0).            [SRC][PUB]
    Runs with readIP4=true / readIP6=false, so IPv6 never enters.
    """
    chunk = chunk.rename(columns={"ip.src": "src_ip", "ip.dst": "dst_ip"})
    chunk = chunk[chunk["src_ip"] != ""]                     # IPv4 only
    if chunk.empty:
        return chunk

    is_tcp = (chunk["tcp.srcport"] != "").values
    is_udp = (~is_tcp) & (chunk["udp.srcport"] != "").values
    is_other = ~(is_tcp | is_udp)

    if not CFG["include_non_tcp_udp"]:
        keep = ~is_other
        chunk = chunk[keep]
        is_tcp, is_udp = is_tcp[keep], is_udp[keep]
        is_other = np.zeros(len(chunk), dtype=bool)
        if chunk.empty:
            return chunk

    ts = pd.to_numeric(chunk["frame.time_epoch"], errors="coerce")
    ok = ts.notna().values
    if not ok.all():
        chunk = chunk[ok]
        is_tcp, is_udp, is_other = is_tcp[ok], is_udp[ok], is_other[ok]
        ts = ts[ok]
        if chunk.empty:
            return chunk

    def num(col, default=0.0):
        return pd.to_numeric(chunk[col], errors="coerce").fillna(default).values

    out = pd.DataFrame(index=range(len(chunk)))
    out["ts"] = ts.values
    out["ts_us"] = (ts.values * 1_000_000).round().astype(np.int64)
    out["src_ip"] = chunk["src_ip"].values
    out["dst_ip"] = chunk["dst_ip"].values
    out["frame_len"] = num("frame.len")
    out["ttl"] = num("ip.ttl")
    out["frag_offset"] = num("ip.frag_offset")
    out["is_tcp"] = is_tcp
    out["is_other"] = is_other

    tsp, tdp = num("tcp.srcport"), num("tcp.dstport")
    usp, udp_ = num("udp.srcport"), num("udp.dstport")
    out["sport"] = np.select([is_tcp, is_udp], [tsp, usp], 0).astype(np.int32)
    out["dport"] = np.select([is_tcp, is_udp], [tdp, udp_], 0).astype(np.int32)

    # protocol: 6 / 17 / 0. Never the real IP protocol number for the else
    # branch - Java simply leaves the field at its default.
    out["proto"] = np.select([is_tcp, is_udp], [6, 17], 0).astype(np.int32)

    tcp_len = num("tcp.len", 0.0)
    tcp_hdr = num("tcp.hdr_len", 20.0)
    udp_len = num("udp.length", float(UDP_HEADER_LEN))
    out["payload_len"] = np.select(
        [is_tcp, is_udp],
        [tcp_len, np.clip(udp_len - UDP_HEADER_LEN, 0, None)], 0.0)
    out["header_len"] = np.select([is_tcp, is_udp],
                                  [tcp_hdr, float(UDP_HEADER_LEN)], 0.0)
    out["tcp_window"] = np.where(is_tcp, num("tcp.window_size_value"),
                                 INIT_WIN_UNSET)

    fl = np.zeros(len(chunk), dtype=np.int64)
    for i, v in enumerate(chunk["tcp.flags"].values):
        if isinstance(v, str) and v:
            try:
                fl[i] = int(v, 16) if v.startswith("0x") else int(v)
            except ValueError:
                fl[i] = 0
    for name, bit in FLAG_BITS:
        out[name] = np.where(is_tcp, (fl & bit) != 0, False)

    out["more_frag"] = (chunk["ip.flags.mf"] == "1").values
    out["dont_frag"] = (chunk["ip.flags.df"] == "1").values
    out["is_retx"] = (chunk["tcp.analysis.retransmission"] == "1").values
    return out


def make_java_flow_id(src_ip, dst_ip, src_port, dst_port, proto):
    """
    Reproduces Java CICFlowMeter's generateFlowId():
    Lexicographically compares the 4 IPv4 octets as Java SIGNED bytes (-128..127).
    If equal, compares port numbers.
    """
    s_bytes = [int(x) if int(x) < 128 else int(x) - 256 for x in src_ip.split(".")]
    d_bytes = [int(x) if int(x) < 128 else int(x) - 256 for x in dst_ip.split(".")]
    if s_bytes < d_bytes:
        return f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{proto}"
    elif s_bytes > d_bytes:
        return f"{dst_ip}-{src_ip}-{dst_port}-{src_port}-{proto}"
    else:
        if int(src_port) <= int(dst_port):
            return f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{proto}"
        else:
            return f"{dst_ip}-{src_ip}-{dst_port}-{src_port}-{proto}"


# =============================================================================
# BasicFlow.java
# =============================================================================
class BasicFlow:
    __slots__ = (
        "src_ip", "dst_ip", "src_port", "dst_port", "proto", "flow_id",
        "flow_start_time", "flow_last_seen", "forward_last_seen",
        "backward_last_seen", "start_active_time", "end_active_time",
        "fwd_pkt_stats", "bwd_pkt_stats", "flow_len_stats",
        "flow_iat", "forward_iat", "backward_iat", "flow_active", "flow_idle",
        "forward_bytes", "backward_bytes", "f_header_bytes", "b_header_bytes",
        "n_fwd", "n_bwd", "f_psh", "b_psh", "f_urg", "b_urg", "f_fin", "b_fin",
        "flag_fin", "flag_syn", "flag_rst", "flag_psh",
        "flag_ack", "flag_urg", "flag_cwr", "flag_ece",
        "init_win_fwd", "init_win_bwd", "min_seg_size_fwd", "act_data_pkt_fwd",
        "sf_last_ts", "sf_count", "sf_ac_helper",
        "fb_duration", "fb_pkt_count", "fb_size_total", "fb_state_count",
        "fb_pkt_helper", "fb_start_helper", "fb_size_helper", "f_last_bulk_ts",
        "bb_duration", "bb_pkt_count", "bb_size_total", "bb_state_count",
        "bb_pkt_helper", "bb_start_helper", "bb_size_helper", "b_last_bulk_ts",
        "m_ttl", "m_win", "m_payload", "m_framelen",
        "any_mf", "any_df", "max_frag_off", "n_retx", "n_syn_pkts", "n_other",
    )

    def __init__(self, packet, flow_src=None, flow_dst=None,
                 flow_src_port=None, flow_dst_port=None):
        """
        The 7-arg timeout-restart constructor is:

            this.initParameters();      // src = null
            this.firstPacket(packet);   // <-- runs while src is STILL null
            this.src = flowSrc;         // forced orientation applied AFTER
            ...

        firstPacket() itself does `if (this.src == null) this.src =
        packet.getSrc();`, so its direction test is trivially true. The first
        packet of a timeout continuation is therefore always booked as forward,
        even when it is backward under the retained orientation. The backward
        branch of firstPacket is dead code.                             [SRC]
        """
        self._init_parameters()
        self._first_packet(packet)
        if flow_src is not None:
            self.src_ip, self.dst_ip = flow_src, flow_dst
            self.src_port, self.dst_port = flow_src_port, flow_dst_port

        # The released CIC-IDS2017 ISCX build uses generateFlowId() from
        # BasicPacketInfo.java which lexicographically compares IPv4 octets
        # as Java SIGNED bytes (-128..127). If equal, it compares ports.
        self.flow_id = make_java_flow_id(self.src_ip, self.dst_ip,
                                          self.src_port, self.dst_port, self.proto)

    def _init_parameters(self):
        self.src_ip = self.dst_ip = None
        self.src_port = self.dst_port = None
        self.proto = 0
        self.flow_id = None

        self.fwd_pkt_stats = SummaryStatistics()
        self.bwd_pkt_stats = SummaryStatistics()
        self.flow_len_stats = SummaryStatistics()
        self.flow_iat = SummaryStatistics()
        self.forward_iat = SummaryStatistics()
        self.backward_iat = SummaryStatistics()
        self.flow_active = SummaryStatistics()
        self.flow_idle = SummaryStatistics()

        self.forward_bytes = self.backward_bytes = 0
        self.f_header_bytes = self.b_header_bytes = 0
        self.n_fwd = self.n_bwd = 0
        self.f_psh = self.b_psh = self.f_urg = self.b_urg = 0
        self.f_fin = self.b_fin = 0
        self.flag_fin = self.flag_syn = self.flag_rst = self.flag_psh = 0
        self.flag_ack = self.flag_urg = self.flag_cwr = self.flag_ece = 0

        self.init_win_fwd = INIT_WIN_UNSET
        self.init_win_bwd = INIT_WIN_UNSET
        self.min_seg_size_fwd = 0
        self.act_data_pkt_fwd = 0

        self.sf_last_ts = -1
        self.sf_count = 0
        self.sf_ac_helper = -1

        self.fb_duration = self.fb_pkt_count = self.fb_size_total = 0
        self.fb_state_count = self.fb_pkt_helper = 0
        self.fb_start_helper = self.fb_size_helper = self.f_last_bulk_ts = 0
        self.bb_duration = self.bb_pkt_count = self.bb_size_total = 0
        self.bb_state_count = self.bb_pkt_helper = 0
        self.bb_start_helper = self.bb_size_helper = self.b_last_bulk_ts = 0

        self.flow_start_time = self.flow_last_seen = 0
        self.forward_last_seen = self.backward_last_seen = 0
        self.start_active_time = self.end_active_time = 0

        self.m_ttl = RunningMoments()
        self.m_win = RunningMoments()
        self.m_payload = RunningMoments()
        self.m_framelen = RunningMoments()
        self.any_mf = self.any_df = False
        self.max_frag_off = 0
        self.n_retx = self.n_syn_pkts = self.n_other = 0

    # ---------------------------------------------------------------- #
    def _is_forward(self, p):
        """Arrays.equals(this.src, packet.getSrc()) - source IP only, no port."""
        return p.src_ip == self.src_ip

    def _check_flags(self, p):
        if p.fin: self.flag_fin += 1
        if p.syn: self.flag_syn += 1
        if p.rst: self.flag_rst += 1
        if p.psh: self.flag_psh += 1
        if p.ack: self.flag_ack += 1
        if p.urg: self.flag_urg += 1
        if p.cwr: self.flag_cwr += 1
        if p.ece: self.flag_ece += 1

    def update_active_idle_time(self, t):
        if (t - self.end_active_time) > ACTIVITY_TIMEOUT_US:
            if (self.end_active_time - self.start_active_time) > 0:
                self.flow_active.add(self.end_active_time - self.start_active_time)
            self.flow_idle.add(t - self.end_active_time)
            self.start_active_time = t
            self.end_active_time = t
        else:
            self.end_active_time = t

    def _detect_update_subflows(self, ts):
        """
        Current master:
            if (sfLastPacketTS == -1) { sfLastPacketTS = ts; sfAcHelper = ts; }
            if (((ts - sfLastPacketTS)/(double)1000000) > 1.0) { sfCount++; ... }
            sfLastPacketTS = ts;

        ISCX build: the paper reports a misplaced parenthesis that made the
        comparison always true, so sfCount rose on every packet. Reconciling
        that against a released row (2 fwd packets, Subflow Fwd Packets == 2,
        therefore sfCount == 1) pins it to "every packet after the first". [INF]
        """
        first = self.sf_last_ts == -1
        if first:
            self.sf_last_ts = ts
            self.sf_ac_helper = ts
            if CFG["subflow_mode"] == "every_packet_after_first":
                return
        if CFG["subflow_mode"] == "gap":
            fire = ((ts - self.sf_last_ts) / 1_000_000.0) > SUBFLOW_GAP_S
        else:
            fire = True
        if fire:
            self.sf_count += 1
            self.update_active_idle_time(ts)
            self.sf_ac_helper = ts
        self.sf_last_ts = ts

    def _update_backward_bulk(self, ts, size, ts_other):
        if ts_other > self.bb_start_helper:
            self.bb_start_helper = 0
        if size <= 0:
            return
        if self.bb_start_helper == 0:
            self.bb_start_helper = ts
            self.bb_pkt_helper = 1
            self.bb_size_helper = size
            self.b_last_bulk_ts = ts
        elif ((ts - self.b_last_bulk_ts) / 1_000_000.0) > BULK_GAP_S:
            self.bb_start_helper = self.b_last_bulk_ts = ts
            self.bb_pkt_helper = 1
            self.bb_size_helper = size
        else:
            self.bb_pkt_helper += 1
            self.bb_size_helper += size
            if self.bb_pkt_helper == BULK_MIN_PACKETS:
                self.bb_state_count += 1
                self.bb_pkt_count += self.bb_pkt_helper
                self.bb_size_total += self.bb_size_helper
                self.bb_duration += ts - self.bb_start_helper
            elif self.bb_pkt_helper > BULK_MIN_PACKETS:
                self.bb_pkt_count += 1
                self.bb_size_total += size
                self.bb_duration += ts - self.b_last_bulk_ts
            self.b_last_bulk_ts = ts

    def _update_forward_bulk(self, ts, size, ts_other):
        if ts_other > self.fb_start_helper:
            self.fb_start_helper = 0
        if size <= 0:
            return
        if self.fb_start_helper == 0:
            self.fb_start_helper = ts
            self.fb_pkt_helper = 1
            self.fb_size_helper = size
            self.f_last_bulk_ts = ts
        elif ((ts - self.f_last_bulk_ts) / 1_000_000.0) > BULK_GAP_S:
            self.fb_start_helper = self.f_last_bulk_ts = ts
            self.fb_pkt_helper = 1
            self.fb_size_helper = size
        else:
            self.fb_pkt_helper += 1
            self.fb_size_helper += size
            if self.fb_pkt_helper == BULK_MIN_PACKETS:
                self.fb_state_count += 1
                self.fb_pkt_count += self.fb_pkt_helper
                self.fb_size_total += self.fb_size_helper
                self.fb_duration += ts - self.fb_start_helper
            elif self.fb_pkt_helper > BULK_MIN_PACKETS:
                self.fb_pkt_count += 1
                self.fb_size_total += size
                self.fb_duration += ts - self.f_last_bulk_ts
            self.f_last_bulk_ts = ts

    def _update_flow_bulk(self, p):
        """
        updateFlowBulk(): `if (this.src == packet.getSrc())` is Java REFERENCE
        equality on byte[], never true, so every packet goes to
        updateBackwardBulk.                                              [SRC]

        In the released CSVs all six bulk columns are constant zero, so the
        CICIDS2017 profile skips accumulation entirely.                  [PUB]
        """
        mode = CFG["bulk_mode"]
        if mode == "zero":
            return
        if mode == "backward_only":
            self._update_backward_bulk(p.ts_us, p.payload_len, self.f_last_bulk_ts)
            return
        if self._is_forward(p):
            self._update_forward_bulk(p.ts_us, p.payload_len, self.b_last_bulk_ts)
        else:
            self._update_backward_bulk(p.ts_us, p.payload_len, self.f_last_bulk_ts)

    def _record_extras(self, p):
        self.m_ttl.add(p.ttl)
        self.m_payload.add(p.payload_len)
        self.m_framelen.add(p.frame_len)
        if p.is_tcp:
            self.m_win.add(p.tcp_window)
            if p.is_retx:
                self.n_retx += 1
            if p.syn:
                self.n_syn_pkts += 1
        if p.is_other:
            self.n_other += 1
        if p.more_frag:
            self.any_mf = True
        if p.dont_frag:
            self.any_df = True
        if p.frag_offset > self.max_frag_off:
            self.max_frag_off = p.frag_offset

    # ---------------------------------------------------------------- #
    def _first_packet(self, p):
        self._update_flow_bulk(p)
        self._detect_update_subflows(p.ts_us)
        self._check_flags(p)

        self.flow_start_time = self.flow_last_seen = p.ts_us
        self.start_active_time = self.end_active_time = p.ts_us

        # Unconditional add. flowLengthStats therefore holds N+1 samples for N
        # packets, which is why Packet Length Mean != Average Packet Size.
        # Verified against a released row: payloads 6 and 6 give
        # Packet Length Mean 6 (18/3) and Average Packet Size 9 (18/2).
        #                                                            [SRC][CSV]
        self.flow_len_stats.add(p.payload_len)

        if self.src_ip is None:
            self.src_ip, self.src_port = p.src_ip, p.sport
        if self.dst_ip is None:
            self.dst_ip, self.dst_port = p.dst_ip, p.dport

        if self._is_forward(p):
            self.min_seg_size_fwd = p.header_len
            self.init_win_fwd = p.tcp_window
            self.flow_len_stats.add(p.payload_len)      # second add
            self.fwd_pkt_stats.add(p.payload_len)
            self.f_header_bytes = p.header_len
            self.forward_last_seen = p.ts_us
            self.forward_bytes += p.payload_len
            self.n_fwd += 1
            if p.psh: self.f_psh += 1
            if p.urg: self.f_urg += 1
        else:
            self.init_win_bwd = p.tcp_window
            self.flow_len_stats.add(p.payload_len)
            self.bwd_pkt_stats.add(p.payload_len)
            self.b_header_bytes = p.header_len
            self.backward_last_seen = p.ts_us
            self.backward_bytes += p.payload_len
            self.n_bwd += 1
            if p.psh: self.b_psh += 1
            if p.urg: self.b_urg += 1

        self.proto = p.proto
        self._record_extras(p)

    def add_packet(self, p):
        """
        addPacket() does NOT touch fFIN_cnt / bFIN_cnt - only setFwdFINFlags()
        and setBwdFINFlags() in FlowGenerator do. It also does not call
        updateActiveIdleTime; FlowGenerator calls that immediately before, and
        not at all on the closing paths.                                 [SRC]
        """
        self._update_flow_bulk(p)
        self._detect_update_subflows(p.ts_us)
        self._check_flags(p)

        ts = p.ts_us
        self.flow_len_stats.add(p.payload_len)

        if self._is_forward(p):
            if p.payload_len >= 1:
                self.act_data_pkt_fwd += 1
            self.fwd_pkt_stats.add(p.payload_len)
            self.f_header_bytes += p.header_len
            self.forward_bytes += p.payload_len
            self.n_fwd += 1
            if self.n_fwd > 1:
                self.forward_iat.add(ts - self.forward_last_seen)
            self.forward_last_seen = ts
            self.min_seg_size_fwd = min(p.header_len, self.min_seg_size_fwd)
            if p.psh: self.f_psh += 1
            if p.urg: self.f_urg += 1
        else:
            self.bwd_pkt_stats.add(p.payload_len)
            # overwritten by every backward packet, so the emitted value is the
            # LAST backward window, not the initial one.             [SRC][PUB]
            self.init_win_bwd = p.tcp_window
            self.b_header_bytes += p.header_len
            self.backward_bytes += p.payload_len
            self.n_bwd += 1
            if self.n_bwd > 1:
                self.backward_iat.add(ts - self.backward_last_seen)
            self.backward_last_seen = ts
            if p.psh: self.b_psh += 1
            if p.urg: self.b_urg += 1

        self.flow_iat.add(ts - self.flow_last_seen)
        self.flow_last_seen = ts
        self._record_extras(p)

    def packet_count(self):
        return self.n_fwd + self.n_bwd

    # ---------------------------------------------------------------- #
    def dump(self):
        dur_us = self.flow_last_seen - self.flow_start_time
        dur_s = dur_us / 1_000_000.0
        n = self.packet_count()

        if (self.end_active_time - self.start_active_time) > 0:
            self.flow_active.add(self.end_active_time - self.start_active_time)

        tot_bytes = self.forward_bytes + self.backward_bytes
        if CFG["infinity_rates"] and dur_s == 0.0:
            flow_bytes_s = float("nan") if tot_bytes == 0 else float("inf")
            flow_pkts_s = float("nan") if n == 0 else float("inf")
        else:
            d = dur_s if dur_s > 0 else 1e-6
            flow_bytes_s = tot_bytes / d
            flow_pkts_s = n / d
        fwd_pkts_s = (self.n_fwd / dur_s) if dur_us > 0 else 0.0
        bwd_pkts_s = (self.n_bwd / dur_s) if dur_us > 0 else 0.0

        if self.n_fwd > 0:
            down_up = (self.n_bwd // self.n_fwd) if CFG["downup_int_div"] \
                else (self.n_bwd / self.n_fwd)
        else:
            down_up = 0

        avg_pkt_size = (self.flow_len_stats.sum() / n) if n else 0.0
        avg_fwd_seg = (self.fwd_pkt_stats.sum() / self.n_fwd) if self.n_fwd else 0.0
        avg_bwd_seg = (self.bwd_pkt_stats.sum() / self.n_bwd) if self.n_bwd else 0.0

        if CFG["bulk_mode"] == "zero":
            fab = fap = far = bab = bap = bar = 0
        else:
            def idiv(a, b):
                return (a // b) if b else 0
            fab = idiv(self.fb_size_total, self.fb_state_count)
            fap = idiv(self.fb_pkt_count, self.fb_state_count)
            bab = idiv(self.bb_size_total, self.bb_state_count)
            bap = idiv(self.bb_pkt_count, self.bb_state_count)
            far = int(self.fb_size_total / (self.fb_duration / 1e6)) if self.fb_duration else 0
            bar = int(self.bb_size_total / (self.bb_duration / 1e6)) if self.bb_duration else 0

        sf = self.sf_count
        sf_fp = (self.n_fwd // sf) if sf > 0 else 0
        sf_fb = (self.forward_bytes // sf) if sf > 0 else 0
        sf_bp = (self.n_bwd // sf) if sf > 0 else 0
        sf_bb = (self.backward_bytes // sf) if sf > 0 else 0

        # --- flag columns ------------------------------------------------
        # dumpFlowBasedFeatures() writes `for (String key : flagCounts.keySet())`
        # over a java.util.HashMap. With those 8 keys in a 16-bucket Java 8
        # table the iteration order is RST, PSH, ECE, SYN, ACK, FIN, URG, CWR,
        # which scrambles 6 of the 8 columns.                            [SRC]
        # Corroboration: the paper reports SYN<->PSH and FIN<->URG inverted,
        # agreeing on SYN, PSH, URG and ACK. It disagrees on FIN, but the same
        # paper confirms URG is never set in any packet in the PCAPs, so a
        # literal FIN<->URG swap would make FIN Flag Count identically zero
        # dataset-wide - and it is not. RST count fits.             [PUB][INF]
        # "pairswap" gives the paper's literal reading if you prefer it.
        real = {"FIN": self.flag_fin, "SYN": self.flag_syn, "RST": self.flag_rst,
                "PSH": self.flag_psh, "ACK": self.flag_ack, "URG": self.flag_urg,
                "CWR": self.flag_cwr, "ECE": self.flag_ece}
        orders = {
            "hashmap":   ["RST", "PSH", "ECE", "SYN", "ACK", "FIN", "URG", "CWR"],
            "pairswap":  ["URG", "PSH", "RST", "SYN", "ACK", "FIN", "CWR", "ECE"],
            "canonical": ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWR", "ECE"],
        }
        hdr = ["FIN Flag Count", "SYN Flag Count", "RST Flag Count",
               "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
               "CWE Flag Count", "ECE Flag Count"]
        flag_cols = {h: real[k]
                     for h, k in zip(hdr, orders[CFG["flag_column_order"]])}

        b_psh = 0 if CFG["force_zero_bwd_psh_urg"] else self.b_psh
        b_urg = 0 if CFG["force_zero_bwd_psh_urg"] else self.b_urg

        ts_s = self.flow_start_time / 1_000_000.0
        row = {
            "Flow ID": self.flow_id,
            "Source IP": self.src_ip,
            "Source Port": self.src_port,
            "Destination IP": self.dst_ip,
            "Destination Port": self.dst_port,
            "Protocol": self.proto,
            "Timestamp": datetime.fromtimestamp(ts_s, tz=DATASET_TZ).strftime(CFG["timestamp_fmt"]),
            "Flow Duration": dur_us,
            "Total Fwd Packets": self.n_fwd,
            "Total Backward Packets": self.n_bwd,
            "Total Length of Fwd Packets": self.fwd_pkt_stats.sum(),
            "Total Length of Bwd Packets": self.bwd_pkt_stats.sum(),
            "Fwd Packet Length Max": self.fwd_pkt_stats.max() if self.n_fwd else 0.0,
            "Fwd Packet Length Min": self.fwd_pkt_stats.min() if self.n_fwd else 0.0,
            "Fwd Packet Length Mean": self.fwd_pkt_stats.mean() if self.n_fwd else 0.0,
            "Fwd Packet Length Std": self.fwd_pkt_stats.std() if self.n_fwd else 0.0,
            "Bwd Packet Length Max": self.bwd_pkt_stats.max() if self.n_bwd else 0.0,
            "Bwd Packet Length Min": self.bwd_pkt_stats.min() if self.n_bwd else 0.0,
            "Bwd Packet Length Mean": self.bwd_pkt_stats.mean() if self.n_bwd else 0.0,
            "Bwd Packet Length Std": self.bwd_pkt_stats.std() if self.n_bwd else 0.0,
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_pkts_s,
            "Flow IAT Mean": self.flow_iat.mean(),
            "Flow IAT Std": self.flow_iat.std(),
            "Flow IAT Max": self.flow_iat.max(),
            "Flow IAT Min": self.flow_iat.min(),
            "Fwd IAT Total": self.forward_iat.sum() if self.n_fwd > 1 else 0.0,
            "Fwd IAT Mean": self.forward_iat.mean() if self.n_fwd > 1 else 0.0,
            "Fwd IAT Std": self.forward_iat.std() if self.n_fwd > 1 else 0.0,
            "Fwd IAT Max": self.forward_iat.max() if self.n_fwd > 1 else 0.0,
            "Fwd IAT Min": self.forward_iat.min() if self.n_fwd > 1 else 0.0,
            "Bwd IAT Total": self.backward_iat.sum() if self.n_bwd > 1 else 0.0,
            "Bwd IAT Mean": self.backward_iat.mean() if self.n_bwd > 1 else 0.0,
            "Bwd IAT Std": self.backward_iat.std() if self.n_bwd > 1 else 0.0,
            "Bwd IAT Max": self.backward_iat.max() if self.n_bwd > 1 else 0.0,
            "Bwd IAT Min": self.backward_iat.min() if self.n_bwd > 1 else 0.0,
            "Fwd PSH Flags": self.f_psh,
            "Bwd PSH Flags": b_psh,
            "Fwd URG Flags": self.f_urg,
            "Bwd URG Flags": b_urg,
            "Fwd Header Length": self.f_header_bytes,
            "Bwd Header Length": self.b_header_bytes,
            "Fwd Packets/s": fwd_pkts_s,
            "Bwd Packets/s": bwd_pkts_s,
            "Min Packet Length": self.flow_len_stats.min(),
            "Max Packet Length": self.flow_len_stats.max(),
            "Packet Length Mean": self.flow_len_stats.mean(),
            "Packet Length Std": self.flow_len_stats.std(),
            "Packet Length Variance": self.flow_len_stats.variance(),
            **flag_cols,
            "Down/Up Ratio": down_up,
            "Average Packet Size": avg_pkt_size,
            "Avg Fwd Segment Size": avg_fwd_seg,
            "Avg Bwd Segment Size": avg_bwd_seg,
        }
        if CFG["emit_dup_fwd_header_len"]:
            row["Fwd Header Length.1"] = self.f_header_bytes
        row.update({
            "Fwd Avg Bytes/Bulk": fab,
            "Fwd Avg Packets/Bulk": fap,
            "Fwd Avg Bulk Rate": far,
            "Bwd Avg Bytes/Bulk": bab,
            "Bwd Avg Packets/Bulk": bap,
            "Bwd Avg Bulk Rate": bar,
            "Subflow Fwd Packets": sf_fp,
            "Subflow Fwd Bytes": sf_fb,
            "Subflow Bwd Packets": sf_bp,
            "Subflow Bwd Bytes": sf_bb,
            "Init_Win_bytes_forward": self.init_win_fwd,
            "Init_Win_bytes_backward": self.init_win_bwd,
            "act_data_pkt_fwd": self.act_data_pkt_fwd,
            "min_seg_size_forward": self.min_seg_size_fwd,
            "Active Mean": self.flow_active.mean(),
            "Active Std": self.flow_active.std(),
            "Active Max": self.flow_active.max(),
            "Active Min": self.flow_active.min(),
            "Idle Mean": self.flow_idle.mean(),
            "Idle Std": self.flow_idle.std(),
            "Idle Max": self.flow_idle.max(),
            "Idle Min": self.flow_idle.min(),

            # ---- packet-level extras, not CICFlowMeter --------------------
            "pkt_ttl_mean": self.m_ttl.mean,
            "pkt_ttl_std": self.m_ttl.std(),
            "pkt_ttl_min": self.m_ttl.min(),
            "pkt_ttl_max": self.m_ttl.max(),
            "pkt_win_mean": self.m_win.mean if self.m_win.n else 0.0,
            "pkt_win_std": self.m_win.std(),
            "pkt_win_min": self.m_win.min(),
            "pkt_win_max": self.m_win.max(),
            "pkt_payload_min": self.m_payload.min(),
            "pkt_payload_max": self.m_payload.max(),
            "pkt_payload_var": self.m_payload.variance(),
            "pkt_payload_skew": self.m_payload.skew(),
            "pkt_payload_kurtosis": self.m_payload.kurtosis(),
            "pkt_payload_nonzero_ratio": self.m_payload.nonzero_ratio(),
            "pkt_framelen_mean": self.m_framelen.mean,
            "pkt_framelen_std": self.m_framelen.std(),
            "frag_more_flag_present": self.any_mf,
            "frag_dont_flag_present": self.any_df,
            "frag_max_offset": self.max_frag_off,
            "retransmission_count": self.n_retx,
            "syn_packet_count": self.n_syn_pkts,
            "non_tcp_udp_packet_count": self.n_other,
            "subflow_count_raw": self.sf_count,
            "packet_count": n,
            "flow_start_epoch": ts_s,
        })
        if CFG["emit_true_flag_columns"]:
            row.update({f"flag_true_{k.lower()}": v for k, v in real.items()})
        row["timestamp_window"] = int(
            self.flow_start_time // (CFG["window_seconds"] * 1_000_000))
        return row


# =============================================================================
# FlowGenerator.java
# =============================================================================
class FlowGenerator:
    def __init__(self, on_flow):
        self.current = {}
        self.on_flow = on_flow

    def add_packet(self, p):
        key = (p.src_ip, p.sport, p.dst_ip, p.dport, p.proto)
        rev = (p.dst_ip, p.dport, p.src_ip, p.sport, p.proto)

        if key in self.current:
            fid = key
        elif rev in self.current:
            fid = rev
        else:
            self.current[key] = BasicFlow(p)
            return

        flow = self.current[fid]

        # --- 1. 120 s timeout, measured from flowStartTime ----------------
        if (p.ts_us - flow.flow_start_time) > FLOW_TIMEOUT_US:
            if flow.packet_count() > 1:
                self.on_flow(flow)
            del self.current[fid]
            self.current[fid] = BasicFlow(p, flow.src_ip, flow.dst_ip,
                                          flow.src_port, flow.dst_port)
            return

        # --- 2. FIN -------------------------------------------------------
        if p.fin:
            if CFG["termination"] == "fin_any":
                # The ISCX-era rule, which sits commented out in current
                # master:  flow.addPacket(packet); emit; remove(id);
                # Note there is no updateActiveIdleTime on this path.   [PUB]
                flow.add_packet(p)
                self._close(fid)
            else:
                # Current master. Both branches test bwd+bwd, a copy-paste
                # slip, so the condition reduces to bwdFINFlags == 1: a forward
                # FIN never closes on its own, the first backward FIN always
                # does, and a second FIN in the same direction is dropped. [SRC]
                if flow._is_forward(p):
                    flow.f_fin += 1
                    if flow.f_fin == 1:
                        if (flow.b_fin + flow.b_fin) == 2:
                            flow.add_packet(p)
                            self._close(fid)
                        else:
                            flow.update_active_idle_time(p.ts_us)
                            flow.add_packet(p)
                else:
                    flow.b_fin += 1
                    if flow.b_fin == 1:
                        if (flow.b_fin + flow.b_fin) == 2:
                            flow.add_packet(p)
                            self._close(fid)
                        else:
                            flow.update_active_idle_time(p.ts_us)
                            flow.add_packet(p)
            return

        # --- 3. RST -------------------------------------------------------
        if p.rst and CFG["rst_closes"]:
            flow.add_packet(p)          # no updateActiveIdleTime here
            self._close(fid)
            return

        # --- 4. normal ----------------------------------------------------
        # Current master:
        #   if (fwdMatch && fwdFIN == 0)  { add }
        #   else if (bwdFIN == 0)         { add }
        #   else                          { drop }
        # The `else if` carries NO direction test, so a forward packet arriving
        # after a forward FIN falls through and IS added.                [SRC]
        if not CFG["half_close_guard"]:
            flow.update_active_idle_time(p.ts_us)
            flow.add_packet(p)
        elif flow._is_forward(p) and flow.f_fin == 0:
            flow.update_active_idle_time(p.ts_us)
            flow.add_packet(p)
        elif flow.b_fin == 0:
            flow.update_active_idle_time(p.ts_us)
            flow.add_packet(p)

    def _close(self, fid):
        flow = self.current.pop(fid)
        if flow.packet_count() > 1:
            self.on_flow(flow)

    def drain(self):
        for fid in list(self.current.keys()):
            flow = self.current.pop(fid)
            if flow.packet_count() > 1:
                self.on_flow(flow)


# =============================================================================
# Driver
# =============================================================================
def process_day(pcap_path, out_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for k, v in COMMON_CFG.items():
        CFG.setdefault(k, v)

    day = os.path.splitext(os.path.basename(pcap_path))[0]
    win_s = CFG.get("window_seconds", 5)
    completed, flush_paths, st = [], [], {"idx": 0, "rows": 0}

    def flush():
        if not completed:
            return
        tmp = os.path.join(LOCAL_TMP_DIR, f"partial_{day}_{st['idx']}.parquet")
        pd.DataFrame(completed).to_parquet(tmp, index=False)
        flush_paths.append(tmp)
        st["idx"] += 1
        completed.clear()
        gc.collect()

    def on_flow(flow):
        completed.append(flow.dump())
        st["rows"] += 1
        if len(completed) >= CFG["completed_flush_every"]:
            flush()

    gen = FlowGenerator(on_flow)
    dport_counts = defaultdict(Counter)
    scan = defaultdict(lambda: {"last": None, "run": 0, "max_run": 0,
                                "seq": 0, "trans": 0})
    i = 0
    for chunk in run_tshark_to_df_iter(pcap_path, CFG["tshark_chunk_size"]):
        chunk = clean_chunk(chunk)
        if chunk.empty:
            continue
        for p in chunk.itertuples(index=False):
            gen.add_packet(p)

            k = (p.src_ip, int(p.ts // win_s))
            dport_counts[k][p.dport] += 1
            s = scan[k]
            if s["last"] is not None:
                s["trans"] += 1
                if abs(p.dport - s["last"]) == 1:
                    s["run"] += 1
                    s["seq"] += 1
                    if s["run"] > s["max_run"]:
                        s["max_run"] = s["run"]
                else:
                    s["run"] = 0
            s["last"] = p.dport

            i += 1
            if CFG["enable_idle_eviction"] and i % CFG["expiry_check_every"] == 0:
                cut = p.ts_us - int(CFG["idle_evict_seconds"] * 1e6)
                for k2 in [k2 for k2, f in gen.current.items()
                           if f.flow_last_seen < cut]:
                    gen._close(k2)
            if CFG["max_active_flows"] and len(gen.current) > CFG["max_active_flows"]:
                old = sorted(gen.current.items(),
                             key=lambda kv: kv[1].flow_start_time)
                for k2, _ in old[:len(gen.current) - CFG["max_active_flows"]]:
                    gen._close(k2)

    gen.drain()
    flush()

    rows = []
    for (ip, w), c in dport_counts.items():
        v = np.fromiter(c.values(), dtype=float)
        pr = v / v.sum()
        s = scan[(ip, w)]
        rows.append((ip, w, float(-(pr * np.log2(pr)).sum()), len(c),
                     s["max_run"], (s["seq"] / s["trans"]) if s["trans"] else 0.0))
    scan_df = pd.DataFrame(rows, columns=[
        "Source IP", "timestamp_window", "port_scan_entropy",
        "unique_dst_ports_per_src", "port_scan_max_sequential_run",
        "port_scan_sequential_ratio"])
    dport_counts.clear()
    scan.clear()
    gc.collect()

    total, writer = 0, None
    try:
        for path in flush_paths:
            part = pd.read_parquet(path)
            if not part.empty:
                part = part.merge(scan_df, on=["Source IP", "timestamp_window"],
                                  how="left")
                for c, f, dt in [("port_scan_entropy", 0.0, "float64"),
                                 ("unique_dst_ports_per_src", 0, "int64"),
                                 ("port_scan_max_sequential_run", 0, "int64"),
                                 ("port_scan_sequential_ratio", 0.0, "float64")]:
                    part[c] = part[c].fillna(f).astype(dt)
                part["source_file"] = day
                part["profile"] = PROFILE
                tbl = pa.Table.from_pandas(part, preserve_index=False)
                if writer is None:
                    target_schema = tbl.schema
                    writer = pq.ParquetWriter(out_path, target_schema)
                elif not tbl.schema.equals(target_schema, check_metadata=False):
                    tbl = tbl.cast(target_schema)
                writer.write_table(tbl)
                total += len(part)
            os.remove(path)
    finally:
        if writer is not None:
            writer.close()
    return total


def build_window_state_vectors(df):
    spec = {
        "Total Fwd Packets": "sum", "Total Backward Packets": "sum",
        "packet_count": "sum", "retransmission_count": "sum",
        "syn_packet_count": "sum", "non_tcp_udp_packet_count": "sum",
        "pkt_ttl_mean": ["mean", "std"], "pkt_ttl_std": "mean",
        "pkt_win_mean": ["mean", "std"], "pkt_payload_var": "mean",
        "pkt_payload_skew": "mean", "pkt_payload_kurtosis": "mean",
        "pkt_payload_nonzero_ratio": "mean",
        "frag_more_flag_present": "max", "frag_dont_flag_present": "max",
        "frag_max_offset": "max", "port_scan_entropy": "max",
        "unique_dst_ports_per_src": "max",
        "port_scan_max_sequential_run": "max",
        "port_scan_sequential_ratio": "max", "Source IP": "count",
    }
    spec = {k: v for k, v in spec.items() if k in df.columns}
    out = df.groupby(["source_file", "timestamp_window"]).agg(spec)
    out.columns = ["_".join(c) if isinstance(c, tuple) else c for c in out.columns]
    return out.rename(columns={"Source IP_count": "flows_in_window"}).reset_index()


# =============================================================================
from collections import namedtuple

PacketTuple = namedtuple(
    "PacketTuple",
    "ts ts_us src_ip dst_ip sport dport proto is_tcp is_other frame_len ttl "
    "payload_len header_len more_frag dont_frag frag_offset tcp_window is_retx "
    "fin syn rst psh ack urg ece cwr")


def self_test():
    """
    Regression against a real CIC-IDS2017 row (8.254.250.126:80 ->
    192.168.10.5:49188, two forward packets, 6 bytes payload each, 4 us apart).
    Released values: Flow Duration 4, Total Fwd Packets 2,
    Total Length of Fwd Packets 12, Packet Length Mean 6,
    Average Packet Size 9, Fwd Header Length 40, min_seg_size_forward 20,
    Subflow Fwd Packets 2, Subflow Fwd Bytes 12, Init_Win_bytes_forward 329,
    Init_Win_bytes_backward -1, act_data_pkt_fwd 1, Down/Up Ratio 0.
    """
    def mk(t, s, d, sp, dp, pl=0, hdr=20, win=329,
           fin=0, syn=0, rst=0, psh=0, ack=1):
        return PacketTuple(t, int(round(t * 1e6)), s, d, sp, dp, 6, True, False,
                           pl + hdr + 20, 128, pl, hdr, False, True, 0, win,
                           False, fin, syn, rst, psh, ack, 0, 0, 0)

    rows = []
    g = FlowGenerator(lambda f: rows.append(f.dump()))
    A, B = "8.254.250.126", "192.168.10.5"
    g.add_packet(mk(0.000000, A, B, 80, 49188, 6))
    g.add_packet(mk(0.000004, A, B, 80, 49188, 6))
    g.drain()

    exp = {"Flow Duration": 4, "Total Fwd Packets": 2,
           "Total Backward Packets": 0, "Total Length of Fwd Packets": 12.0,
           "Packet Length Mean": 6.0, "Average Packet Size": 9.0,
           "Fwd Header Length": 40, "Fwd Header Length.1": 40,
           "min_seg_size_forward": 20, "Subflow Fwd Packets": 2,
           "Subflow Fwd Bytes": 12, "Init_Win_bytes_forward": 329,
           "Init_Win_bytes_backward": -1, "act_data_pkt_fwd": 1,
           "Down/Up Ratio": 0, "Fwd Avg Bytes/Bulk": 0}
    if PROFILE != "CICIDS2017":
        print(f"self_test targets CICIDS2017; PROFILE is {PROFILE}. Skipping.")
        return True

    r = rows[0]
    bad = []
    for k, v in exp.items():
        got = r.get(k)
        mark = "ok " if got == v else "BAD"
        if got != v:
            bad.append((k, got, v))
        print(f"  [{mark}] {k:30s} got={got!r:>10}  want={v!r}")
    print("SELF TEST:", "PASS" if not bad else f"FAIL {bad}")
    return not bad


def main():
    import shutil, time
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
    os.makedirs(LOCAL_PCAP_TMP, exist_ok=True)

    files = [
        "Monday-WorkingHours.pcap",
        # "Tuesday-WorkingHours.pcap",
        # "Wednesday-workingHours.pcap",
        # "Thursday-WorkingHours.pcap",
        # "Friday-WorkingHours.pcap",
    ]
    for fname in files:
        day = os.path.splitext(fname)[0]
        out = os.path.join(OUTPUT_DIR, f"unified_{PROFILE}_{day}.parquet")
        if os.path.exists(out):
            print(f"skip {day}")
            continue
        local = os.path.join(LOCAL_PCAP_TMP, fname)
        print(f"copying {day} ...")
        shutil.copy(os.path.join(PCAP_DIR, fname), local)
        try:
            t0 = time.time()
            n = process_day(local, out)
            print(f"  {n} flows -> {out}  ({time.time() - t0:.0f}s)")
        finally:
            if os.path.exists(local):
                os.remove(local)
    print("done")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    main()