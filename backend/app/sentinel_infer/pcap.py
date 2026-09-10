"""Offline PCAP -> flow rows.

Reads a capture with scapy (no Npcap / live-capture privileges needed for a
file) and reassembles packets into the same CICFlowMeter-schema rows an uploaded
CSV would carry, via the vendored FlowMeter. The result feeds straight into
``InferenceEngine.forecast``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Union

import pandas as pd

from .flowmeter import FlowMeter


class BadPcap(ValueError):
    """422 — the upload is not a readable pcap/pcapng, or has no usable packets."""


def pcap_to_flows(data: Union[bytes, str, Path]) -> pd.DataFrame:
    """bytes (an upload) or a path -> DataFrame of flow rows, time-sorted."""
    try:
        from scapy.utils import PcapReader
    except Exception as e:                                   # pragma: no cover
        raise BadPcap(f"scapy is required for PCAP ingestion: {e}")

    tmp: Path | None = None
    if isinstance(data, (bytes, bytearray)):
        if len(data) < 24:
            raise BadPcap("file too small to be a pcap")
        fh = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
        fh.write(data)
        fh.close()
        path = Path(fh.name)
        tmp = path
    else:
        path = Path(data)
        if not path.is_file():
            raise BadPcap(f"no such file: {path}")

    meter = FlowMeter(idle_timeout=1e9, active_timeout=1e9)  # never time-evict offline
    n_pkts = 0
    try:
        with PcapReader(str(path)) as rd:
            for pkt in rd:
                ts = float(getattr(pkt, "time", 0.0)) or None
                meter.add_packet(pkt, ts)
                n_pkts += 1
    except Exception as e:
        raise BadPcap(f"could not parse the capture: {e}")
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass

    if n_pkts == 0:
        raise BadPcap("the capture contains no packets")

    rows = meter.flush_all()
    if not rows:
        raise BadPcap("no TCP/UDP flows could be assembled from the capture")
    df = pd.DataFrame.from_records(rows).sort_values("flow_start_epoch")
    return df.reset_index(drop=True)
