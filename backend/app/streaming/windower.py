"""Online 10-second windowing for the telemetry WebSocket.

The research package has no streaming path (`state_windows._assign_windows` is
batch-only). This keeps a bounded rolling buffer of raw flow records and, each
time a window closes past a grace watermark, re-aggregates the trailing
`L + 2` windows via `state_windows.build_state_windows` and scores the newest
anchor with `forward_sim.simulate_anchor` - identical maths to a CSV upload,
just incremental.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from app.inference.loader import get_engine
from app.inference.pipeline import forecast
from app.sentinel_infer.forecast import BadUpload, normalise_upload


class StreamingWindower:
    def __init__(self, session_id: str, family_hint: Optional[str] = None,
                 explain: bool = True):
        self.session_id = session_id
        self.family_hint = family_hint
        self.explain = explain
        eng = get_engine()
        self.W = eng.window_seconds
        self.L = eng.L
        self.grace = None                      # set from settings lazily
        self._buf: list[dict] = []
        self._t0: Optional[float] = None       # epoch of the very first flow
        self._emitted_through = -1             # highest global window index scored
        self._max_epoch = 0.0
        self._total_in = 0

    # -- ingest ----------------------------------------------------------
    def add_flows(self, records: list[dict]) -> int:
        if not records:
            return 0
        df = pd.DataFrame.from_records(records)
        df = normalise_upload(df)              # raises BadUpload on bad schema
        rows = df.to_dict("records")
        for r in rows:
            e = float(r["flow_start_epoch"])
            if self._t0 is None or e < self._t0:
                self._t0 = e
            self._max_epoch = max(self._max_epoch, e)
        self._buf.extend(rows)
        self._total_in += len(rows)
        return len(rows)

    # -- score any windows that have now closed -------------------------
    def poll_ready(self) -> list[dict]:
        from app.settings import settings
        if self.grace is None:
            self.grace = settings.stream_grace_seconds
        if self._t0 is None or not self._buf:
            return []
        watermark = self._max_epoch - self.grace
        last_closed = int(math.floor((watermark - self._t0) / self.W))
        out = []
        for wi in range(max(self._emitted_through + 1, self.L - 1), last_closed + 1):
            fc = self._score_window(wi)
            self._emitted_through = wi
            if fc is not None:
                out.append(fc)
        self._evict(last_closed)
        return out

    def flush(self) -> list[dict]:
        """score whatever remains (call on `close`)."""
        if self._t0 is None or not self._buf:
            return []
        last = int(math.floor((self._max_epoch - self._t0) / self.W))
        out = []
        for wi in range(max(self._emitted_through + 1, self.L - 1), last + 1):
            fc = self._score_window(wi)
            self._emitted_through = wi
            if fc is not None:
                out.append(fc)
        return out

    # -- internals -----------------------------------------------------
    def _win_of(self, epoch: float) -> int:
        return int(math.floor((epoch - self._t0) / self.W))

    def _score_window(self, wi: int) -> Optional[dict]:
        lo = wi - (self.L + 1)
        slice_rows = [r for r in self._buf
                      if lo <= self._win_of(float(r["flow_start_epoch"])) <= wi]
        if not slice_rows:
            return None
        df = pd.DataFrame.from_records(slice_rows)
        try:
            res = forecast(df, family_hint=self.family_hint, explain=self.explain)
        except BadUpload:
            return None
        if not res["anchors"]:
            return None
        anchor = res["anchors"][-1]            # newest = window `wi`
        anchor.setdefault("meta", {})
        anchor["meta"].update(session_id=self.session_id, stream_window=wi,
                              flows_scored=len(slice_rows))
        return anchor

    def _evict(self, last_closed: int) -> None:
        keep_from = last_closed - (self.L + 3)
        if keep_from <= 0:
            return
        self._buf = [r for r in self._buf
                     if self._win_of(float(r["flow_start_epoch"])) >= keep_from]

    @property
    def stats(self) -> dict:
        return {"session_id": self.session_id, "buffered": len(self._buf),
                "total_received": self._total_in,
                "emitted_through_window": self._emitted_through}
