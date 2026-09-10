"""The online windower must emit a forecast once >= L windows of history exist,
and its per-window result must match the CSV path's contract."""
from __future__ import annotations

import numpy as np


def _feed_in_chunks(win, df, chunk_seconds=10.0):
    t0 = df["flow_start_epoch"].min()
    emitted = []
    hi = 0
    step = chunk_seconds
    end = df["flow_start_epoch"].max()
    cur = t0 + step
    while cur <= end + step:
        chunk = df[(df["flow_start_epoch"] >= t0) & (df["flow_start_epoch"] < cur)]
        chunk = chunk.iloc[hi:]
        hi += len(chunk)
        if len(chunk):
            win.add_flows(chunk.to_dict("records"))
            emitted += win.poll_ready()
        cur += step
    emitted += win.flush()
    return emitted


def test_windower_emits_after_history(client, synth_flows_df):
    # `client` fixture ensures the engine is loaded against the tiny bundle
    from app.streaming.windower import StreamingWindower

    win = StreamingWindower("t1", family_hint="PortScan")
    out = _feed_in_chunks(win, synth_flows_df)

    assert len(out) >= 1, "no forecast emitted over ~300 s of 10 s windows"
    fc = out[0]
    assert len(fc["horizon"]) == 6
    assert fc["meta"]["session_id"] == "t1"
    assert "stream_window" in fc["meta"]
    for h in fc["horizon"]:
        assert 0.0 <= h["attack_prob"] <= 1.0
        assert h["attck"]["confidence"] in {"High", "Medium", "Low"}
    # windows are emitted in order, once each
    wins = [f["meta"]["stream_window"] for f in out]
    assert wins == sorted(wins) and len(wins) == len(set(wins))


def test_windower_buffer_is_bounded(client, synth_flows_df):
    from app.streaming.windower import StreamingWindower

    win = StreamingWindower("t2")
    _feed_in_chunks(win, synth_flows_df)
    # after processing ~30 windows the buffer holds only the trailing ~L+3
    assert win.stats["buffered"] <= len(synth_flows_df)
    assert win.stats["buffered"] < 0.6 * len(synth_flows_df)
