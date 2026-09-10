"""Manual WebSocket smoke test: replay a flow CSV in 10 s slices over WS /stream
and print the forecasts the server pushes back.

    # terminal 1
    cd backend && SENTINEL_WM_MODEL_DIR=../models uvicorn app.main:app
    # terminal 2
    python tests/ws_smoke.py path/to/flows.csv --hint PortScan
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--url", default="ws://localhost:8000/stream")
    ap.add_argument("--hint", default=None)
    ap.add_argument("--window", type=float, default=10.0)
    a = ap.parse_args()

    try:
        from websockets.sync.client import connect
    except Exception:
        print("pip install websockets", file=sys.stderr)
        return 2

    df = pd.read_csv(a.csv, low_memory=False).sort_values("flow_start_epoch")
    t0 = float(df["flow_start_epoch"].min())
    n_fc = 0

    with connect(a.url) as ws:
        print("<<", ws.recv())                          # ready
        ws.send(json.dumps({"type": "hello", "family_hint": a.hint, "explain": True}))
        cur, sent = t0 + a.window, 0
        end = float(df["flow_start_epoch"].max())
        while cur <= end + a.window:
            chunk = df[(df["flow_start_epoch"] >= t0) & (df["flow_start_epoch"] < cur)].iloc[sent:]
            sent += len(chunk)
            if len(chunk):
                ws.send(json.dumps({"type": "flows",
                                    "records": chunk.to_dict("records")}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg["type"] == "ack":
                        break
                    if msg["type"] == "forecast":
                        n_fc += 1
                        peak = max(msg["horizon"], key=lambda h: h["attack_prob"])
                        print(f"  window {msg['meta'].get('stream_window'):>3} | "
                              f"peak P={peak['attack_prob']:.2f} @+{peak['horizon_seconds']}s | "
                              f"{peak['attck']['kill_chain_phase']} ({peak['attck']['confidence']})")
                    elif msg["type"] == "error":
                        print("ERROR", msg["detail"]); return 1
            cur += a.window
        ws.send(json.dumps({"type": "close"}))
        while True:
            msg = json.loads(ws.recv())
            if msg["type"] == "forecast":
                n_fc += 1
            if msg["type"] == "bye":
                print("<< bye", msg); break

    print(f"\n{n_fc} forecasts received")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
