"""Telemetry WebSocket.

Client -> server messages (JSON):
  {"type": "hello",  "family_hint": "PortScan"?, "explain": true?}   optional, first
  {"type": "flows",  "records": [ {<flow row>}, ... ]}               repeated
  {"type": "close"}                                                  final flush

Server -> client messages:
  {"type": "ready",    "L":12, "K":6, "window_seconds":10}
  {"type": "ack",      "buffered":N, "total_received":M}
  {"type": "forecast", ...one simulate_anchor dict... }              per closed window
  {"type": "error",    "detail": "..."}
  {"type": "bye"}
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app.inference.loader import BundleNotFound, get_engine
from app.inference.pipeline import BadUpload
from app.streaming.windower import StreamingWindower

router = APIRouter(tags=["stream"])


@router.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        eng = get_engine()
    except BundleNotFound as e:
        await ws.send_json({"type": "error", "detail": str(e)})
        await ws.close()
        return

    win: StreamingWindower | None = None
    sid = f"ws-{id(ws):x}"
    await ws.send_json({"type": "ready", "L": eng.L, "K": eng.K,
                        "window_seconds": eng.window_seconds})
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "hello":
                win = StreamingWindower(sid, family_hint=msg.get("family_hint"),
                                        explain=bool(msg.get("explain", True)))
                continue

            if win is None:
                win = StreamingWindower(sid)

            if mtype == "flows":
                try:
                    n = win.add_flows(msg.get("records") or [])
                except BadUpload as e:
                    await ws.send_json({"type": "error", "detail": f"bad records: {e}"})
                    continue
                forecasts = await run_in_threadpool(win.poll_ready)
                for fc in forecasts:
                    await ws.send_json({"type": "forecast", **fc})
                await ws.send_json({"type": "ack", "buffered": win.stats["buffered"],
                                    "total_received": win.stats["total_received"],
                                    "added": n})

            elif mtype == "close":
                forecasts = await run_in_threadpool(win.flush)
                for fc in forecasts:
                    await ws.send_json({"type": "forecast", **fc})
                await ws.send_json({"type": "bye", **win.stats})
                break

            else:
                await ws.send_json({"type": "error",
                                    "detail": f"unknown message type {mtype!r}"})
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass
