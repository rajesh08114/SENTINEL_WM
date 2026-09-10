"""Manual smoke: start a synthetic live session against a RUNNING backend and
print streamed forecasts.

    # terminal 1
    cd backend && uvicorn app.main:app
    # terminal 2
    cd backend && python -m tests.live_smoke --scenario portscan --speed 30

Needs `websockets` (already a uvicorn[standard] dep) and `httpx`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx
import websockets


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--scenario", default="portscan")
    ap.add_argument("--rate", type=float, default=25)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--speed", type=float, default=30)
    ap.add_argument("--watch", type=float, default=40, help="seconds to stream")
    a = ap.parse_args()

    async with httpx.AsyncClient(base_url=a.api, timeout=10) as c:
        r = await c.post("/live/sessions", json={
            "source": "synthetic", "scenario": a.scenario, "rate": a.rate,
            "duration_s": a.duration, "speed": a.speed})
        r.raise_for_status()
        sid = r.json()["id"]
        print(f"session {sid}  scenario={a.scenario} speed={a.speed}x")

        ws_url = a.api.replace("http", "ws", 1) + f"/live/sessions/{sid}/stream"
        n = 0
        try:
            async with websockets.connect(ws_url) as ws:
                loop = asyncio.get_event_loop()
                end = loop.time() + a.watch
                while loop.time() < end:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    except asyncio.TimeoutError:
                        continue
                    if msg["type"] == "forecast":
                        n += 1
                        h0 = (msg.get("horizon") or [{}])[0]
                        states = "".join(
                            (s.get("progression_state", "?")[:2])
                            for s in (msg.get("horizon") or []))
                        print(f"  win {msg.get('meta', {}).get('stream_window')}  "
                              f"alert={msg.get('alert')}  "
                              f"P(atk)+1={h0.get('attack_prob'):.3f}  "
                              f"prog=[{states}]  "
                              f"tactic={(h0.get('attck') or {}).get('mitre_tactic')}")
                    elif msg["type"] == "status":
                        print(f"  [status] {msg.get('state')} stats={msg.get('stats')}")
                    elif msg["type"] == "bye":
                        print("  [bye]")
                        break
        finally:
            await c.delete(f"/live/sessions/{sid}")
            print(f"deleted session; {n} forecast frames seen")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
