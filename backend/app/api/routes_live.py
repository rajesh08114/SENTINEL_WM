"""Live real-time forecasting sessions.

  POST   /live/sessions            start a synthetic or capture session
  GET    /live/sessions            list sessions
  GET    /live/sessions/{id}       one session's info
  DELETE /live/sessions/{id}       stop + remove
  WS     /live/sessions/{id}/stream   subscribe: status + ring replay, then live

Server -> WS frames: {"type":"status",...} | {"type":"forecast",...anchor...}
                     | {"type":"error","detail":...} | {"type":"bye"}
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.live.errors import AgentUnavailable, LiveCapacityError, LiveNotFound
from app.live.session import MANAGER
from app.live.sources import AgentSource, SyntheticSource
from app.schemas import LiveSessionCreate, LiveSessionInfo
from app.settings import settings
from app.synth.scenarios import SCENARIOS, build_config

router = APIRouter(prefix="/live", tags=["live"])

# so the ATT&CK mapper can name a tactic for a synthetic run (normalise_upload
# strips the row-level attack_family; the scenario is the ground truth we have)
_SCENARIO_FAMILY_HINT = {
    "portscan": "PortScan", "dos_hulk": "DoS Hulk", "bruteforce": "SSH-Patator",
    "botnet_c2": "Bot", "exfil": "Infiltration", "benign": None,
}


@router.post("/sessions", response_model=LiveSessionInfo, status_code=201)
async def create_session(body: LiveSessionCreate) -> LiveSessionInfo:
    if body.source == "synthetic":
        try:
            cfg = build_config(
                body.scenario or "portscan", rate=body.rate,
                duration_s=body.duration_s, seed=body.seed, speed=body.speed,
                attacker_ip=body.attacker_ip, victim_ip=body.victim_ip)
        except KeyError:
            raise HTTPException(422, f"unknown scenario; have {sorted(SCENARIOS)}")
        hint = body.family_hint or _SCENARIO_FAMILY_HINT.get(cfg.name)
        try:
            s = MANAGER.create("synthetic", {
                "scenario": cfg.name, "rate": cfg.rate, "duration_s": cfg.duration_s,
                "seed": cfg.seed, "speed": cfg.speed,
                "family_hint": hint, "explain": body.explain})
        except LiveCapacityError as e:
            raise HTTPException(409, str(e))
        src = SyntheticSource(s, cfg)
        s._source = src                                   # keep a ref
        s.attach_task(asyncio.create_task(src.run()))
        return LiveSessionInfo(**s.info())

    if body.source == "capture":
        if not settings.agent_enabled:
            raise HTTPException(503, "capture agent support is disabled")
        from app.agent.registry import REGISTRY
        agent = REGISTRY.first()
        if agent is None:
            raise HTTPException(503, "no capture agent connected")
        if not body.iface:
            raise HTTPException(422, "`iface` is required for a capture session")
        try:
            s = MANAGER.create("capture", {"iface": body.iface, "bpf": body.bpf,
                                           "family_hint": body.family_hint,
                                           "explain": body.explain})
        except LiveCapacityError as e:
            raise HTTPException(409, str(e))
        src = AgentSource(s, agent=agent, iface=body.iface, bpf=body.bpf)
        s._source = src
        agent.bind(s.id, src.on_flows)
        await src.start()
        return LiveSessionInfo(**s.info())

    raise HTTPException(422, "source must be 'synthetic' or 'capture'")


@router.get("/sessions", response_model=list[LiveSessionInfo])
async def list_sessions() -> list[dict]:
    return MANAGER.list()


@router.get("/sessions/{sid}", response_model=LiveSessionInfo)
async def get_session(sid: str) -> dict:
    try:
        return MANAGER.get(sid).info()
    except LiveNotFound:
        raise HTTPException(404, "no such live session")


@router.delete("/sessions/{sid}", status_code=204)
async def delete_session(sid: str) -> None:
    try:
        await MANAGER.remove(sid)
    except LiveNotFound:
        raise HTTPException(404, "no such live session")


@router.websocket("/sessions/{sid}/stream")
async def stream(ws: WebSocket, sid: str) -> None:
    await ws.accept()
    try:
        s = MANAGER.get(sid)
    except LiveNotFound:
        await ws.send_json({"type": "error", "detail": "no such live session"})
        await ws.close()
        return
    await s.subscribe(ws)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "stop":
                await MANAGER.remove(sid)
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        s.unsubscribe(ws)
        try:
            await ws.close()
        except RuntimeError:
            pass
