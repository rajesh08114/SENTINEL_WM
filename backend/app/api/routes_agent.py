"""Capture-agent channel.

  WS  /agent            a sentinel-capture agent connects here
  GET /agent/status     what the console shows in the interface picker
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.registry import REGISTRY, AgentConnection

router = APIRouter(tags=["agent"])


@router.get("/agent/status")
async def agent_status() -> dict:
    return REGISTRY.snapshot()


@router.websocket("/agent")
async def agent_ws(ws: WebSocket) -> None:
    await ws.accept()
    conn: AgentConnection | None = None
    try:
        hello = await ws.receive_json()
        if hello.get("type") != "hello":
            await ws.send_json({"type": "error", "detail": "expected a hello frame"})
            await ws.close()
            return
        conn = AgentConnection(ws, hello.get("name") or "agent",
                               hello.get("interfaces") or [])
        REGISTRY.register(conn)
        await ws.send_json({"type": "ready"})

        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "flows":
                await conn.deliver_flows(msg.get("records") or [])
            elif mtype == "interfaces":
                conn.interfaces = msg.get("interfaces") or conn.interfaces
            elif mtype in ("started", "stopped", "hello"):
                continue
            else:
                await ws.send_json({"type": "error",
                                    "detail": f"unknown type {mtype!r}"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if conn is not None:
            REGISTRY.unregister(conn)
            if conn.session_id:
                from app.live.session import MANAGER
                try:
                    await MANAGER.remove(conn.session_id)
                except Exception:
                    pass
        try:
            await ws.close()
        except RuntimeError:
            pass
