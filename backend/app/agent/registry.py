"""In-memory registry of connected capture agents (single-tenant)."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

OnFlows = Callable[[list[dict]], Awaitable[None]]


class AgentConnection:
    def __init__(self, ws: Any, name: str, interfaces: list[dict]):
        self.ws = ws
        self.name = name
        self.interfaces = interfaces or []
        self.session_id: Optional[str] = None
        self._on_flows: Optional[OnFlows] = None

    async def send_cmd(self, payload: dict) -> None:
        await self.ws.send_json(payload)

    def bind(self, session_id: str, on_flows: OnFlows) -> None:
        self.session_id = session_id
        self._on_flows = on_flows

    def unbind(self) -> None:
        self.session_id = None
        self._on_flows = None

    async def deliver_flows(self, rows: list[dict]) -> None:
        if self._on_flows and rows:
            await self._on_flows(rows)

    def snapshot(self) -> dict:
        return {"name": self.name, "interfaces": self.interfaces,
                "session_id": self.session_id}


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: list[AgentConnection] = []

    def register(self, conn: AgentConnection) -> None:
        self._agents.append(conn)

    def unregister(self, conn: AgentConnection) -> None:
        if conn in self._agents:
            self._agents.remove(conn)

    def first(self) -> Optional[AgentConnection]:
        return self._agents[0] if self._agents else None

    def by_name(self, name: str) -> Optional[AgentConnection]:
        return next((a for a in self._agents if a.name == name), None)

    def all(self) -> list[AgentConnection]:
        return list(self._agents)

    def snapshot(self) -> dict:
        return {"connected": bool(self._agents),
                "count": len(self._agents),
                "agents": [a.snapshot() for a in self._agents]}


REGISTRY = AgentRegistry()
