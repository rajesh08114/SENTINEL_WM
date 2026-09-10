"""Live-subsystem exceptions. Routes translate these to HTTP status codes."""
from __future__ import annotations


class LiveError(Exception):
    """Base for all live-subsystem errors."""


class LiveCapacityError(LiveError):
    """Too many concurrent live sessions (settings.live_max_sessions)."""


class LiveNotFound(LiveError):
    """No live session with the given id."""


class AgentUnavailable(LiveError):
    """A capture session was requested but no capture agent is connected."""
