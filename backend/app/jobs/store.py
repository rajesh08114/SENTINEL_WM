"""Tiny SQLite job store - status + progress for async forecast jobs.

No Redis, no ORM. One file, WAL mode, safe for the single-process backend with a
thread pool (sqlite3 with check_same_thread=False + a module lock)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from app.settings import settings

_DB = settings.job_dir / "jobs.sqlite"
_LOCK = threading.Lock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL,
    progress     REAL NOT NULL DEFAULT 0,
    error        TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    result_path  TEXT
);
"""


def init_db() -> None:
    global _conn
    with _LOCK:
        if _conn is None:
            settings.job_dir.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(_DB, check_same_thread=False)
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()


def _c() -> sqlite3.Connection:
    if _conn is None:
        init_db()
    assert _conn is not None
    return _conn


def create(kind: str) -> str:
    jid = uuid.uuid4().hex[:16]
    with _LOCK:
        _c().execute(
            "INSERT INTO jobs(id,kind,status,created_at) VALUES(?,?,?,?)",
            (jid, kind, "queued", time.strftime("%Y-%m-%dT%H:%M:%S")))
        _c().commit()
    return jid


def set_running(jid: str) -> None:
    _update(jid, status="running", progress=0.05)


def set_progress(jid: str, p: float) -> None:
    _update(jid, progress=max(0.0, min(1.0, p)))


def finish(jid: str, result_path: str) -> None:
    _update(jid, status="done", progress=1.0, result_path=result_path,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def fail(jid: str, err: str) -> None:
    _update(jid, status="error", error=err[:2000],
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def _update(jid: str, **fields) -> None:
    cols = ", ".join(f"{k}=?" for k in fields)
    with _LOCK:
        _c().execute(f"UPDATE jobs SET {cols} WHERE id=?",
                     (*fields.values(), jid))
        _c().commit()


def get(jid: str) -> Optional[dict]:
    with _LOCK:
        row = _c().execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["has_result"] = bool(d.get("result_path") and Path(d["result_path"]).exists())
    return d


def result(jid: str) -> Optional[dict]:
    j = get(jid)
    if not j or not j["has_result"]:
        return None
    return json.loads(Path(j["result_path"]).read_text())


def list_jobs(limit: int = 50) -> list[dict]:
    with _LOCK:
        rows = _c().execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["has_result"] = bool(d.get("result_path") and Path(d["result_path"]).exists())
        out.append(d)
    return out
