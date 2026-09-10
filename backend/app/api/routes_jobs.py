from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.jobs import store
from app.schemas import JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobStatus])
def list_jobs(limit: int = 50) -> list[dict]:
    return store.list_jobs(limit)


@router.get("/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> dict:
    j = store.get(job_id)
    if not j:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
    return j


@router.get("/{job_id}/result")
def job_result(job_id: str) -> dict:
    j = store.get(job_id)
    if not j:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
    if j["status"] == "error":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, j.get("error") or "job failed")
    if j["status"] != "done":
        raise HTTPException(status.HTTP_409_CONFLICT, f"job {j['status']}")
    return store.result(job_id) or {}
