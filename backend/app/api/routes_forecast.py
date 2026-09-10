from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.inference.loader import BundleContractError, BundleNotFound, get_engine
from app.inference.pipeline import BadUpload, forecast
from app.jobs import worker
from app.schemas import ForecastResponse, JobCreated
from app.sentinel_infer.pcap import BadPcap, pcap_to_flows
from app.settings import settings

router = APIRouter(tags=["forecast"])


@router.post("/forecast/csv",
             responses={200: {"model": ForecastResponse}, 202: {"model": JobCreated}})
async def forecast_csv(
    file: UploadFile = File(..., description="CIC-IDS-2017 / CICFlowMeter flow CSV"),
    family_hint: str | None = Form(None, description="optional attack-family guess for the ATT&CK phase"),
    explain: bool = Form(True),
):
    try:
        get_engine()
    except (BundleNotFound, BundleContractError) as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))

    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "empty file")

    # peek row count cheaply
    try:
        n_rows = sum(1 for _ in io.BytesIO(raw)) - 1
    except Exception:
        n_rows = settings.max_sync_flows + 1

    if n_rows > settings.max_sync_flows:
        jid = worker.submit_csv_forecast(raw, family_hint, explain)
        return JobCreated(job_id=jid)

    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        result = await run_in_threadpool(forecast, df, family_hint, explain)
    except BadUpload as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    except Exception as e:                                   # pragma: no cover
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))
    return result


@router.post("/forecast/pcap", responses={200: {"model": ForecastResponse}})
async def forecast_pcap(
    file: UploadFile = File(..., description=".pcap / .pcapng capture"),
    family_hint: str | None = Form(None),
    explain: bool = Form(True),
):
    """Offline PCAP -> reassemble bidirectional flows -> forecast. Same output
    contract as /forecast/csv. Reads only TCP/UDP; other packets are ignored."""
    try:
        get_engine()
    except (BundleNotFound, BundleContractError) as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))

    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "empty file")
    if len(raw) > settings.pcap_max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"pcap is {len(raw) // 1_000_000} MB; ceiling is "
            f"{settings.pcap_max_bytes // 1_000_000} MB. Slice it "
            f"(`editcap -A/-B` or `tshark -c`) or convert to a flow CSV.",
        )

    def _run() -> ForecastResponse:
        df = pcap_to_flows(raw)
        return forecast(df, family_hint, explain)

    try:
        return await run_in_threadpool(_run)
    except (BadPcap, BadUpload) as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    except Exception as e:                                   # pragma: no cover
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))
