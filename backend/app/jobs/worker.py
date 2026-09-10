"""In-process job execution.

A `ThreadPoolExecutor` (not processes) - the world model is loaded once in this
process and torch releases the GIL during inference, so threads share the model
with no re-import cost. Set SENTINEL_JOB_EXECUTOR=process only if you later add a
GIL-bound job kind (e.g. PCAP extraction) that needs isolation.
"""
from __future__ import annotations

import io
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from app.jobs import store
from app.settings import settings

_EXECUTOR = ThreadPoolExecutor(max_workers=settings.workers,
                               thread_name_prefix="sentinel-job")


def shutdown() -> None:
    _EXECUTOR.shutdown(wait=False, cancel_futures=True)


def submit_csv_forecast(csv_bytes: bytes, family_hint: str | None,
                        explain: bool) -> str:
    jid = store.create("forecast_csv")
    _EXECUTOR.submit(_run_csv_forecast, jid, csv_bytes, family_hint, explain)
    return jid


def _run_csv_forecast(jid: str, csv_bytes: bytes, family_hint: str | None,
                      explain: bool) -> None:
    from app.inference.pipeline import forecast, BadUpload
    try:
        store.set_running(jid)
        df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
        store.set_progress(jid, 0.15)
        result = forecast(df, family_hint=family_hint, explain=explain)
        out = Path(settings.job_dir) / f"{jid}.json"
        out.write_text(json.dumps(result, default=float))
        store.finish(jid, str(out))
    except BadUpload as e:
        store.fail(jid, f"bad upload: {e}")
    except Exception as e:                                   # pragma: no cover
        store.fail(jid, f"{e}\n{traceback.format_exc()}")


# generic CPU-bound offload for the sync path and the websocket
def run_blocking(fn, *args, **kwargs):
    return _EXECUTOR.submit(fn, *args, **kwargs)
