"""Ingestion adapter: an uploaded/streamed flows DataFrame -> forecast JSON.

All the real work (normalise -> clean -> state windows -> L-window tensors ->
simulate_anchor -> system blend) lives in `app.sentinel_infer.forecast`. This
module is a stable import surface for the API routes and the streaming windower.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.sentinel_infer.forecast import BadUpload, normalise_upload  # re-export
from app.inference.loader import InferenceEngine, get_engine

__all__ = ["forecast", "BadUpload", "normalise_upload"]


def forecast(df: pd.DataFrame, family_hint: Optional[str] = None,
             explain: bool = True, eng: Optional[InferenceEngine] = None) -> dict:
    return (eng or get_engine()).forecast(df, family_hint=family_hint,
                                          explain=explain)
