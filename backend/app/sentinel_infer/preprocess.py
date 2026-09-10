"""Flow-table cleaning. Vendored subset of research/sentinel_wm/preprocessing.py
(clean_flow_frame + label/day helpers). The `load_and_clean` / parquet paths are
NOT vendored - the backend never reads the raw CSV corpus.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from . import schema as C

_FAMILY_RULES = [
    ("benign", "BENIGN"),
    ("hulk", "DoS Hulk"),
    ("goldeneye", "DoS GoldenEye"),
    ("slowloris", "DoS slowloris"),
    ("slowhttptest", "DoS Slowhttptest"),
    ("heartbleed", "Heartbleed"),
    ("ftp-patator", "FTP-Patator"),
    ("ssh-patator", "SSH-Patator"),
    ("portscan", "PortScan"),
    ("port scan", "PortScan"),
    ("ddos", "DDoS"),
    ("bot", "Bot"),
    ("brute force", "Web Attack Brute Force"),
    ("brute-force", "Web Attack Brute Force"),
    ("xss", "Web Attack XSS"),
    ("sql injection", "Web Attack SQL Injection"),
    ("sql-injection", "Web Attack SQL Injection"),
    ("infiltration", "Infiltration"),
]


def normalise_label(raw: str) -> str:
    s = str(raw).strip().lower()
    for needle, fam in _FAMILY_RULES:
        if needle in s:
            return fam
    return str(raw).strip() or "BENIGN"


def derive_day(source_file: str) -> str:
    s = str(source_file)
    for key, (day, _ord, _split) in C.DAY_SCHEDULE.items():
        if key.lower() in s.lower():
            return day
    return "Unknown"


def _numeric_model_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in (C.TIER1_FLOW_COLS + C.TIER2_PACKET_COLS + C.TRUE_FLAG_COLS)
            if c in df.columns]


def clean_flow_frame(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Clean one flow dataframe and return it. Requires `Label` and
    `flow_start_epoch`; the ingestion adapter injects `Label="BENIGN"`."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if C.LABEL_COL not in df.columns:
        raise KeyError(f"'{C.LABEL_COL}' column missing")
    df[C.LABEL_COL] = df[C.LABEL_COL].fillna("BENIGN").astype(str).str.strip()
    df["attack_family"] = df[C.LABEL_COL].map(normalise_label)
    df["is_attack"] = (df["attack_family"] != "BENIGN").astype(np.int8)

    if "source_day" in df.columns:
        df["day"] = df["source_day"].map(derive_day)
    elif "source_file" in df.columns:
        df["day"] = df["source_file"].map(derive_day)
    elif "day" not in df.columns:
        df["day"] = "Unknown"
    if C.ORDER_COL not in df.columns:
        raise KeyError(f"'{C.ORDER_COL}' missing")
    df = df[df[C.ORDER_COL].notna()].copy()
    df[C.ORDER_COL] = df[C.ORDER_COL].astype(np.float64)

    num_cols = _numeric_model_cols(df)
    for c in num_cols:
        s = df[c]
        if s.dtype == bool:
            df[c] = s.astype(np.float32)
            continue
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        df[c] = s.astype(np.float32)
    for c in num_cols:
        if int(df[c].isna().sum()):
            df[c] = df[c].fillna(0.0)

    # winsorise the unbounded rate columns (train-days-only fit; degrades to all)
    _train_days = {d for d, (_n, _o, s) in C.DAY_SCHEDULE.items() if s == "train"}
    fit_mask = (df["day"].isin(_train_days).to_numpy()
                if "day" in df.columns and df["day"].isin(_train_days).any()
                else np.ones(len(df), bool))
    for c in ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"):
        if c in df.columns:
            hi = np.nanpercentile(df.loc[fit_mask, c].values, 99.9)
            if np.isfinite(hi) and hi > 0:
                df[c] = df[c].clip(upper=float(hi))

    for c in ("Source Port", "Destination Port", "Protocol"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int32)
    for c in ("Source IP", "Destination IP", "Flow ID"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df.sort_values(C.ORDER_COL, kind="mergesort").reset_index(drop=True)
