#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  preprocessing.py   (PHASE 1 - data pre-processing)
# -----------------------------------------------------------------------------
# Input  : one or more `unified_*_labeled.csv` files (extractor.py + test1.ipynb)
# Output : a single tidy `clean_flows.parquet` with
#            - identity columns kept (for windowing / graph, not model input)
#            - Tier 1 + Tier 2 numeric columns cleaned (inf/NaN handled, typed)
#            - a normalised `Label` + `attack_family` + `is_attack` column
#            - `flow_start_epoch` as the single canonical time axis
#            - `day` column derived from `source_file`
#
# Design rules (from the proposal risk table):
#   * raw IPs / timestamps / source_file are NOT dropped here (windowing needs
#     them) but they are tagged so downstream code never feeds them to a model.
#   * CICFlowMeter's Infinity/NaN rate cells are made finite.
#   * no row is silently deleted - counts are reported.
# =============================================================================
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from sentinel_wm import config as C


# -----------------------------------------------------------------------------
# Label normalisation
# -----------------------------------------------------------------------------
# CIC-IDS-2017 label strings vary across day files (en-dash vs hyphen, casing).
# Collapse each raw label to a canonical `attack_family`.
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


# -----------------------------------------------------------------------------
# Core cleaning
# -----------------------------------------------------------------------------
def _numeric_model_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in (C.TIER1_FLOW_COLS + C.TIER2_PACKET_COLS + C.TRUE_FLAG_COLS)
            if c in df.columns]
    return cols


def clean_flow_frame(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean a single unified labelled flow dataframe in place-ish and return it."""
    n0 = len(df)
    df = df.copy()
    df.columns = df.columns.str.strip()

    # ---- labels -------------------------------------------------------------
    if C.LABEL_COL not in df.columns:
        raise KeyError(f"'{C.LABEL_COL}' column missing - run the label mapping "
                       f"notebook (test1.ipynb) first.")
    df[C.LABEL_COL] = df[C.LABEL_COL].fillna("BENIGN").astype(str).str.strip()
    df["attack_family"] = df[C.LABEL_COL].map(normalise_label)
    df["is_attack"] = (df["attack_family"] != "BENIGN").astype(np.int8)

    # ---- day / ordering --------------------------------------------------- -
    if "source_file" in df.columns:
        df["day"] = df["source_file"].map(derive_day)
    else:
        df["day"] = "Unknown"
    if C.ORDER_COL not in df.columns:
        raise KeyError(f"'{C.ORDER_COL}' missing - the extractor must emit it.")
    df = df[df[C.ORDER_COL].notna()].copy()
    df[C.ORDER_COL] = df[C.ORDER_COL].astype(np.float64)

    # ---- numeric columns: bool->int, inf->NaN, NaN->0, clip extremes ------
    num_cols = _numeric_model_cols(df)
    for c in num_cols:
        s = df[c]
        if s.dtype == bool:
            df[c] = s.astype(np.float32)
            continue
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        df[c] = s.astype(np.float32)

    inf_fixed = 0
    for c in num_cols:
        n_nan = int(df[c].isna().sum())
        inf_fixed += n_nan
        if n_nan:
            df[c] = df[c].fillna(0.0)

    # winsorise the notorious unbounded rate columns to a high percentile so a
    # single zero-duration flow cannot dominate window aggregates.
    for c in ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"):
        if c in df.columns:
            hi = np.nanpercentile(df[c].values, 99.9)
            if np.isfinite(hi) and hi > 0:
                df[c] = df[c].clip(upper=float(hi))

    # ---- identity typing --------------------------------------------------- -
    for c in ("Source Port", "Destination Port", "Protocol"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int32)
    for c in ("Source IP", "Destination IP", "Flow ID"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    df = df.sort_values(C.ORDER_COL, kind="mergesort").reset_index(drop=True)

    if verbose:
        print(f"  cleaned {n0:,} -> {len(df):,} rows | "
              f"{len(num_cols)} numeric model cols | "
              f"{inf_fixed:,} inf/NaN cells zero-filled")
        print(f"  family distribution:\n"
              + df["attack_family"].value_counts().to_string().replace("\n", "\n    "))
    return df


def load_and_clean(csv_paths: Optional[List[str]] = None,
                   out_path: Optional[str] = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Load every raw day CSV, clean, concatenate, persist to parquet."""
    csv_paths = csv_paths or C.RAW_FLOW_CSVS
    csv_paths = [p for p in csv_paths if os.path.exists(p)]
    if not csv_paths:
        raise FileNotFoundError(
            "No raw flow CSVs found. Expected at least "
            f"{C.RAW_FLOW_CSVS[0]}")

    frames = []
    for p in csv_paths:
        if verbose:
            print(f"[load] {os.path.basename(p)}")
        df = pd.read_csv(p, low_memory=False)
        frames.append(clean_flow_frame(df, verbose=verbose))

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["day", C.ORDER_COL], kind="mergesort").reset_index(drop=True)

    out_path = out_path or C.CLEAN_FLOWS_PARQUET
    full.to_parquet(out_path, index=False)
    if verbose:
        print(f"[save] {out_path}  ({len(full):,} rows, {full.shape[1]} cols)")
        print(f"[days] {sorted(full['day'].unique())}")
    return full


def load_clean(path: Optional[str] = None) -> pd.DataFrame:
    path = path or C.CLEAN_FLOWS_PARQUET
    if not os.path.exists(path):
        return load_and_clean()
    return pd.read_parquet(path)


if __name__ == "__main__":
    load_and_clean()
