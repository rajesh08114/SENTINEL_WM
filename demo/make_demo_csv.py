"""Carve small, in-distribution demo CSVs from the full CIC-IDS-2017 Wednesday
capture (real DoS Hulk / slowloris / Slowhttptest traffic).

    python demo/make_demo_csv.py                       # rebuild both demo/*.csv
    python demo/make_demo_csv.py --src data/other.csv  # a different source

Output:
  demo/demo_benign.csv     ~4.5k rows, 45 min, no attack   -> expect 0 alerts
  demo/demo_dos_onset.csv  ~7k rows, 36 min, benign -> sustained DoS escalation
                           -> expect the forecaster to alert ~60 s before onset
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "data" / "unified_Wednesday-WorkingHours_labeled.csv"


def _slice(df: pd.DataFrame, start: str, end: str, target_rows: int, path: pathlib.Path) -> None:
    w = df[(df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))]
    step = max(1, len(w) // target_rows)
    out = w.iloc[::step].drop(columns=["ts"])
    out.to_csv(path, index=False)
    ts = pd.to_datetime(out["Timestamp"], dayfirst=True, errors="coerce")
    atk = out["Label"].ne("BENIGN").mean()
    print(f"{path.name}: {len(out)} rows  {ts.min()}..{ts.max()}  attack={atk:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--nrows", type=int, default=400_000)
    a = ap.parse_args()

    df = pd.read_csv(a.src, nrows=a.nrows, low_memory=False)
    df["ts"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    print(f"source span {df['ts'].min()} .. {df['ts'].max()}  ({len(df):,} rows read)")

    _slice(df, "2017-07-05 08:45:00", "2017-07-05 09:30:00", 4000,
           HERE / "demo_benign.csv")
    _slice(df, "2017-07-05 10:20:00", "2017-07-05 10:56:00", 7000,
           HERE / "demo_dos_onset.csv")


if __name__ == "__main__":
    main()
