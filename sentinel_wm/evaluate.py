#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  evaluate.py   -  the benchmark table (proposal section 10)
# -----------------------------------------------------------------------------
# Reads the metric JSONs written by baselines.py and world_model_train.py and
# renders a single side-by-side comparison:
#     * classification  : F1 / Precision / Recall / FPR / AUROC   (any-horizon)
#     * forecasting      : per-horizon F1 table, Mean Lead Time, detection rate
#     * calibration      : Brier(k1), ECE(k1)
#
# Writes artifacts/reports/benchmark.json and artifacts/reports/benchmark.md.
# If a metric file is missing it tells you which phase to run.
# =============================================================================
from __future__ import annotations

import json
import os
from typing import Dict, List

from sentinel_wm import config as C

BASE_JSON = os.path.join(C.BASELINE_DIR, "baseline_metrics.json")
WM_JSON = os.path.join(C.REPORT_DIR, "world_model_metrics.json")


def _load(path, phase):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run `{phase}` first")
    with open(path) as fh:
        return json.load(fh)


def _row(tag, any_h, lt=None, brier=None, ece=None) -> Dict:
    return dict(
        model=tag,
        f1=any_h["f1"], precision=any_h["precision"], recall=any_h["recall"],
        fpr=any_h["fpr"], auroc=any_h["auroc"],
        mean_lead_time_s=(lt or {}).get("mean_lead_time_s", 0.0),
        detection_rate=(lt or {}).get("detection_rate", 0.0),
        false_alarm_rate=(lt or {}).get("false_alarm_rate", 0.0),
        brier_k1=brier, ece_k1=ece)


def collect() -> Dict:
    base = _load(BASE_JSON, "python cli.py baseline")
    wm = _load(WM_JSON, "python cli.py train")

    rows: List[Dict] = []
    for name in ("logistic_regression", "random_forest"):
        if name in base:
            b = base[name]
            rows.append(_row(name, b["any_horizon"], b.get("lead_time"),
                             b.get("brier_k1"), b.get("ece_k1")))
    if "persistence" in base:
        rows.append(_row("persistence", base["persistence"]["any_horizon"]))
    rows.append(_row("SENTINEL-WM", wm["test"]["any_horizon"],
                     wm["test"]["lead_time"], wm["test"]["brier_k1"],
                     wm["test"]["ece_k1"]))

    horizons = {
        "SENTINEL-WM": wm["test"]["per_horizon"],
        "logistic_regression": base.get("logistic_regression", {}).get("per_horizon", []),
        "random_forest": base.get("random_forest", {}).get("per_horizon", []),
    }
    return dict(summary=rows, per_horizon=horizons,
                progression_acc=wm["test"].get("progression_acc"),
                world_model_threshold=wm["test"]["threshold"])


def _md_table(rows: List[Dict]) -> str:
    head = ("| Model | F1 | Precision | Recall | FPR | AUROC | "
            "Mean Lead Time (s) | Detection | FA rate | Brier(k1) | ECE(k1) |")
    sep = "|" + "---|" * 11
    out = [head, sep]
    for r in rows:
        out.append("| {model} | {f1:.3f} | {precision:.3f} | {recall:.3f} | "
                   "{fpr:.3f} | {auroc:.3f} | {mean_lead_time_s:.1f} | "
                   "{detection_rate:.2f} | {false_alarm_rate:.3f} | "
                   "{b} | {e} |".format(
                       b=("-" if r["brier_k1"] is None else f"{r['brier_k1']:.3f}"),
                       e=("-" if r["ece_k1"] is None else f"{r['ece_k1']:.3f}"),
                       **r))
    return "\n".join(out)


def _md_horizon(horizons: Dict) -> str:
    out = ["| k (horizon) | SENTINEL-WM F1 | LogReg F1 | RandForest F1 |",
           "|---|---|---|---|"]
    wm = horizons.get("SENTINEL-WM", [])
    lr = {r["k"]: r for r in horizons.get("logistic_regression", [])}
    rf = {r["k"]: r for r in horizons.get("random_forest", [])}
    for r in wm:
        k = r["k"]
        out.append(f"| +{r['horizon_seconds']}s | {r['f1']:.3f} | "
                   f"{lr.get(k, {}).get('f1', float('nan')):.3f} | "
                   f"{rf.get(k, {}).get('f1', float('nan')):.3f} |")
    return "\n".join(out)


def benchmark(verbose: bool = True) -> Dict:
    data = collect()
    md = ["# SENTINEL-WM Benchmark", "",
          "## Any-horizon classification + forecasting", "",
          _md_table(data["summary"]), "",
          "## Forecast-horizon accuracy (F1 vs lead time)", "",
          _md_horizon(data["per_horizon"]), "",
          f"World-model progression-state accuracy (test): "
          f"**{data['progression_acc']:.3f}**",
          f"World-model alert threshold (FPR-calibrated on val): "
          f"**{data['world_model_threshold']:.2f}**", ""]
    md_text = "\n".join(md)

    with open(os.path.join(C.REPORT_DIR, "benchmark.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    with open(os.path.join(C.REPORT_DIR, "benchmark.md"), "w") as fh:
        fh.write(md_text)

    if verbose:
        print(md_text)
        print(f"\n[evaluate] -> {os.path.join(C.REPORT_DIR, 'benchmark.md')}")
    return data


if __name__ == "__main__":
    benchmark()
