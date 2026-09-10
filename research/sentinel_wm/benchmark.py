#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  benchmark.py   -  one comparable scoreboard for every model
# -----------------------------------------------------------------------------
# Discovers every trained model and scores it on the SAME test anchors from
# sequences.npz, with the SAME metric code (metrics.py):
#
#   classical zoo      research/models/classical/*.pkl  (+ .meta.json)
#   neural zoo         research/models/nn/*.pt          (mlp/lstm/gru/tcn)
#   graph attention    research/models/nn/gat.pt
#   world model        artifacts/world_model.pt         (Temporal Transformer + STN)
#   persistence        A_{t+k}=A_t                       (trivial reference)
#
# Outputs (research/benchmarks/ + research/figures/):
#   benchmark_full.csv   one row/model: F1 P R FPR AUROC acc Brier ECE
#                        MLT medianLT detection FA prog_acc params infer_ms + f1_k1..k6
#   per_horizon_f1.csv / per_horizon_auroc.csv
#   leadtime.csv
#   benchmark.md         ranked tables
#   benchmark.json       everything, incl. raw per-model test probabilities meta
#   figures/horizon_f1.png roc.png pr.png reliability.png lead_time.png
# =============================================================================
from __future__ import annotations

import glob
import json
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from sentinel_wm import config as C
from sentinel_wm import metrics as M
from sentinel_wm.sequences import load_sequences

def set_output_root(root: str):
    """redirect all benchmark outputs to `<root>/...` (used for research_zeroshot/)."""
    global BENCH_DIR, FIG_DIR, CLASSICAL_DIR, NN_DIR
    BENCH_DIR = os.path.join(root, "benchmarks")
    FIG_DIR = os.path.join(root, "figures")
    CLASSICAL_DIR = os.path.join(root, "models", "classical")
    NN_DIR = os.path.join(root, "models", "nn")
    for d in (BENCH_DIR, FIG_DIR, CLASSICAL_DIR, NN_DIR):
        os.makedirs(d, exist_ok=True)


set_output_root(C.research_dir())


# -----------------------------------------------------------------------------
# per-model scorers  ->  (probs_k [N,K], prog_k [N,K] or None, meta dict)
# -----------------------------------------------------------------------------
def _score_classical(pkl_path: str, seq: Dict, te_mask) -> Tuple[np.ndarray, None, Dict]:
    meta_path = pkl_path[:-4] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    kind = meta.get("input_kind", "window")
    with open(pkl_path, "rb") as fh:
        ests = pickle.load(fh)
    x = seq["X"][te_mask]
    X = x[:, -1, :] if kind == "window" else x.reshape(x.shape[0], -1)
    from sentinel_wm.baselines import _proba
    t0 = time.perf_counter()
    probs = np.stack([_proba(e, X) for e in ests], 1)
    infer_ms = (time.perf_counter() - t0) * 1000 / max(len(X), 1)
    return probs, None, dict(family="classical", input_kind=kind,
                             threshold=meta.get("threshold"),
                             params=None, infer_ms=infer_ms)


def _score_nn(pt_path: str, seq: Dict, te_mask, device="cpu"
              ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    import torch
    name = os.path.basename(pt_path)[:-3]
    if name == "gat":
        return _score_gat(pt_path, seq, te_mask, device)
    from sentinel_wm.nn_common import load_nn_checkpoint
    model, ck = load_nn_checkpoint(pt_path, device)

    def _run(mask):
        x = torch.as_tensor(seq["X"][mask], dtype=torch.float32, device=device)
        dt = torch.as_tensor(seq["dt"][mask], dtype=torch.float32, device=device)
        pa = []
        pp = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                o = model(x[i:i + 512], dt[i:i + 512])
                pa.append(torch.sigmoid(o["attack_logits_k"]).cpu().numpy())
                pp.append(o["prog_logits_k"].argmax(-1).cpu().numpy())
        return np.concatenate(pa), np.concatenate(pp)

    t0 = time.perf_counter()
    probs, progs = _run(te_mask)
    infer_ms = (time.perf_counter() - t0) * 1000 / max(int(te_mask.sum()), 1)
    # fresh best-F1 threshold on validation (see _score_world_model)
    va = seq["split"] == "val"
    p_va, _ = _run(va)
    thr = M.calibrate_threshold(seq["y_atk"][va].max(1).astype(int),
                                p_va.max(1), C.CONFIG.train.target_fpr)
    return (probs, progs,
            dict(family=ck.get("family", "nn"), kind=ck.get("kind"),
                 threshold=thr,
                 params=sum(p.numel() for p in model.parameters()),
                 infer_ms=infer_ms))


def _score_gat(pt_path, seq, te_mask, device="cpu"):
    import torch
    from sentinel_wm.gat import GATForecaster, _GraphCtx, _GraphDataset
    from torch.utils.data import DataLoader
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    L, K = ck["sequence"]["L"], ck["sequence"]["K"]
    ex = ck.get("extra", {})
    ctx = _GraphCtx(L)
    model = GATForecaster(ex.get("n_node_feat", ctx.Fn), L, K,
                          len(C.PROGRESSION_STATES),
                          d_model=ex.get("d_model", 96))
    model.load_state_dict(ck["state_dict"]); model.eval().to(device)

    def _run(mask):
        ds = _GraphDataset(ctx, seq["day"][mask], seq["window_index"][mask],
                           seq["y_atk"][mask], seq["y_prog"][mask])
        pa, pp = [], []
        with torch.no_grad():
            for nf, aj, mk, _a, _p in DataLoader(ds, batch_size=256):
                o = model(nf.to(device), aj.to(device), mk.to(device))
                pa.append(torch.sigmoid(o["attack_logits_k"]).cpu().numpy())
                pp.append(o["prog_logits_k"].argmax(-1).cpu().numpy())
        return np.concatenate(pa), np.concatenate(pp)

    t0 = time.perf_counter()
    probs, progs = _run(te_mask)
    infer_ms = (time.perf_counter() - t0) * 1000 / max(int(te_mask.sum()), 1)
    va = seq["split"] == "val"
    p_va, _ = _run(va)
    thr = M.calibrate_threshold(seq["y_atk"][va].max(1).astype(int),
                                p_va.max(1), C.CONFIG.train.target_fpr)
    return (probs, progs,
            dict(family="graph", kind="gat", threshold=thr,
                 params=sum(p.numel() for p in model.parameters()),
                 infer_ms=infer_ms))


def _score_world_model(seq: Dict, te_mask, device="cpu"):
    import torch
    from sentinel_wm.models import build_model, wm_predict
    if not os.path.exists(C.WORLD_MODEL_PT):
        return None
    ck = torch.load(C.WORLD_MODEL_PT, map_location=device, weights_only=False)
    cfg = C.CONFIG
    cfg.sequence.horizon = ck["sequence"]["K"]
    model = build_model(ck["config"]["n_features"], cfg).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    snaps = ck.get("snapshots", [])
    self_ens = bool(ck.get("self_ensemble", False))
    t0 = time.perf_counter()
    probs, progs = wm_predict(model, seq["X"][te_mask], seq["dt"][te_mask], device,
                              snapshots=snaps, self_ensemble=self_ens)
    infer_ms = (time.perf_counter() - t0) * 1000 / max(int(te_mask.sum()), 1)
    kind = "gru+STN" if getattr(cfg.model, "encoder", "gru") == "gru" \
        else "temporal_transformer+STN"
    if self_ens or snaps:
        kind += f" self-ens({len(snaps)}snap)"
    n_params = sum(p.numel() for p in model.parameters())

    # fresh best-F1 threshold on VALIDATION (the stored alert_threshold was
    # FPR-calibrated during training and transfers badly to the test prevalence)
    va = seq["split"] == "val"
    wm_va, _ = wm_predict(model, seq["X"][va], seq["dt"][va], device,
                          snapshots=snaps, self_ensemble=self_ens)
    thr_wm = M.calibrate_threshold(seq["y_atk"][va].max(1).astype(int),
                                   wm_va.max(1), cfg.train.target_fpr)
    out = [(probs, progs, dict(family="world_model", kind=kind,
                               threshold=thr_wm,
                               params=n_params, infer_ms=infer_ms))]

    # the deployed "SENTINEL-WM system" = WM self-ensemble blended with the
    # strong sequence models (its distillation teachers + GAT); blend weight
    # tuned on VALIDATION. This is what you ship.
    if getattr(cfg.train, "system_blend_teachers", False):
        members = getattr(cfg.train, "system_members", ())
        mp_te = _member_probs(seq, te_mask, members, device)
        mp_va = _member_probs(seq, va, members, device)
        if mp_te is not None and mp_va is not None:
            y_va = seq["y_atk"][va].max(1).astype(int)
            best_w, best_s = 0.5, -1.0
            for w in np.linspace(0.2, 0.85, 14):
                s = M.binary_scores(y_va, (w * wm_va + (1 - w) * mp_va).max(1), 0.5)
                if s["pr_auc"] and s["pr_auc"] > best_s:
                    best_s, best_w = s["pr_auc"], float(w)
            sysp = best_w * probs + (1 - best_w) * mp_te
            thr_sys = M.calibrate_threshold(
                y_va, (best_w * wm_va + (1 - best_w) * mp_va).max(1),
                cfg.train.target_fpr)
            out.append((sysp, progs, dict(
                family="system",
                kind=f"WM self-ens x{best_w:.2f} + [{'+'.join(members)}]",
                threshold=thr_sys, params=n_params, infer_ms=infer_ms)))
    return out


def _member_probs(seq, mask, names, device="cpu"):
    """per-horizon P(attack) for a set of already-trained models, on `mask`."""
    import glob, pickle
    from sentinel_wm.baselines import _proba
    cdir = os.path.join(C.research_dir(), "models", "classical")
    Xseq = seq["X"][mask].reshape(int(mask.sum()), -1)
    got = []
    for nm in names:
        pk = os.path.join(cdir, f"{nm}.pkl")
        pt = os.path.join(NN_DIR, f"{nm}.pt")
        try:
            if os.path.exists(pk):
                with open(pk, "rb") as fh:
                    ests = pickle.load(fh)
                got.append(np.stack([_proba(e, Xseq) for e in ests], 1))
            elif os.path.exists(pt):
                p, _pr, _m = _score_nn(pt, seq, mask, device)
                got.append(p)
        except Exception:
            pass
    return np.mean(got, 0) if got else None


# -----------------------------------------------------------------------------
def _metrics_row(name: str, probs_k, prog_k, meta, seq, te_mask, cfg) -> Dict:
    W, K = cfg.window.window_seconds, int(seq["K"])
    y_atk = seq["y_atk"][te_mask].astype(int)
    y_prog = seq["y_prog"][te_mask].astype(int)
    y_now = seq["y_now"][te_mask].astype(int)
    wi = seq["window_index"][te_mask]

    thr = meta.get("threshold")
    if thr is None:
        thr = M.calibrate_threshold(y_atk.max(1), probs_k.max(1), cfg.train.target_fpr)
    any_k = M.binary_scores(y_atk.max(1), probs_k.max(1), thr)
    per_h = M.horizon_table(y_atk, probs_k, W, thr)
    lt = M.lead_time(wi, y_now, probs_k.max(1), W, thr, K)
    cal = M.expected_calibration_error(y_atk[:, 0], probs_k[:, 0])
    row = dict(
        model=name, family=meta.get("family", "?"), kind=meta.get("kind", ""),
        input=meta.get("input_kind", "sequence"),
        params=meta.get("params"), infer_ms=round(meta.get("infer_ms", 0.0), 4),
        threshold=round(float(thr), 3),
        f1=any_k["f1"], f1_best=any_k.get("f1_best"),
        precision=any_k["precision"], recall=any_k["recall"],
        fpr=any_k["fpr"], auroc=any_k["auroc"], pr_auc=any_k.get("pr_auc"),
        accuracy=any_k["accuracy"],
        brier_k1=M.brier_score(y_atk[:, 0], probs_k[:, 0]), ece_k1=cal["ece"],
        mean_lead_time_s=lt["mean_lead_time_s"],
        median_lead_time_s=lt["median_lead_time_s"],
        detection_rate=lt["detection_rate"], false_alarm_rate=lt["false_alarm_rate"],
        n_episodes=lt["n_episodes"], n_episodes_warned=lt["n_episodes_warned"])
    for r in per_h:
        row[f"f1_k{r['k']}"] = round(r["f1"], 4)
        row[f"auroc_k{r['k']}"] = round(r["auroc"], 4)
    if prog_k is not None:
        row["progression_acc"] = float((prog_k == y_prog).mean())
        row["progression_acc_k1"] = float((prog_k[:, 0] == y_prog[:, 0]).mean())
    else:
        row["progression_acc"] = None
        row["progression_acc_k1"] = None
    # ---- per-attack-family breakdown --------------------------------------
    fams = seq.get("dominant_family")
    fams = fams[te_mask] if fams is not None else _anchor_families(seq, te_mask)
    if fams is not None:
        row["_per_family"] = M.per_family_scores(
            y_atk.max(1), probs_k.max(1), fams, thr, min_support=8)

    row["_per_horizon"] = per_h
    row["_probs_any_k"] = probs_k.max(1)          # kept for ROC/PR plots
    row["_probs_k"] = probs_k                     # kept for the ensemble
    return row


_FAM_LOOKUP = {}


def _anchor_families(seq, te_mask):
    """fallback: (day, window_index) -> dominant_family via state_windows.parquet."""
    global _FAM_LOOKUP
    if not _FAM_LOOKUP:
        try:
            import pandas as pd
            sw = pd.read_parquet(C.STATE_WINDOWS_PARQUET,
                                 columns=["day", "window_index", "dominant_family"])
            _FAM_LOOKUP = {(str(r.day), int(r.window_index)): str(r.dominant_family)
                           for r in sw.itertuples()}
        except Exception:
            return None
    days = seq["day"][te_mask]
    wins = seq["window_index"][te_mask]
    return np.array([_FAM_LOOKUP.get((str(d), int(w)), "?")
                     for d, w in zip(days, wins)], object)


# -----------------------------------------------------------------------------
def run_benchmark(cfg: C.Config = None, device: str = None,
                  make_figures: bool = True, verbose: bool = True) -> Dict:
    cfg = cfg or C.CONFIG
    device = C.resolve_device(device or "cpu")
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    seq = load_sequences()
    te = seq["split"] == "test"
    K = int(seq["K"])

    rows: List[Dict] = []

    # persistence
    y_now = seq["y_now"][te].astype(int)
    pers = np.repeat(y_now[:, None], K, axis=1).astype(float)
    rows.append(_metrics_row("persistence", pers, None,
                             dict(family="reference", threshold=0.5), seq, te, cfg))

    # classical
    for pkl in sorted(glob.glob(os.path.join(CLASSICAL_DIR, "*.pkl"))):
        name = os.path.basename(pkl)[:-4]
        try:
            p, pr, meta = _score_classical(pkl, seq, te)
            rows.append(_metrics_row(name, p, pr, meta, seq, te, cfg))
        except Exception as e:
            print(f"[bench] classical {name} failed: {e}")

    # neural zoo + gat
    for pt in sorted(glob.glob(os.path.join(NN_DIR, "*.pt"))):
        name = os.path.basename(pt)[:-3]
        try:
            p, pr, meta = _score_nn(pt, seq, te, device)
            rows.append(_metrics_row(name, p, pr, meta, seq, te, cfg))
        except Exception as e:
            print(f"[bench] nn {name} failed: {e}")

    # world model (+ optional "SENTINEL-WM (system)" = WM self-ens + teachers)
    wm = _score_world_model(seq, te, device)
    if wm is not None:
        for j, (p, pr, meta) in enumerate(wm):
            nm = "SENTINEL-WM" if meta.get("family") == "world_model" \
                else "SENTINEL-WM (system)"
            rows.append(_metrics_row(nm, p, pr, meta, seq, te, cfg))

    # (a plain unweighted mean-ensemble of the same members is a strictly weaker
    # version of "SENTINEL-WM (system)" above - it used the WM's single head, no
    # weight tuning - so it is not reported as a separate row.)

    # rank by PR-AUC (threshold-free; robust to the val->test prevalence shift
    # that makes a fixed-FPR threshold's F1 noisy), then f1_best.
    rows.sort(key=lambda r: (-(r.get("pr_auc") or 0), -(r.get("f1_best") or 0)))
    _write_tables(rows, seq, te, cfg, verbose)
    if make_figures:
        try:
            _figures(rows, seq, te, cfg)
        except Exception as e:
            print(f"[bench] figures skipped: {e}")
    return dict(rows=[{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rows])


# -----------------------------------------------------------------------------
def _write_tables(rows, seq, te, cfg, verbose):
    import csv
    K = int(seq["K"]); W = cfg.window.window_seconds
    cols = ["model", "family", "kind", "input", "params", "infer_ms", "threshold",
            "f1", "f1_best", "precision", "recall", "fpr", "auroc", "pr_auc",
            "accuracy", "brier_k1", "ece_k1", "mean_lead_time_s", "median_lead_time_s",
            "detection_rate", "false_alarm_rate", "n_episodes_warned",
            "n_episodes", "progression_acc", "progression_acc_k1"] + \
           [f"f1_k{k+1}" for k in range(K)] + [f"auroc_k{k+1}" for k in range(K)]

    with open(os.path.join(BENCH_DIR, "benchmark_full.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})

    with open(os.path.join(BENCH_DIR, "per_horizon_f1.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + [f"+{(k+1)*W}s" for k in range(K)])
        for r in rows:
            w.writerow([r["model"]] + [r.get(f"f1_k{k+1}", "") for k in range(K)])

    with open(os.path.join(BENCH_DIR, "leadtime.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "mean_lead_time_s", "median_lead_time_s",
                    "detection_rate", "false_alarm_rate",
                    "episodes_warned", "episodes"])
        for r in rows:
            w.writerow([r["model"], r["mean_lead_time_s"], r["median_lead_time_s"],
                        r["detection_rate"], r["false_alarm_rate"],
                        r["n_episodes_warned"], r["n_episodes"]])

    # markdown
    def fmt(v, p=3):
        return "-" if v is None or v == "" else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    md = ["# SENTINEL-WM - Full Model Benchmark", "",
          f"Test split = **Friday** ({int(te.sum())} sequences, "
          f"{seq['y_atk'][te].max(1).mean():.1%} attack). Threshold = best-F1 on "
          f"validation within an FPR<=15% budget (per model).",
          "", "## Ranked by PR-AUC (threshold-free)", "",
          "`F1` = at the val-tuned best-F1 threshold - `F1*` = best achievable by "
          "sweeping the threshold on test (ceiling) - `PR-AUC` / `AUROC` = "
          "threshold-free. With ~76 positive test sequences the top cluster is "
          "within noise; PR-AUC / F1* / AUROC are the trustworthy ranks.",
          "",
          "| Model | Family | In | F1 | F1* | PR-AUC | Prec | Rec | FPR | AUROC | "
          "Brier | ECE | MLT (s) | Detect | FA | ProgAcc | Params | infer ms |",
          "|" + "---|" * 19]
    for r in rows:
        md.append("| {model} | {family} | {input} | {f1} | {f1b} | {prauc} | "
                  "{precision} | {recall} | "
                  "{fpr} | {auroc} | {brier_k1} | {ece_k1} | {mlt} | {det} | {fa} | "
                  "{pa} | {pr} | {im} |".format(
                      f1b=fmt(r.get("f1_best")), prauc=fmt(r.get("pr_auc")),
                      model=r["model"], family=r["family"], input=r["input"],
                      f1=fmt(r["f1"]), precision=fmt(r["precision"]),
                      recall=fmt(r["recall"]), fpr=fmt(r["fpr"]),
                      auroc=fmt(r["auroc"]), brier_k1=fmt(r["brier_k1"]),
                      ece_k1=fmt(r["ece_k1"]), mlt=fmt(r["mean_lead_time_s"], 1),
                      det=fmt(r["detection_rate"], 2), fa=fmt(r["false_alarm_rate"], 3),
                      pa=fmt(r["progression_acc"]),
                      pr=("-" if r["params"] in (None, "") else f"{int(r['params']):,}"),
                      im=fmt(r["infer_ms"], 3)))
    md += ["", "## Forecast-horizon F1 (holds up as the horizon grows?)", "",
           "| Model | " + " | ".join(f"+{(k+1)*W}s" for k in range(K)) + " |",
           "|" + "---|" * (K + 1)]
    for r in rows:
        md.append("| " + r["model"] + " | " +
                  " | ".join(fmt(r.get(f"f1_k{k+1}")) for k in range(K)) + " |")

    # ---- per-attack-family breakdown -----------------------------------------
    pf_rows = _collect_per_family(rows)
    if pf_rows:
        with open(os.path.join(BENCH_DIR, "per_family.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "family", "windows", "positives", "threshold",
                        "f1", "f1_best", "precision", "recall", "fpr", "auroc",
                        "pr_auc", "skipped"])
            for pr in pf_rows:
                w.writerow(pr)
        # markdown: family x top-8 models, F1 (recall)
        fams = sorted({pr[1] for pr in pf_rows if pr[1] != "BENIGN"})
        top_models = [r["model"] for r in rows[:8]]
        by = {(pr[0], pr[1]): pr for pr in pf_rows}
        md += ["", "## Per-attack-family detection (F1 / recall, non-skipped)",
               "", "| Family | n+ | " + " | ".join(top_models) + " |",
               "|" + "---|" * (len(top_models) + 2)]
        for f in fams:
            npos = next((by[(m, f)][3] for m in top_models if (m, f) in by), "-")
            cells = []
            for m in top_models:
                pr = by.get((m, f))
                if pr is None or pr[12] in ("True", True):
                    cells.append("-")
                else:
                    cells.append(f"{float(pr[5]):.2f}/{float(pr[8]):.2f}")  # f1/recall
            md.append(f"| {f} | {npos} | " + " | ".join(cells) + " |")
        skipped = sorted({pr[1] for pr in pf_rows if pr[12] in ("True", True)})
        if skipped:
            md.append("")
            md.append(f"_Skipped (< 8 positive test windows): {', '.join(skipped)}._")

    md_txt = "\n".join(md) + "\n"
    open(os.path.join(BENCH_DIR, "benchmark.md"), "w", encoding="utf-8").write(md_txt)

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    json.dump({"rows": clean,
               "split_mode": getattr(cfg.split, "mode", None),
               "per_family": [dict(zip(
                   ["model", "family", "windows", "positives", "threshold", "f1",
                    "f1_best", "precision", "recall", "fpr", "auroc", "pr_auc",
                    "skipped"], pr)) for pr in (pf_rows or [])]},
              open(os.path.join(BENCH_DIR, "benchmark.json"), "w"),
              indent=2, default=float)
    if verbose:
        print(md_txt)
        print(f"[bench] -> {BENCH_DIR}")


def _collect_per_family(rows):
    """flatten every row's `_per_family` dict into csv-ready tuples."""
    out = []
    for r in rows:
        pf = r.get("_per_family")
        if not pf:
            continue
        for fam, s in pf.items():
            if s.get("skipped"):
                out.append((r["model"], fam, s.get("windows", 0),
                            s.get("positives", 0), "", "", "", "", "", "", "",
                            "", True))
            else:
                out.append((r["model"], fam, s.get("n", 0), s.get("positives", 0),
                            round(s.get("threshold", 0), 3),
                            round(s.get("f1", 0), 4), round(s.get("f1_best", 0), 4),
                            round(s.get("precision", 0), 4), round(s.get("recall", 0), 4),
                            round(s.get("fpr", 0), 4),
                            round(s.get("auroc", 0) if s.get("auroc") == s.get("auroc") else 0, 4),
                            round(s.get("pr_auc", 0) if s.get("pr_auc") == s.get("pr_auc") else 0, 4),
                            False))
    return out


# -----------------------------------------------------------------------------
def _figures(rows, seq, te, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    K = int(seq["K"]); W = cfg.window.window_seconds
    y_any = seq["y_atk"][te].max(1).astype(int)
    top = rows[:8]

    # horizon F1
    plt.figure(figsize=(8, 5))
    for r in top:
        plt.plot([(k + 1) * W for k in range(K)],
                 [r.get(f"f1_k{k+1}", np.nan) for k in range(K)], "o-", label=r["model"])
    plt.xlabel("forecast horizon (s)"); plt.ylabel("F1"); plt.grid(alpha=.3)
    plt.legend(fontsize=8); plt.title("Forecast-horizon F1")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "horizon_f1.png"), dpi=130)
    plt.close()

    # ROC + PR (any-horizon)
    for fn, curve, xl, yl, title in [
        ("roc.png", roc_curve, "FPR", "TPR", "ROC (any-horizon attack)"),
        ("pr.png", precision_recall_curve, "Recall", "Precision", "PR (any-horizon attack)")]:
        plt.figure(figsize=(6, 6))
        for r in top:
            p = r["_probs_any_k"]
            if fn == "roc.png":
                a, b, _ = curve(y_any, p)
            else:
                b, a, _ = curve(y_any, p)
            plt.plot(a, b, label=r["model"], lw=1.4)
        plt.xlabel(xl); plt.ylabel(yl); plt.title(title); plt.legend(fontsize=8)
        plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, fn), dpi=130); plt.close()

    # lead-time bar
    plt.figure(figsize=(9, 4))
    names = [r["model"] for r in rows]
    plt.bar(names, [r["mean_lead_time_s"] or 0 for r in rows])
    plt.ylabel("Mean Lead Time (s)"); plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.title("Mean Lead Time by model"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "lead_time.png"), dpi=130); plt.close()
    print(f"[bench] figures -> {FIG_DIR}")


# back-compat: old name used by cli.py / evaluate.py
def benchmark(verbose: bool = True):
    return run_benchmark(verbose=verbose)


if __name__ == "__main__":
    run_benchmark()
