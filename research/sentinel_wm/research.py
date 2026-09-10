#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  research.py   -  reproducible research run -> research/ folder
# -----------------------------------------------------------------------------
#   python -m sentinel_wm.research all          # everything, ~30-60 min
#   python -m sentinel_wm.research all --quick   # small epoch budgets
#   python -m sentinel_wm.research profile|baselines|worldmodel|nn|gat|benchmark|
#                                  explain|simulate|report
#
# Populates research/ with every artefact + its proof:
#   data_profile/   label & timeline stats, split balance
#   models/         classical/*.pkl  nn/*.pt  world_model.pt  registry.json
#   benchmarks/     benchmark_full.csv/.md/.json  per_horizon_f1.csv  leadtime.csv
#   figures/        horizon_f1 roc pr lead_time reliability  + data-profile plots
#   explainability/ shap_*.json  saliency  per-class attribution
#   simulations/    forward_sim scenarios (probability timelines + ATT&CK stages)
#   reports/RESEARCH_REPORT.md   narrative with every claim linked to a file
#   run_config.json logs/
# =============================================================================
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import time
from dataclasses import asdict

import numpy as np

from sentinel_wm import config as C

R = ""
DIRS = {}


def _set_output_root(name: str = None):
    """point the generated-output dir (and benchmark's dirs) at `<ROOT>/<name>`
    - lets a zero-shot run write to `<runs>_zeroshot/` without clobbering the
    primary. Default name = C.RUN_DIR_NAME ("runs")."""
    global R, DIRS
    name = name or C.RUN_DIR_NAME
    R = os.path.join(C.ROOT, name)
    DIRS = {k: os.path.join(R, k) for k in
            ("data_profile", "models", "benchmarks", "figures",
             "explainability", "simulations", "reports", "logs")}
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(DIRS["models"], "classical"), exist_ok=True)
    os.makedirs(os.path.join(DIRS["models"], "nn"), exist_ok=True)
    os.environ["SENTINEL_WM_RESEARCH_DIR"] = R
    from sentinel_wm import benchmark as _b
    _b.set_output_root(R)


_set_output_root(os.environ.get("SENTINEL_WM_RESEARCH_DIR_NAME", C.RUN_DIR_NAME))


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# -----------------------------------------------------------------------------
def step_profile(verbose=True):
    import pandas as pd
    from sentinel_wm import preprocessing
    from sentinel_wm.state_windows import load_state_windows
    from sentinel_wm.sequences import load_sequences, assign_split
    plt = _mpl()

    flows = preprocessing.load_clean()
    lab = flows["attack_family"].value_counts()
    lab.to_csv(os.path.join(DIRS["data_profile"], "flow_label_distribution.csv"))

    plt.figure(figsize=(9, 4))
    lab.plot(kind="bar", logy=True)
    plt.ylabel("flows (log)"); plt.title("CIC-IDS-2017 flow label distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS["figures"], "flow_label_distribution.png"), dpi=130)
    plt.close()

    # per-day attack timeline
    g = flows.copy()
    g["min"] = g.groupby("day")["flow_start_epoch"].transform(lambda s: (s - s.min()) / 60)
    plt.figure(figsize=(11, 5))
    fams = [f for f in g.loc[g.is_attack == 1, "attack_family"].unique()]
    for i, (day, sub) in enumerate(g[g.is_attack == 1].groupby("day")):
        for fam in fams:
            s2 = sub[sub.attack_family == fam]
            if len(s2):
                plt.scatter(s2["min"], [f"{day[:3]}-{fam[:14]}"] * len(s2), s=2)
    plt.xlabel("minutes into capture day"); plt.title("attack flow timeline by day")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS["figures"], "attack_timeline_by_day.png"), dpi=130)
    plt.close()

    sw = load_state_windows()
    sw = sw.assign(split=assign_split(sw, C.CONFIG.split, C.CONFIG.window, verbose=False))
    rows = []
    for s in ("train", "val", "test"):
        m = sw.split == s
        rows.append(dict(split=s, windows=int(m.sum()),
                         attack_windows=int(sw.loc[m, "y_attack"].sum()),
                         attack_frac=round(float(sw.loc[m, "y_attack"].mean()), 4),
                         days=",".join(sorted(sw.loc[m, "day"].unique())),
                         families=";".join(sorted(
                             x for x in sw.loc[m & (sw.dominant_family != "BENIGN"),
                                               "dominant_family"].unique()))))
    pd.DataFrame(rows).to_csv(os.path.join(DIRS["data_profile"], "split_summary.csv"),
                              index=False)
    (sw.groupby(["split", "progression_state"]).size().unstack(fill_value=0)
       .to_csv(os.path.join(DIRS["data_profile"], "split_progression_states.csv")))
    # per-family coverage: attack windows per (dominant_family, split) - the
    # artifact that proves every family spans all 3 splits under `stratified`
    (sw[sw.y_attack == 1]
       .pivot_table(index="dominant_family", columns="split",
                    values="window_index", aggfunc="count", fill_value=0)
       .to_csv(os.path.join(DIRS["data_profile"], "split_family_windows.csv")))
    seq = load_sequences()
    prof = dict(n_flows=int(len(flows)), n_windows=int(len(sw)),
                n_sequences=int(len(seq["X"])), L=int(seq["L"]), K=int(seq["K"]),
                n_features=int(seq["X"].shape[-1]),
                split_summary=rows)
    # ---- flow-level augmentation provenance (if any) ----------------------
    if "is_synthetic" in sw.columns and int(sw["is_synthetic"].sum()) > 0:
        syn = sw["is_synthetic"] == 1
        tr = sw["split"] == "train"
        prof["flow_augment"] = True
        prof["n_synthetic_windows"] = int(syn.sum())
        prof["n_synthetic_attack_windows"] = int(sw.loc[syn, "y_attack"].sum())
        prof["train_attack_frac_before"] = round(
            float(sw.loc[tr & ~syn, "y_attack"].mean()), 4)
        prof["train_attack_frac_after"] = round(
            float(sw.loc[tr, "y_attack"].mean()), 4)
        if "is_synthetic" in seq:
            prof["n_synthetic_sequences"] = int((seq["is_synthetic"] == 1).sum())
    else:
        prof["flow_augment"] = False
    json.dump(prof, open(os.path.join(DIRS["data_profile"], "profile.json"), "w"),
              indent=2, default=float)
    if verbose:
        print("[profile]\n" + json.dumps(rows, indent=2))
    return prof


# -----------------------------------------------------------------------------
def step_flowaug(verbose=True):
    from sentinel_wm import flow_augment
    return flow_augment.build_augmented_flows(verbose=verbose)


def step_baselines(verbose=True):
    from sentinel_wm import baselines
    return baselines.run_baselines(verbose=verbose)


def step_worldmodel(epochs=40, device=None, verbose=True):
    from sentinel_wm import train
    from sentinel_wm.pretrain import OUT as PRE_OUT
    # fresh SSL pretrain + fresh snapshots for this run
    for p in (PRE_OUT,):
        if os.path.exists(p):
            os.remove(p)
    sd = os.path.join(C.ARTIFACTS, "world_model_snapshots")
    if os.path.isdir(sd):
        shutil.rmtree(sd)
    ns = argparse.Namespace(test=False, epochs=epochs, batch_size=None,
                            lr=None, device=device)
    rep = train.train(C.CONFIG, ns)
    shutil.copy(C.WORLD_MODEL_PT, os.path.join(DIRS["models"], "world_model.pt"))
    if os.path.exists(C.SCALER_PKL):
        shutil.copy(C.SCALER_PKL, os.path.join(DIRS["models"], "state_scaler.pkl"))
    return rep


def step_nn(epochs=50, device=None, kinds=None, verbose=True):
    from sentinel_wm.nn_common import run_nn_zoo
    return run_nn_zoo(kinds=kinds, epochs=epochs, device=device, verbose=verbose)


def step_gat(epochs=50, device=None, verbose=True):
    from sentinel_wm.graph_windows import GRAPH_NPZ, build_graph_windows
    from sentinel_wm.gat import run_gat
    if not os.path.exists(GRAPH_NPZ):
        build_graph_windows(verbose=verbose)
    return run_gat(epochs=epochs, device=device, verbose=verbose)


def step_benchmark(device=None, verbose=True):
    from sentinel_wm import benchmark, registry
    data = benchmark.run_benchmark(device=device, verbose=verbose)
    # mirror benchmark outputs into research/benchmarks (they already write there)
    registry.build_registry(verbose=verbose)
    return data


def step_explain(device=None, verbose=True):
    from sentinel_wm import explain
    rep = explain.run_all(device or "cpu")
    src = os.path.join(C.REPORT_DIR, "explainability.json")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DIRS["explainability"], "explainability.json"))
    # figure: world-model SHAP bar
    try:
        plt = _mpl()
        tf = rep["world_model_shap"]["top_features"][:12]
        plt.figure(figsize=(7, 5))
        plt.barh([t["feature"] for t in tf][::-1],
                 [t["mean_abs_shap"] for t in tf][::-1])
        plt.xlabel("mean |attribution|")
        plt.title(f"World model - {rep['world_model_shap']['method']}")
        plt.tight_layout()
        plt.savefig(os.path.join(DIRS["figures"], "shap_world_model.png"), dpi=130)
        plt.close()
    except Exception as e:
        print(f"[explain] figure skipped: {e}")
    return rep


def step_simulate(device=None, verbose=True):
    from sentinel_wm import forward_sim
    plt = _mpl()
    out = {}
    res = forward_sim.simulate_split("test", limit=None, device=device or "cpu",
                                     with_explain=True)
    json.dump(res, open(os.path.join(DIRS["simulations"], "forward_sim_test.json"), "w"),
              indent=2, default=float)
    # pick a few interesting anchors: highest max_attack_prob among true attacks
    atk = [r for r in res if r["meta"].get("y_now") == 1]
    atk.sort(key=lambda r: -r["max_attack_prob"])
    sample = atk[:6] + [r for r in res if not r["alert"]][:2]
    plt.figure(figsize=(10, 5))
    for r in sample:
        ks = [h["horizon_seconds"] for h in r["horizon"]]
        plt.plot(ks, [h["attack_prob"] for h in r["horizon"]], "o-",
                 label=f"win{r['meta']['window_index']} now={r['meta'].get('y_now')}")
    plt.axhline(res[0]["alert_threshold"], color="r", ls="--", label="alert thr")
    plt.xlabel("horizon (s)"); plt.ylabel("P(attack)"); plt.legend(fontsize=7)
    plt.title("K-step forward-simulation probability timelines")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS["figures"], "forward_sim_timelines.png"), dpi=130)
    plt.close()
    # ATT&CK stage table
    import csv
    with open(os.path.join(DIRS["simulations"], "attck_stage_forecast.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_index", "y_now", "k", "horizon_s", "attack_prob",
                    "progression_state", "attck_phase", "confidence"])
        for r in res:
            for h in r["horizon"]:
                w.writerow([r["meta"]["window_index"], r["meta"].get("y_now"),
                            h["k"], h["horizon_seconds"], round(h["attack_prob"], 4),
                            h["progression_state"],
                            h["attck"]["kill_chain_phase"], h["attck"]["confidence"]])
    out["n_anchors"] = len(res)
    out["n_alert"] = sum(r["alert"] for r in res)
    if verbose:
        print(f"[simulate] {out['n_anchors']} anchors, {out['n_alert']} alerts")
    return out


# -----------------------------------------------------------------------------
def step_report(verbose=True):
    bpath = os.path.join(DIRS["benchmarks"], "benchmark.json")
    bench = json.load(open(bpath)) if os.path.exists(bpath) else {"rows": []}
    prof_p = os.path.join(DIRS["data_profile"], "profile.json")
    prof = json.load(open(prof_p)) if os.path.exists(prof_p) else {}
    rows = bench.get("rows", [])
    rows_sorted = sorted(rows, key=lambda r: -(r.get("pr_auc") or 0))

    def line(r):
        pa = r.get("progression_acc")
        pa_s = "-" if pa is None else f"{pa:.3f}"
        fb = r.get("f1_best"); fb_s = "-" if fb is None else f"{fb:.3f}"
        pr = r.get("pr_auc"); pr_s = "-" if pr is None else f"{pr:.3f}"
        return (f"| {r['model']} | {r.get('family','')} | {pr_s} | "
                f"{r.get('f1',0):.3f} | {fb_s} | {r.get('auroc',0):.3f} | "
                f"{r.get('mean_lead_time_s',0):.0f} | "
                f"{r.get('detection_rate',0):.2f} | {pa_s} |")

    md = [
        "# SENTINEL-WM - Research Report", "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_  |  "
        f"config: [`run_config.json`](../run_config.json)", "",
        "## 1. Data", "",
        f"- Flows: **{prof.get('n_flows','?'):,}**  |  10-s state windows: "
        f"**{prof.get('n_windows','?'):,}**  |  sequences (L={prof.get('L')}, "
        f"K={prof.get('K')}, F={prof.get('n_features')}): **{prof.get('n_sequences','?'):,}**",
        "- Label distribution: [`data_profile/flow_label_distribution.csv`]"
        "(../data_profile/flow_label_distribution.csv)  -  "
        "[timeline](../figures/attack_timeline_by_day.png)",
        f"- Split mode = **{C.CONFIG.split.mode}**"
        + (" (auto -> stratified)" if C.CONFIG.split.mode == "auto" else "")
        + ", leakage-safe (contiguous chunks, scaler fit on real-train only): "
        "[`data_profile/split_summary.csv`](../data_profile/split_summary.csv)  -  "
        "per-family coverage: [`data_profile/split_family_windows.csv`]"
        "(../data_profile/split_family_windows.csv)", "",
    ]
    for s in prof.get("split_summary", []):
        md.append(f"  - **{s['split']}**: {s['windows']} windows, "
                  f"{s['attack_frac']:.1%} attack")
    md += [
        "", "## 2. Model benchmark", "",
        "Full table + per-horizon F1: [`benchmarks/benchmark.md`]"
        "(../benchmarks/benchmark.md)  -  raw: "
        "[`benchmark_full.csv`](../benchmarks/benchmark_full.csv)  -  "
        "figures: [horizon F1](../figures/horizon_f1.png), "
        "[ROC](../figures/roc.png), [PR](../figures/pr.png), "
        "[lead time](../figures/lead_time.png)", "",
        "Ranked by **PR-AUC** (threshold-free; robust to the val->test attack-"
        "prevalence shift). `F1` = at the FPR<=5% threshold; `F1*` = best "
        "achievable by sweeping it.", "",
        "| Model | Family | PR-AUC | F1 | F1* | AUROC | MLT (s) | Detect | ProgAcc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    md += [line(r) for r in rows_sorted]

    # per-horizon F1 for the head of the field
    K = prof.get("K", 6)
    W = C.CONFIG.window.window_seconds
    md += ["", "### Forecast-horizon F1 (does it hold up as the horizon grows?)", "",
           "| Model | " + " | ".join(f"+{(k+1)*W}s" for k in range(K)) + " |",
           "|" + "---|" * (K + 1)]
    for r in rows_sorted[:8]:
        md.append("| " + r["model"] + " | " +
                  " | ".join(f"{r.get(f'f1_k{k+1}', float('nan')):.3f}" for k in range(K)) + " |")

    # ---- 2b. per-attack-family detection --------------------------------
    pf = bench.get("per_family", [])
    if pf:
        ref_models = {r["model"] for r in rows
                      if r.get("family") in ("reference",)}   # persistence
        best = {}                         # family -> (model, f1_best, recall, n_pos)
        for r in pf:
            if r.get("skipped") in (True, "True") or r["family"] == "BENIGN":
                continue
            if r["model"] in ref_models:
                continue
            f, f1 = r["family"], float(r["f1_best"] or r["f1"] or 0)
            if f not in best or f1 > best[f][1]:
                best[f] = (r["model"], f1, float(r["recall"] or 0), r["positives"])
        skipped = sorted({r["family"] for r in pf
                          if r.get("skipped") in (True, "True")})
        md += ["", "### 2b. Per-attack-family detection (best model per family)", "",
               "| Family | best model | F1 | recall | test +windows |",
               "|---|---|---|---|---|"]
        for f in sorted(best, key=lambda k: -best[k][1]):
            m, f1, rec, n = best[f]
            md.append(f"| {f} | {m} | {f1:.3f} | {rec:.2f} | {n} |")
        if skipped:
            md.append("")
            md.append(f"_Insufficient test data (< 8 positive windows), excluded: "
                      f"{', '.join(skipped)}._")
        md.append("")
        md.append("Full matrix: [`benchmarks/per_family.csv`](../benchmarks/per_family.csv).")

    # ---- 2c. zero-shot generalisation (secondary benchmark) -------------
    zdir = os.path.join(C.ROOT, C.ZEROSHOT_DIR_NAME)
    zpath = os.path.join(zdir, "benchmarks", "benchmark.json")
    if os.path.exists(zpath) and os.path.abspath(R) != os.path.abspath(zdir):
        zb = json.load(open(zpath))
        zrows = sorted(zb.get("rows", []), key=lambda r: -(r.get("pr_auc") or 0))
        zmode = zb.get("split_mode") or "family/day"
        md += ["", "## 3. Zero-shot generalisation (novel-family holdout)", "",
               f"Secondary benchmark - `split mode = {zmode}`. Whole attack "
               "families are held out of training and appear only at test time, so "
               "these numbers measure detection of **previously unseen** attacks, "
               "not the headline. F1 in the 0.3-0.5 band is expected and is the "
               "point of the experiment. Regenerate with "
               f"`python -m sentinel_wm.research all --split family --outdir {C.ZEROSHOT_DIR_NAME}`.",
               "", "| Model | Family | PR-AUC | F1 | F1* | AUROC |",
               "|---|---|---|---|---|---|"]
        for r in zrows[:12]:
            fb = r.get("f1_best"); fb_s = "-" if fb is None else f"{fb:.3f}"
            pr = r.get("pr_auc"); pr_s = "-" if pr is None else f"{pr:.3f}"
            md.append(f"| {r['model']} | {r.get('family','')} | {pr_s} | "
                      f"{r.get('f1',0):.3f} | {fb_s} | {r.get('auroc',0):.3f} |")
        md.append("")
        md.append(f"Full table: [`../../{C.ZEROSHOT_DIR_NAME}/benchmarks/benchmark.md`]"
                  f"(../../{C.ZEROSHOT_DIR_NAME}/benchmarks/benchmark.md).")

    wm = next((r for r in rows if r["model"] == "SENTINEL-WM"), None)
    top = next((r for r in rows_sorted
                if r.get("family") not in ("reference", "ensemble")), rows_sorted[0])
    ens = next((r for r in rows if str(r["model"]).startswith("ensemble")), None)
    md += ["", "**Reading it.** " + (
        f"Ranked by PR-AUC the strongest single model is **{top['model']}** "
        f"(PR-AUC {top.get('pr_auc',0):.3f}, AUROC {top.get('auroc',0):.3f}). " +
        (f"The **{ens['model'].split('(')[0]}** of the decorrelated top models "
         f"reaches PR-AUC {ens.get('pr_auc',0):.3f} / F1* {ens.get('f1_best',0):.3f}. "
         if ens else "") +
        (f"**SENTINEL-WM** (PR-AUC {wm.get('pr_auc',0):.3f}, AUROC {wm['auroc']:.3f}, "
         f"progression accuracy {wm['progression_acc']:.3f}, detection "
         f"{wm['detection_rate']:.2f}) is the only model that also carries a "
         f"progression-state head and a calibrated K-step Monte-Carlo rollout "
         f"with ATT&CK phase + confidence. " if wm else "") +
        f"The `F1` column lags `F1*` because the FPR<=5% threshold is fitted on "
        f"validation (higher attack prevalence) and transfers imperfectly to the "
        f"test day; PR-AUC / AUROC / F1* are the fair headline numbers. Mean Lead "
        f"Time stays low under the block split (attacks land mid-episode); use "
        f"`SplitConfig.mode='episode_chrono'` for a lead-time-focused run.")]

    md += [
        "", "## 4. Explainability", "",
        "SHAP + attention + gradient attribution: "
        "[`explainability/explainability.json`](../explainability/explainability.json)  -  "
        "[world-model SHAP](../figures/shap_world_model.png)", "",
        "## 5. Forward simulation (infiltration prediction engine)", "",
        "Per-anchor K-step Monte-Carlo rollouts with ATT&CK phase + confidence: "
        "[`simulations/forward_sim_test.json`](../simulations/forward_sim_test.json)  -  "
        "[`attck_stage_forecast.csv`](../simulations/attck_stage_forecast.csv)  -  "
        "[timelines](../figures/forward_sim_timelines.png)", "",
        "## 6. Saved models (for the serving app)", "",
        "Uniform loader: `from sentinel_wm.registry import load_predictor`  -  "
        "index: [`models/registry.json`](../models/registry.json)", "",
        "## 7. Reproduce", "",
        "```bash\npython -m sentinel_wm.research all\n```", "",
    ]
    io.open(os.path.join(DIRS["reports"], "RESEARCH_REPORT.md"), "w",
            encoding="utf-8").write("\n".join(md))

    # generated source-of-truth for numbers that docs/technical_reference.md hard-codes
    add = [
        "# technical_reference.md — live-numbers addendum", "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `sentinel_wm.research`._",
        "", "| key | value |", "|---|---|",
        f"| split mode | {C.CONFIG.split.mode} |",
        f"| features F | {prof.get('n_features')} |",
        f"| history L | {prof.get('L')} |",
        f"| horizon K | {prof.get('K')} |",
        f"| d_model | {C.CONFIG.model.d_model} |",
        f"| encoder | {getattr(C.CONFIG.model, 'encoder', 'gru')} |",
        f"| epochs | {C.CONFIG.train.epochs} |",
        f"| flows | {prof.get('n_flows')} |",
        f"| state windows | {prof.get('n_windows')} |",
        f"| sequences | {prof.get('n_sequences')} |",
    ]
    wmm = os.path.join(C.REPORT_DIR, "world_model_metrics.json")
    if os.path.exists(wmm):
        d = json.load(open(wmm))
        add.append(f"| world-model params | {d.get('params')} |")
        add.append(f"| world-model epochs run | {d.get('epochs_run')} |")
    for s in prof.get("split_summary", []):
        add.append(f"| split {s['split']} | {s['windows']} windows, "
                   f"{s['attack_frac']:.1%} attack |")
    io.open(os.path.join(DIRS["reports"], "technical_reference_addendum.md"), "w",
            encoding="utf-8").write("\n".join(add) + "\n")

    readme = [
        "# research/", "",
        "Everything the SENTINEL-WM study produced, each number backed by a file.",
        "Regenerate with `python -m sentinel_wm.research all`.", "",
        "| folder | contents |", "|---|---|",
        "| `data_profile/` | label & timeline stats, split balance |",
        "| `models/` | every trained model (`classical/*.pkl`, `nn/*.pt`, "
        "`world_model.pt`) + `registry.json` |",
        "| `benchmarks/` | `benchmark_full.csv/.md/.json`, `per_horizon_f1.csv`, "
        "`leadtime.csv` |",
        "| `figures/` | all plots (horizon F1, ROC/PR, lead time, SHAP, timelines) |",
        "| `explainability/` | SHAP / attention / gradient attribution JSON |",
        "| `simulations/` | K-step forward-simulation runs + ATT&CK stage forecasts |",
        "| `reports/` | `RESEARCH_REPORT.md` |",
        "| `logs/` | training curves per model |", "",
        "Start at [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md).", "",
    ]
    open(os.path.join(R, "README.md"), "w").write("\n".join(readme))
    if verbose:
        print(f"[report] -> {os.path.join(DIRS['reports'], 'RESEARCH_REPORT.md')}")


# -----------------------------------------------------------------------------
def _save_run_config(args):
    cfg = C.CONFIG
    json.dump(dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        raw_csvs=[os.path.basename(p) for p in C.RAW_FLOW_CSVS],
        window=asdict(cfg.window), sequence=asdict(cfg.sequence),
        split=asdict(cfg.split), model=asdict(cfg.model), train=asdict(cfg.train),
        args=vars(args)),
        open(os.path.join(R, "run_config.json"), "w"), indent=2, default=str)


def run_all(args):
    t0 = time.time()
    _save_run_config(args)
    ew = 12 if args.quick else args.wm_epochs
    en = 8 if args.quick else args.nn_epochs
    eg = 6 if args.quick else args.gat_epochs
    skip = set(args.skip or [])

    if "data" not in skip:
        # rebuild the derived tensors so a config change (features / L / labels /
        # split) always propagates - cheap next to training.
        print("\n########## DATA (windows + sequences + graphs) ##########")
        from sentinel_wm.state_windows import build_state_windows
        from sentinel_wm.sequences import build_sequences
        from sentinel_wm.graph_windows import build_graph_windows
        if getattr(C.CONFIG.window, "flow_augment", False):
            from sentinel_wm import flow_augment
            flow_augment.build_augmented_flows(verbose=True)
        sw = build_state_windows(verbose=True)
        build_sequences(sw, verbose=True)
        build_graph_windows(verbose=True)
    if "profile" not in skip:
        print("\n########## PROFILE ##########"); step_profile()
    if "baselines" not in skip:
        print("\n########## BASELINES ##########"); step_baselines()
    if "worldmodel" not in skip:
        print("\n########## WORLD MODEL ##########"); step_worldmodel(ew, args.device)
    if "nn" not in skip:
        print("\n########## NEURAL ZOO ##########"); step_nn(en, args.device)
    if "gat" not in skip:
        print("\n########## GAT ##########"); step_gat(eg, args.device)
    if "benchmark" not in skip:
        print("\n########## BENCHMARK ##########"); step_benchmark(args.device)
    if "explain" not in skip:
        print("\n########## EXPLAIN ##########"); step_explain(args.device)
    if "simulate" not in skip:
        print("\n########## SIMULATE ##########"); step_simulate(args.device)
    print("\n########## REPORT ##########"); step_report()
    if "bundle" not in skip and R == os.path.join(C.ROOT, C.RUN_DIR_NAME):
        # only the PRIMARY run assembles the deploy bundle (not zero-shot)
        print("\n########## BUNDLE ##########")
        try:
            from sentinel_wm import bundle
            bundle.assemble_bundle(C.MODEL_DIR)
        except Exception as e:
            print(f"[bundle] skipped: {e}")
    print(f"\n[research] done in {(time.time()-t0)/60:.1f} min -> {R}")


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="SENTINEL-WM research run")
    p.add_argument("step", choices=["all", "flowaug", "profile", "baselines",
                                    "worldmodel", "nn", "gat", "benchmark",
                                    "explain", "simulate", "report"])
    p.add_argument("--quick", action="store_true", help="small epoch budgets")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--wm-epochs", type=int, default=40, dest="wm_epochs")
    p.add_argument("--nn-epochs", type=int, default=50, dest="nn_epochs")
    p.add_argument("--gat-epochs", type=int, default=50, dest="gat_epochs")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--split", default=None,
                   choices=["auto", "stratified", "block", "day", "family",
                            "episode_chrono", "chronological"],
                   help="override SplitConfig.mode for this run (retrains everything)")
    p.add_argument("--outdir", default=C.RUN_DIR_NAME,
                   help=f"output folder under repo root (default {C.RUN_DIR_NAME}; "
                        f"e.g. {C.ZEROSHOT_DIR_NAME} for the zero-shot holdout benchmark)")
    a = p.parse_args()

    if a.split:
        C.CONFIG.split.mode = a.split
    if a.outdir and a.outdir != C.RUN_DIR_NAME:
        _set_output_root(a.outdir)

    if a.step == "all":
        run_all(a)
    else:
        {"flowaug": step_flowaug,
         "profile": step_profile, "baselines": step_baselines,
         "worldmodel": lambda: step_worldmodel(12 if a.quick else a.wm_epochs, a.device),
         "nn": lambda: step_nn(8 if a.quick else a.nn_epochs, a.device),
         "gat": lambda: step_gat(6 if a.quick else a.gat_epochs, a.device),
         "benchmark": lambda: step_benchmark(a.device),
         "explain": lambda: step_explain(a.device),
         "simulate": lambda: step_simulate(a.device),
         "report": step_report}[a.step]()


if __name__ == "__main__":
    main()
