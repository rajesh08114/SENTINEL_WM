#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  cli.py   -  the offline command-line interface
# -----------------------------------------------------------------------------
# No Flask, no Streamlit. Everything runs locally from the terminal.
#
#   python cli.py preprocess                 # PHASE 1  raw CSV -> clean_flows
#   python cli.py windows                    # PHASE 2  clean_flows -> state windows
#   python cli.py sequences                  # PHASE 2c state windows -> sequences.npz
#   python cli.py baseline                   # PHASE 3  LR / RF comparison floor
#   python cli.py train  [--epochs N ...]    # PHASE 4  world model
#   python cli.py simulate [--split test]    # PHASE 5  K-step forward simulation
#   python cli.py explain                    # PHASE 6  SHAP + attention + saliency
#   python cli.py evaluate                   # side-by-side benchmark table
#   python cli.py all                        # phases 1-6 in order
#   python cli.py demo [--csv FILE]          # end-to-end on a CSV, prints timeline
# =============================================================================
from __future__ import annotations

import argparse
import json
import os

from sentinel_wm import config as C


def _p(msg): print(f"\n=== {msg} ===")


def cmd_preprocess(a):
    from sentinel_wm import preprocessing
    preprocessing.load_and_clean(verbose=True)


def cmd_windows(a):
    from sentinel_wm import state_windows
    state_windows.build_state_windows(verbose=True)


def cmd_sequences(a):
    from sentinel_wm import sequences
    sequences.build_sequences(verbose=True)


def cmd_baseline(a):
    from sentinel_wm import baselines
    baselines.run_baselines(verbose=True)


def cmd_train(a):
    from sentinel_wm import train as W
    cfg = C.CONFIG
    if a.epochs:
        cfg.train.epochs = a.epochs
    if a.batch_size:
        cfg.train.batch_size = a.batch_size
    if a.lr:
        cfg.train.lr = a.lr
    ns = argparse.Namespace(test=False, epochs=a.epochs, batch_size=a.batch_size,
                            lr=a.lr, device=a.device)
    W.train(cfg, ns)


def cmd_simulate(a):
    from sentinel_wm import forward_sim
    res = forward_sim.simulate_split(a.split, a.limit, a.device or "cpu",
                                     with_explain=a.explain)
    forward_sim.print_timeline(res)
    out = os.path.join(C.REPORT_DIR, f"forward_sim_{a.split}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[cli] -> {out}")


def cmd_explain(a):
    from sentinel_wm import explain
    explain.run_all(a.device or "cpu")


def cmd_evaluate(a):
    from sentinel_wm import evaluate
    evaluate.benchmark(verbose=True)


def cmd_all(a):
    for fn in (cmd_preprocess, cmd_windows, cmd_sequences, cmd_baseline):
        _p(fn.__name__); fn(a)
    _p("train"); cmd_train(a)
    _p("simulate"); cmd_simulate(a)
    _p("explain"); cmd_explain(a)
    _p("evaluate"); cmd_evaluate(a)


def cmd_demo(a):
    from sentinel_wm import preprocessing, state_windows, sequences, forward_sim, evaluate
    if a.csv:
        C.RAW_FLOW_CSVS[:] = [a.csv]
    _p("PHASE 1  preprocess"); preprocessing.load_and_clean(verbose=True)
    _p("PHASE 2  state windows"); state_windows.build_state_windows(verbose=True)
    _p("PHASE 2c sequences"); sequences.build_sequences(verbose=True)
    if not os.path.exists(C.WORLD_MODEL_PT):
        print("\n[demo] no trained checkpoint - run `python cli.py train` first.")
        return
    _p("PHASE 5  forward simulation (test split)")
    res = forward_sim.simulate_split("test", a.limit or 60, a.device or "cpu",
                                     with_explain=True)
    forward_sim.print_timeline(res)
    _p("BENCHMARK  world model vs baselines"); evaluate.benchmark(verbose=True)


def build_parser():
    p = argparse.ArgumentParser(prog="sentinel-wm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preprocess").set_defaults(func=cmd_preprocess)
    sub.add_parser("windows").set_defaults(func=cmd_windows)
    sub.add_parser("sequences").set_defaults(func=cmd_sequences)
    sub.add_parser("baseline").set_defaults(func=cmd_baseline)

    t = sub.add_parser("train"); t.set_defaults(func=cmd_train)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)

    s = sub.add_parser("simulate"); s.set_defaults(func=cmd_simulate)
    s.add_argument("--split", default="test", choices=["train", "val", "test"])
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--explain", action="store_true")

    sub.add_parser("explain").set_defaults(func=cmd_explain)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)

    al = sub.add_parser("all"); al.set_defaults(func=cmd_all)
    al.add_argument("--epochs", type=int, default=None)
    al.add_argument("--batch-size", type=int, default=None)
    al.add_argument("--lr", type=float, default=None)
    al.add_argument("--split", default="test")
    al.add_argument("--limit", type=int, default=None)
    al.add_argument("--explain", action="store_true", default=True)

    d = sub.add_parser("demo"); d.set_defaults(func=cmd_demo)
    d.add_argument("--csv", default=None, help="a unified_*_labeled.csv file")
    d.add_argument("--limit", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    # fill attributes other commands expect
    for attr in ("epochs", "batch_size", "lr", "split", "limit", "explain", "csv"):
        if not hasattr(args, attr):
            setattr(args, attr, None if attr != "explain" else False)
    args.func(args)


if __name__ == "__main__":
    main()
