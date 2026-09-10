#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  evaluate.py   -  thin back-compat shim
# -----------------------------------------------------------------------------
# The full model scoreboard now lives in `sentinel_wm.benchmark` (it discovers
# every trained model in research/models/ and scores them on the same test
# anchors). This module keeps the old `evaluate.benchmark()` entry point working
# for cli.py and any existing notebooks / scripts.
# =============================================================================
from __future__ import annotations

from sentinel_wm.benchmark import run_benchmark


def benchmark(verbose: bool = True):
    return run_benchmark(verbose=verbose)


if __name__ == "__main__":
    benchmark()
