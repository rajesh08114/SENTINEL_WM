#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  bundle.py   -  assemble a portable, self-contained model bundle
# -----------------------------------------------------------------------------
# The serving backend (../backend) must not depend on the research working tree
# (`runs/`, `artifacts/`). `sentinel-wm bundle <dst>` copies everything a
# predictor needs into one directory and rewrites the internal paths so it is
# relocatable:
#
#   <dst>/
#     world_model.pt              (+ snapshots rewritten to bundle-relative)
#     state_scaler.pkl
#     graph_windows.npz           (optional - GAT / system blend only)
#     graph_node_scaler.pkl       (optional)
#     world_model_snapshots/*.pt
#     models/
#       classical/*.pkl  *.meta.json
#       nn/*.pt  *.meta.json
#       registry.json              (paths relative to <dst>)
#     bundle.json                  manifest (git sha, timestamp, L/K/F, models)
#
# `config.bundled()` + `registry` + `forward_sim.load_checkpoint` prefer
# `SENTINEL_WM_MODEL_DIR` (this directory) when it is present.
# =============================================================================
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Optional

from sentinel_wm import config as C


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _copy(src: str, dst: str, verbose: bool) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    if verbose:
        print(f"  + {os.path.relpath(dst, os.path.dirname(dst.rstrip(os.sep)))}"
              if False else f"  + {os.path.basename(dst)}")
    return True


def assemble_bundle(dst: Optional[str] = None, verbose: bool = True) -> dict:
    import torch
    from sentinel_wm import train, graph_windows

    dst = os.path.abspath(dst or C.MODEL_DIR)
    os.makedirs(dst, exist_ok=True)
    if verbose:
        print(f"[bundle] -> {dst}")

    # ---- 1. world model + scaler + (optional) graph tensors --------------
    wm_src = C.WORLD_MODEL_PT
    if not os.path.exists(wm_src):
        raise FileNotFoundError(
            f"{wm_src} not found - run `python -m sentinel_wm.research all` "
            f"(or `sentinel-wm train`) first")
    _copy(wm_src, os.path.join(dst, "world_model.pt"), verbose)
    _copy(C.SCALER_PKL, os.path.join(dst, "state_scaler.pkl"), verbose)
    _copy(graph_windows.GRAPH_NPZ, os.path.join(dst, "graph_windows.npz"), verbose)
    _copy(graph_windows.GRAPH_SCALER, os.path.join(dst, "graph_node_scaler.pkl"), verbose)

    # ---- 2. snapshots + rewrite the checkpoint's snapshot list ----------
    snap_dst = os.path.join(dst, "world_model_snapshots")
    n_snap = 0
    if os.path.isdir(train.SNAP_DIR):
        for f in sorted(os.listdir(train.SNAP_DIR)):
            if f.endswith(".pt"):
                _copy(os.path.join(train.SNAP_DIR, f),
                      os.path.join(snap_dst, f), verbose=False)
                n_snap += 1
    ck = torch.load(os.path.join(dst, "world_model.pt"),
                    map_location="cpu", weights_only=False)
    ck["snapshots"] = [os.path.join("world_model_snapshots", f).replace("\\", "/")
                       for f in sorted(os.listdir(snap_dst))
                       if f.endswith(".pt")] if os.path.isdir(snap_dst) else []
    torch.save(ck, os.path.join(dst, "world_model.pt"))

    # ---- 3. classical + nn zoo -----------------------------------------
    src_models = os.path.join(C.research_dir(), "models")
    n_models = 0
    for sub in ("classical", "nn"):
        sd = os.path.join(src_models, sub)
        if not os.path.isdir(sd):
            continue
        for f in sorted(os.listdir(sd)):
            if f.endswith((".pkl", ".pt", ".meta.json")):
                _copy(os.path.join(sd, f),
                      os.path.join(dst, "models", sub, f), verbose=False)
                n_models += f.endswith((".pkl", ".pt"))

    # ---- 4. rebuild registry.json against the bundle ------------------
    from sentinel_wm import registry as _reg
    reg = _reg.build_registry(verbose=False, base_dir=dst)

    # ---- 5. manifest -------------------------------------------------
    manifest = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_sha=_git_sha(),
        L=int(ck["sequence"]["L"]), K=int(ck["sequence"]["K"]),
        n_features=int(ck["config"]["n_features"]),
        window_seconds=int(C.CONFIG.window.window_seconds),
        encoder=ck.get("config", {}).get("model", {}).get("encoder", "gru"),
        alert_threshold=ck.get("alert_threshold"),
        progression_states=list(C.PROGRESSION_STATES),
        snapshots=len(ck["snapshots"]),
        classical_models=n_models,
        registry=sorted(reg.keys()),
    )
    with open(os.path.join(dst, "bundle.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)

    if verbose:
        print(f"[bundle] {n_models} zoo models, {len(ck['snapshots'])} snapshots, "
              f"registry={len(reg)} -> {os.path.join(dst, 'bundle.json')}")
        print(f"[bundle] serve with:  SENTINEL_WM_MODEL_DIR={dst}")
    return manifest


if __name__ == "__main__":
    import sys
    assemble_bundle(sys.argv[1] if len(sys.argv) > 1 else None)
