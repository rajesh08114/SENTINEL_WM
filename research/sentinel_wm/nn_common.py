#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  nn_common.py   -  shared training / eval loop for deep models
# -----------------------------------------------------------------------------
# One training loop reused by every neural baseline in `nn_zoo.py` and by the
# GAT in `gat.py`. Anything whose `forward(x, dt)` (or `forward(**batch)`)
# returns {attack_logits_k [B,K], prog_logits_k [B,K,S]} can be trained and
# scored here with the identical protocol used for the world model:
#   * class-weighted BCE(attack) + 0.5 * CE(progression)
#   * AdamW + cosine schedule, grad clip, early stop on (val F1 + 0.5 prog_acc)
#   * alert threshold FPR-calibrated on validation, frozen for the test report
#   * checkpoint saved to research/models/nn/<name>.pt in a load-ready format
# =============================================================================
from __future__ import annotations

import csv
import json
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sentinel_wm import config as C
from sentinel_wm import metrics as M
from sentinel_wm.sequences import load_sequences
from sentinel_wm.train import evaluate_split, _tensors, set_seed

def _nn_dir():
    return os.path.join(C.research_dir(), "models", "nn")


def _log_dir():
    return os.path.join(C.research_dir(), "logs")


NN_DIR = _nn_dir()          # back-compat; prefer _nn_dir() for a live value
LOG_DIR = _log_dir()


# -----------------------------------------------------------------------------
def _class_weights(y_atk: torch.Tensor, y_prog: torch.Tensor, device):
    pos = y_atk.float().mean().clamp(1e-3, 1 - 1e-3)
    apw = ((1 - pos) / pos).to(device)
    cnt = np.bincount(y_prog.numpy().ravel(),
                      minlength=len(C.PROGRESSION_STATES)).astype(float)
    pw = torch.as_tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5,
                         dtype=torch.float32, device=device)
    return apw, pw / pw.mean()


def _loss(out: Dict[str, torch.Tensor], y_atk, y_prog, apw, pw, gamma=0.0):
    from sentinel_wm.models import focal_bce
    la = focal_bce(out["attack_logits_k"], y_atk.float(), gamma, apw)
    B, K, S = out["prog_logits_k"].shape
    lp = F.cross_entropy(out["prog_logits_k"].reshape(B * K, S),
                         y_prog.reshape(B * K).long(), weight=pw)
    return la + 0.5 * lp, la.detach(), lp.detach()


def _balanced_sampler(y_atk: torch.Tensor, min_pos: float):
    """WeightedRandomSampler that lifts positive-window frequency to `min_pos`."""
    from torch.utils.data import WeightedRandomSampler
    pos = (y_atk.sum(1) > 0).float().numpy()          # window has any future attack
    p = pos.mean()
    if p <= 0 or p >= min_pos or min_pos <= 0:
        return None
    w_pos = min_pos / p
    w_neg = (1 - min_pos) / (1 - p)
    weights = pos * w_pos + (1 - pos) * w_neg
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# -----------------------------------------------------------------------------
def train_nn(model: torch.nn.Module, name: str, *,
             kind: str, family: str = "nn",
             cfg: C.Config = None, device: str = None,
             epochs: int = 60, batch_size: int = 256, lr: float = 8e-4,
             seq: Optional[Dict] = None,
             batch_forward: Optional[Callable] = None,
             extra_meta: Optional[Dict] = None,
             verbose: bool = True) -> Dict:
    """
    batch_forward(model, batch_tuple, device) -> (out_dict, y_atk, y_prog)
        override for models whose inputs are not just (X, dt) (e.g. GAT).
        Default expects a DataLoader over (X, dt, y_atk, y_prog).
    """
    cfg = cfg or C.CONFIG
    device = C.resolve_device(device or cfg.train.device)
    set_seed(cfg.train.seed)
    nn_dir = _nn_dir(); os.makedirs(nn_dir, exist_ok=True)
    run_log = os.path.join(_log_dir(), "nn")
    os.makedirs(run_log, exist_ok=True)

    seq = seq or load_sequences()
    tr = _tensors(seq, "train", device)
    va = _tensors(seq, "val", device)
    te = _tensors(seq, "test", device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    apw, pw = _class_weights(tr["y_atk"], tr["y_prog"], device)
    gamma = getattr(cfg.model, "focal_gamma", 0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg.train.weight_decay)
    T0 = getattr(cfg.train, "warm_restart_period", 0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T0)
             if T0 and T0 > 0
             else torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs))
    min_pos = getattr(cfg.train, "balanced_sampler_min_pos", 0.0)

    if batch_forward is None:
        samp = _balanced_sampler(tr["y_atk"], min_pos)
        if getattr(cfg.train, "augment", False):
            from sentinel_wm.augment import AugmentedSeqDataset
            ds = AugmentedSeqDataset(dict(
                X=tr["X"].numpy(), dt=tr["dt"].numpy(),
                y_atk=tr["y_atk"].numpy(), y_prog=tr["y_prog"].numpy()))
        else:
            ds = TensorDataset(tr["X"], tr["dt"], tr["y_atk"], tr["y_prog"])
        loader = DataLoader(ds, batch_size=batch_size,
                            sampler=samp, shuffle=samp is None)

        def _step(batch):
            xb, dtb, ya, yp = (t.to(device) for t in batch[:4])  # ignore teacher col
            return model(xb, dtb), ya, yp
    else:
        loader = batch_forward("loader", (tr, batch_size), device)

        def _step(batch):
            return batch_forward(model, batch, device)

    curve, best, best_state, patience = [], -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        agg = {"total": 0.0, "atk": 0.0, "prog": 0.0}
        t0 = time.time()
        n = 0
        for batch in loader:
            out, ya, yp = _step(batch)
            loss, la, lp = _loss(out, ya, yp, apw, pw, gamma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            bs = ya.shape[0]; n += bs
            agg["total"] += float(loss) * bs
            agg["atk"] += float(la) * bs
            agg["prog"] += float(lp) * bs
        sched.step()
        for k in agg:
            agg[k] /= max(n, 1)

        val = _eval(model, va, cfg, device, batch_forward, seq, "val")
        score = val["any_horizon"]["f1"] + 0.5 * val["progression_acc"]
        curve.append(dict(epoch=ep, **{f"loss_{k}": agg[k] for k in agg},
                          val_f1=val["any_horizon"]["f1"],
                          val_auroc=val["any_horizon"]["auroc"],
                          val_prog_acc=val["progression_acc"]))
        if verbose:
            print(f"  [{name}] ep{ep:03d} {time.time()-t0:4.1f}s "
                  f"L={agg['total']:.4f} valF1={val['any_horizon']['f1']:.3f} "
                  f"AUROC={val['any_horizon']['auroc']:.3f} "
                  f"prog={val['progression_acc']:.3f}")
        if score > best:
            best, best_state, patience = score, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                if verbose:
                    print(f"  [{name}] early stop @ {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    val_final = _eval(model, va, cfg, device, batch_forward, seq, "val")
    thr = val_final["threshold"]
    test_final = _eval(model, te, cfg, device, batch_forward, seq, "test", thr)

    ckpt = os.path.join(nn_dir, f"{name}.pt")
    torch.save(dict(state_dict=model.state_dict(), kind=kind, family=family,
                    feature_names=list(seq["feature_names"]),
                    alert_threshold=float(thr),
                    n_features=int(seq["X"].shape[-1]),
                    sequence=dict(L=int(seq["L"]), K=int(seq["K"])),
                    extra=extra_meta or {}), ckpt)
    with open(os.path.join(nn_dir, f"{name}.meta.json"), "w") as fh:
        json.dump(dict(name=name, family=family, kind=kind,
                       params=n_params, threshold=float(thr),
                       epochs_run=len(curve),
                       metrics={"f1": test_final["any_horizon"]["f1"],
                                "auroc": test_final["any_horizon"]["auroc"],
                                "mean_lead_time_s": test_final["lead_time"]["mean_lead_time_s"],
                                "progression_acc": test_final["progression_acc"]}),
                  fh, indent=2)
    with open(os.path.join(run_log, f"{name}_curve.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys())); w.writeheader()
        w.writerows(curve)

    report = dict(model=name, kind=kind, family=family, params=n_params,
                  epochs_run=len(curve), device=device,
                  val=val_final, test=test_final)
    if verbose:
        h = test_final["any_horizon"]
        print(f"[{name}] TEST F1={h['f1']:.3f} AUROC={h['auroc']:.3f} "
              f"MLT={test_final['lead_time']['mean_lead_time_s']:.0f}s "
              f"prog={test_final['progression_acc']:.3f} ({n_params/1e3:.0f}k params)")
    return report


# -----------------------------------------------------------------------------
def _eval(model, d, cfg, device, batch_forward, seq, split, thr=None):
    if batch_forward is None:
        return evaluate_split(model, d, cfg, device, threshold=thr)
    return batch_forward("evaluate", (model, d, cfg, device, seq, split, thr),
                         device)


# -----------------------------------------------------------------------------
def load_nn_checkpoint(path: str, device: str = "cpu"):
    from sentinel_wm.nn_zoo import build_nn_model
    ck = torch.load(path, map_location=device, weights_only=False)
    seq_l = ck["sequence"]["L"]
    model = build_nn_model(ck["kind"], ck["n_features"], seq_l)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def run_nn_zoo(kinds=None, cfg: C.Config = None, device: str = None,
               epochs: int = 60, verbose: bool = True) -> Dict:
    from sentinel_wm.nn_zoo import build_nn_model, NN_KINDS
    cfg = cfg or C.CONFIG
    seq = load_sequences()
    L, Fdim = int(seq["L"]), int(seq["X"].shape[-1])
    kinds = kinds or NN_KINDS
    reports = {}
    for kind in kinds:
        if verbose:
            print(f"\n=== NN baseline: {kind} ===")
        model = build_nn_model(kind, Fdim, L, cfg)
        reports[kind] = train_nn(model, kind, kind=kind, cfg=cfg, device=device,
                                 epochs=epochs, seq=seq, verbose=verbose)
    return reports


if __name__ == "__main__":
    run_nn_zoo()
