#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  world_model_train.py   (PHASE 4 - temporal world model)
# -----------------------------------------------------------------------------
#   python world_model_train.py            # train, checkpoint, eval on test
#   python world_model_train.py --test     # load checkpoint, eval only
#   python world_model_train.py --epochs 60 --batch-size 128 --device cpu
#
# Produces:
#   artifacts/world_model.pt      weights + config + fitted alert threshold
#   artifacts/reports/world_model_metrics.json
#   artifacts/reports/training_curve.csv
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sentinel_wm import config as C
from sentinel_wm import metrics as M
from sentinel_wm.models import build_model, joint_loss
from sentinel_wm.sequences import load_sequences


# -----------------------------------------------------------------------------
def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _tensors(seq: Dict, split: str, device):
    m = seq["split"] == split
    def t(key, dtype):
        return torch.as_tensor(seq[key][m], dtype=dtype)
    return dict(
        X=t("X", torch.float32), dt=t("dt", torch.float32),
        x_next=t("x_next", torch.float32),
        y_atk=t("y_atk", torch.float32), y_prog=t("y_prog", torch.long),
        y_now=t("y_now", torch.long),
        window_index=seq["window_index"][m], day=seq["day"][m])


def _loader(d: Dict, batch_size: int, shuffle: bool):
    ds = TensorDataset(d["X"], d["dt"], d["x_next"], d["y_atk"], d["y_prog"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# -----------------------------------------------------------------------------
def evaluate_split(model, d: Dict, cfg: C.Config, device,
                   threshold: float = None) -> Dict:
    model.eval()
    K = cfg.sequence.horizon
    probs_k, prog_k = [], []
    with torch.no_grad():
        for i in range(0, len(d["X"]), 512):
            x = d["X"][i:i + 512].to(device)
            dt = d["dt"][i:i + 512].to(device)
            out = model(x, dt)
            probs_k.append(torch.sigmoid(out["attack_logits_k"]).cpu().numpy())
            prog_k.append(out["prog_logits_k"].argmax(-1).cpu().numpy())
    probs_k = np.concatenate(probs_k, 0)                 # [N, K]
    prog_k = np.concatenate(prog_k, 0)
    y_atk = d["y_atk"].numpy().astype(int)
    y_prog = d["y_prog"].numpy().astype(int)
    y_now = d["y_now"].numpy().astype(int)
    wi = d["window_index"]

    if threshold is None:
        threshold = M.calibrate_threshold(y_atk.max(1), probs_k.max(1),
                                          cfg.train.target_fpr)

    per_h = M.horizon_table(y_atk, probs_k, cfg.window.window_seconds, threshold)
    any_k = M.binary_scores(y_atk.max(1), probs_k.max(1), threshold)
    cal = M.expected_calibration_error(y_atk[:, 0], probs_k[:, 0])
    lt = M.lead_time(wi, y_now, probs_k.max(1),
                     cfg.window.window_seconds, threshold, K)
    prog_acc = float((prog_k == y_prog).mean())
    prog_acc_k1 = float((prog_k[:, 0] == y_prog[:, 0]).mean())

    return dict(threshold=float(threshold), any_horizon=any_k,
                per_horizon=per_h, brier_k1=M.brier_score(y_atk[:, 0], probs_k[:, 0]),
                ece_k1=cal["ece"], reliability=cal["bins"],
                progression_acc=prog_acc, progression_acc_k1=prog_acc_k1,
                lead_time=lt)


# -----------------------------------------------------------------------------
def train(cfg: C.Config, args) -> Dict:
    set_seed(cfg.train.seed)
    device = C.resolve_device(args.device or cfg.train.device)
    print(f"[train] device={device}")

    seq = load_sequences()
    n_feat = seq["X"].shape[-1]
    tr = _tensors(seq, "train", device)
    va = _tensors(seq, "val", device)
    te = _tensors(seq, "test", device)
    print(f"[train] features={n_feat} L={int(seq['L'])} K={int(seq['K'])} | "
          f"train={len(tr['X'])} val={len(va['X'])} test={len(te['X'])}")

    model = build_model(n_feat, cfg).to(device)

    # class weights from the training split
    pos = tr["y_atk"].mean().clamp(1e-3, 1 - 1e-3)
    attack_pos_weight = ((1 - pos) / pos).to(device)
    prog_counts = np.bincount(tr["y_prog"].numpy().ravel(),
                              minlength=len(C.PROGRESSION_STATES)).astype(float)
    prog_w = torch.as_tensor(
        (prog_counts.sum() / np.maximum(prog_counts, 1)) ** 0.5,
        dtype=torch.float32, device=device)
    prog_w = prog_w / prog_w.mean()
    print(f"[train] attack_pos_weight={attack_pos_weight.item():.2f} "
          f"prog_w={np.round(prog_w.cpu().numpy(), 2)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.epochs)
    loader = _loader(tr, cfg.train.batch_size, shuffle=True)

    curve, best_val, best_state, patience = [], -1.0, None, 0
    for ep in range(1, cfg.train.epochs + 1):
        model.train()
        agg = {}
        t0 = time.time()
        for xb, dtb, xnb, yak, ypk in loader:
            xb, dtb, xnb = xb.to(device), dtb.to(device), xnb.to(device)
            yak, ypk = yak.to(device), ypk.to(device)
            with torch.no_grad():
                # z_{t+1} target = encoder latent one window ahead.
                # Reuse x_next as the last row of a shifted history: cheap proxy
                # is encoding the history with its final row replaced by x_next.
                x_shift = torch.cat([xb[:, 1:, :], xnb.unsqueeze(1)], dim=1)
                z_next_target = model.encoder(x_shift, dtb)["z"]
            out = model(xb, dtb)
            losses = joint_loss(out, dict(y_atk=yak, y_prog=ypk),
                                z_next_target, cfg.model,
                                attack_pos_weight, prog_w)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v) * len(xb)
        sched.step()
        for k in agg:
            agg[k] /= len(tr["X"])

        val = evaluate_split(model, va, cfg, device)
        score = val["any_horizon"]["f1"] + 0.5 * val["progression_acc"]
        curve.append(dict(epoch=ep, **{f"loss_{k}": agg[k] for k in agg},
                          val_f1=val["any_horizon"]["f1"],
                          val_auroc=val["any_horizon"]["auroc"],
                          val_prog_acc=val["progression_acc"],
                          val_mlt=val["lead_time"]["mean_lead_time_s"]))
        print(f"  ep{ep:03d} {time.time()-t0:4.1f}s "
              f"L={agg['total']:.4f} (atk {agg['attack']:.3f} prog {agg['prog']:.3f} "
              f"next {agg['next_state']:.3f}) | "
              f"val F1={val['any_horizon']['f1']:.3f} AUROC={val['any_horizon']['auroc']:.3f} "
              f"progAcc={val['progression_acc']:.3f} MLT={val['lead_time']['mean_lead_time_s']:.0f}s")

        if score > best_val:
            best_val, best_state, patience = score, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[train] early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # freeze alert threshold on validation, then evaluate test
    val_final = evaluate_split(model, va, cfg, device)
    thr = val_final["threshold"]
    test_final = evaluate_split(model, te, cfg, device, threshold=thr)

    torch.save(dict(state_dict=model.state_dict(),
                    config=model.config_dict(),
                    feature_names=list(seq["feature_names"]),
                    alert_threshold=thr,
                    sequence=dict(L=int(seq["L"]), K=int(seq["K"]))),
               C.WORLD_MODEL_PT)

    with open(os.path.join(C.REPORT_DIR, "training_curve.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)

    report = dict(val=val_final, test=test_final, epochs_run=len(curve),
                  device=device, params=sum(p.numel() for p in model.parameters()))
    with open(os.path.join(C.REPORT_DIR, "world_model_metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    _print_report(test_final, cfg)
    print(f"[train] checkpoint -> {C.WORLD_MODEL_PT}")
    return report


# -----------------------------------------------------------------------------
def test_only(cfg: C.Config, args) -> Dict:
    device = C.resolve_device(args.device or cfg.train.device)
    ckpt = torch.load(C.WORLD_MODEL_PT, map_location=device, weights_only=False)
    seq = load_sequences()
    model = build_model(seq["X"].shape[-1], cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    te = _tensors(seq, "test", device)
    rep = evaluate_split(model, te, cfg, device, threshold=ckpt["alert_threshold"])
    _print_report(rep, cfg)
    with open(os.path.join(C.REPORT_DIR, "world_model_test_only.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def _print_report(rep: Dict, cfg: C.Config):
    print("\n================  WORLD MODEL - TEST  ================")
    print(M.summarise("any-horizon", rep["any_horizon"]))
    print(f"{'k':>3} {'horizon':>8} {'F1':>6} {'prec':>6} {'rec':>6} "
          f"{'FPR':>6} {'AUROC':>6} {'Brier':>6}")
    for r in rep["per_horizon"]:
        print(f"{r['k']:>3} {r['horizon_seconds']:>6}s  {r['f1']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['fpr']:>6.3f} "
              f"{r['auroc']:>6.3f} {r['brier']:>6.3f}")
    lt = rep["lead_time"]
    print(f"progression acc (all k) : {rep['progression_acc']:.3f}   "
          f"k=1 : {rep['progression_acc_k1']:.3f}")
    print(f"Mean Lead Time          : {lt['mean_lead_time_s']:.1f} s  "
          f"(median {lt['median_lead_time_s']:.0f}s, max {lt['max_lead_time_s']:.0f}s)")
    print(f"episodes warned         : {lt['n_episodes_warned']}/{lt['n_episodes']} "
          f"({lt['detection_rate']:.2f})   false-alarm rate {lt['false_alarm_rate']:.3f}")
    print(f"Brier(k1) {rep['brier_k1']:.3f}   ECE(k1) {rep['ece_k1']:.3f}   "
          f"alert threshold {rep['threshold']:.2f}")
    print("====================================================\n")


# -----------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Train the SENTINEL-WM world model")
    p.add_argument("--test", action="store_true", help="evaluate checkpoint only")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = C.CONFIG
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.test:
        test_only(cfg, args)
    else:
        train(cfg, args)


if __name__ == "__main__":
    main()
