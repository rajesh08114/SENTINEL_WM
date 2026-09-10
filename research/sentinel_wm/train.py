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


SNAP_DIR = os.path.join(C.ARTIFACTS, "world_model_snapshots")


def _teacher_probs(seq, cfg, tr) -> "torch.Tensor|None":
    """mean per-horizon P(attack) of the strong classical `__seq` teachers,
    on the TRAIN split, aligned to `tr` order."""
    import glob
    import pickle
    from sentinel_wm.baselines import _proba
    cdir = os.path.join(C.research_dir(), "models", "classical")
    m = seq["split"] == "train"
    Xseq = seq["X"][m].reshape(m.sum(), -1)              # flattened -> `__seq` input
    K = int(seq["K"])
    got = []
    for name in cfg.train.distill_teachers:
        p = os.path.join(cdir, f"{name}.pkl")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as fh:
                ests = pickle.load(fh)
            pr = np.stack([_proba(e, Xseq) for e in ests], 1)   # [N, K]
            got.append(pr)
        except Exception as e:
            print(f"[train] teacher {name} skipped: {e}")
    if not got:
        print("[train] no distillation teachers found - run `baseline` first")
        return None
    return torch.as_tensor(np.mean(got, 0).astype(np.float32))


# -----------------------------------------------------------------------------
def evaluate_split(model, d: Dict, cfg: C.Config, device,
                   threshold: float = None, snapshots=None,
                   self_ensemble: bool = False) -> Dict:
    from sentinel_wm.models import wm_predict
    model.eval()
    K = cfg.sequence.horizon
    probs_k, prog_k = wm_predict(model, d["X"].numpy(), d["dt"].numpy(), device,
                                 snapshots=snapshots, self_ensemble=self_ensemble)
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

    # ---- self-supervised encoder pre-training (masked-window reconstruction) --
    if getattr(cfg.train, "ssl_pretrain", False):
        from sentinel_wm.pretrain import pretrain_encoder, load_pretrained_into, OUT
        if not os.path.exists(OUT):
            pretrain_encoder(cfg, device, epochs=cfg.train.ssl_epochs, verbose=True)
        load_pretrained_into(model.encoder, verbose=True)

    # ---- distillation teacher: soft targets from the strong classical models --
    teacher_prob = None
    if getattr(cfg.train, "distill", False):
        teacher_prob = _teacher_probs(seq, cfg, tr)
        if teacher_prob is not None:
            teacher_prob = teacher_prob.to(device)
            print(f"[train] distillation ON  teachers={cfg.train.distill_teachers} "
                  f"w_distill={cfg.train.w_distill}")

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
    T0 = getattr(cfg.train, "warm_restart_period", 0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T0)
             if T0 and T0 > 0
             else torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.epochs))

    # balanced batch sampler: >= balanced_sampler_min_pos positive windows/batch
    from sentinel_wm.nn_common import _balanced_sampler
    from torch.utils.data import DataLoader, TensorDataset
    samp = _balanced_sampler(tr["y_atk"],
                             getattr(cfg.train, "balanced_sampler_min_pos", 0.0))
    K = int(seq["K"])
    tp_np = (teacher_prob.cpu().numpy() if teacher_prob is not None
             else np.zeros((len(tr["X"]), K), np.float32))
    if getattr(cfg.train, "augment", False):
        from sentinel_wm.augment import AugmentedSeqDataset
        ds = AugmentedSeqDataset(
            dict(X=tr["X"].numpy(), dt=tr["dt"].numpy(), x_next=tr["x_next"].numpy(),
                 y_atk=tr["y_atk"].numpy(), y_prog=tr["y_prog"].numpy(),
                 teacher=tp_np))
        print("[train] sequence augmentation ON")
    else:
        ds = TensorDataset(tr["X"], tr["dt"], tr["x_next"], tr["y_atk"],
                           tr["y_prog"], torch.as_tensor(tp_np))
    loader = DataLoader(ds, batch_size=cfg.train.batch_size,
                        sampler=samp, shuffle=samp is None)

    # two-stage: stage A pretrains encoder + attack head only (loss=w_attack*BCE)
    two_stage = getattr(cfg.train, "two_stage", False)
    stage_a_epochs = int(cfg.train.epochs * getattr(cfg.train, "two_stage_frac", 0.35)) \
        if two_stage else 0

    freeze_enc = two_stage and getattr(cfg.train, "two_stage_freeze_encoder", False)

    curve, best_val, best_state, patience = [], -1.0, None, 0
    best_a_state, best_a = None, -1.0
    for ep in range(1, cfg.train.epochs + 1):
        stage = "A" if ep <= stage_a_epochs else "full"
        if ep == stage_a_epochs + 1:
            # start stage B from stage A's BEST-val encoder+heads, not its last
            if best_a_state is not None:
                model.load_state_dict(best_a_state)
            if freeze_enc:
                for p in model.encoder.parameters():
                    p.requires_grad_(False)
                print("[train] stage B: encoder frozen at stage-A best; "
                      "training STN + progression + light attack-head fine-tune")
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.train.lr * 0.5, weight_decay=cfg.train.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, max(cfg.train.epochs - stage_a_epochs, 1))
        model.train()
        agg = {}
        t0 = time.time()
        kd_w = (cfg.train.w_distill * max(0.0, 1.0 - ep / cfg.train.epochs)
                if teacher_prob is not None else 0.0)     # decay KD over training
        for xb, dtb, xnb, yak, ypk, tpk in loader:
            xb, dtb, xnb = xb.to(device), dtb.to(device), xnb.to(device)
            yak, ypk, tpk = yak.to(device), ypk.to(device), tpk.to(device)
            with torch.no_grad():
                # z_{t+1} target = encoder latent one window ahead.
                x_shift = torch.cat([xb[:, 1:, :], xnb.unsqueeze(1)], dim=1)
                z_next_target = model.encoder(x_shift, dtb)["z"]
            out = model(xb, dtb)
            losses = joint_loss(out, dict(y_atk=yak, y_prog=ypk, teacher=tpk,
                                          kd_w=kd_w),
                                z_next_target, cfg.model,
                                attack_pos_weight, prog_w, stage=stage)
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
        curve.append(dict(epoch=ep, stage=stage,
                          **{f"loss_{k}": agg[k] for k in agg},
                          val_f1=val["any_horizon"]["f1"],
                          val_auroc=val["any_horizon"]["auroc"],
                          val_prog_acc=val["progression_acc"],
                          val_mlt=val["lead_time"]["mean_lead_time_s"]))
        print(f"  ep{ep:03d}[{stage}] {time.time()-t0:4.1f}s "
              f"L={agg['total']:.4f} (atk {agg['attack']:.3f} prog {agg['prog']:.3f} "
              f"next {agg['next_state']:.3f}) | "
              f"val F1={val['any_horizon']['f1']:.3f} AUROC={val['any_horizon']['auroc']:.3f} "
              f"progAcc={val['progression_acc']:.3f} MLT={val['lead_time']['mean_lead_time_s']:.0f}s")

        # stage A: track the best-val encoder+head (by attack score only), no patience
        if stage == "A":
            a_score = val["any_horizon"]["f1"] + 0.5 * val["any_horizon"]["auroc"]
            if a_score > best_a:
                best_a = a_score
                best_a_state = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
            continue
        if ep == stage_a_epochs + 1:
            best_val, patience = -1.0, 0        # reset at the A->full transition
        if score > best_val:
            best_val, best_state, patience = score, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[train] early stop at epoch {ep}")
                break

        # snapshot ensemble: save at each cosine warm-restart trough
        if getattr(cfg.train, "snapshot_ensemble", False) and T0 and ep > stage_a_epochs \
           and (ep - stage_a_epochs) % T0 == 0:
            os.makedirs(SNAP_DIR, exist_ok=True)
            sp = os.path.join(SNAP_DIR, f"snap_ep{ep:03d}.pt")
            torch.save({k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}, sp)
            print(f"[train] snapshot -> {sp}")

    if best_state is not None:
        model.load_state_dict(best_state)

    import glob
    snaps = sorted(glob.glob(os.path.join(SNAP_DIR, "snap_ep*.pt")))
    self_ens = bool(getattr(cfg.train, "self_ensemble", False))

    # freeze alert threshold on validation (using the SAME ensemble that the
    # benchmark will use), then evaluate test
    val_final = evaluate_split(model, va, cfg, device, snapshots=snaps,
                               self_ensemble=self_ens)
    thr = val_final["threshold"]
    test_final = evaluate_split(model, te, cfg, device, threshold=thr,
                                snapshots=snaps, self_ensemble=self_ens)

    torch.save(dict(state_dict=model.state_dict(),
                    config=model.config_dict(),
                    feature_names=list(seq["feature_names"]),
                    alert_threshold=thr,
                    snapshots=[os.path.relpath(s, C.ROOT) for s in snaps],
                    self_ensemble=self_ens,
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
