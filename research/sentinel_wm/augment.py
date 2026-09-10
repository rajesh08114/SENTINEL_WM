#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  augment.py   -  train-only sequence augmentation
# -----------------------------------------------------------------------------
# The bottleneck is not total volume (~14 k windows) but POSITIVE diversity:
# ~1 150 attack training windows spread over ~13 families, several with < 30
# (Heartbleed = 11 flows and Infiltration = 36 flows in the whole dataset are
# unrecoverable - no augmentation fixes those). These transforms enlarge and
# perturb the learnable positive families (DoS*, DDoS, PortScan, Bot, Web*).
#
# They operate on the ALREADY-SCALED sequences.npz tensors (X [N,L,F], dt [N,L])
# so they are cheap and model-agnostic. Applied only to the TRAIN split, on the
# fly in the DataLoader. Enable with `TrainConfig.augment = True`.
#
#   jitter        X += N(0, sigma * jitter_scale)                per feature
#   mixup         lambda*X_a + (1-lambda)*X_b   (a,b both positive)   label = OR
#   window_dropout replace r of the L history windows with the left neighbour
#   time_roll     small circular shift of the history (+- roll_max windows)
#
# Flow-level augmentation (port shuffle / IAT jitter / flow dropout / cross-day
# transplant) - the complementary, stronger lever that synthesises genuinely new
# positive windows - is implemented in `sentinel_wm/flow_augment.py` (opt-in
# `WindowConfig.flow_augment`). See docs/technical_reference.md Part 2 #8.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class AugConfig:
    p_jitter: float = 0.6
    jitter_scale: float = 0.10
    p_mixup: float = 0.4
    mixup_alpha: float = 0.4
    p_window_dropout: float = 0.3
    window_dropout_max: int = 2
    p_time_roll: float = 0.3
    roll_max: int = 1
    seed: int = 1337


class AugmentedSeqDataset(Dataset):
    """Wraps the train arrays; augments X (and dt) per __getitem__.

    arrays: dict with X [N,L,F] float32, dt [N,L] float32,
            y_atk [N,K] int, y_prog [N,K] int   (+ optionally x_next [N,F]).
    Extra keys are passed through unchanged so it drops into existing loops.
    """

    KEYS = ("X", "dt", "x_next", "y_atk", "y_prog", "teacher")

    def __init__(self, arrays: dict, cfg: Optional[AugConfig] = None):
        self.a = {k: np.asarray(arrays[k]) for k in self.KEYS if k in arrays}
        self.cfg = cfg or AugConfig()
        self.n = len(self.a["X"])
        self.F = self.a["X"].shape[-1]
        self.L = self.a["X"].shape[1]
        # per-feature std on the (scaled) train set -> jitter magnitude
        self.sigma = self.a["X"].reshape(-1, self.F).std(0).clip(1e-3, None)
        # indices of positive sequences (any future attack) for mixup partners
        self.pos_idx = np.where(self.a["y_atk"].max(1) > 0)[0]
        self.rng = np.random.default_rng(self.cfg.seed)

    def __len__(self):
        return self.n

    def _augment(self, x, dt, ya, yp, tp=None):
        c = self.cfg
        # --- mixup with another positive (only if this one is positive) -----
        if ya.max() > 0 and len(self.pos_idx) > 1 and self.rng.random() < c.p_mixup:
            b = int(self.rng.choice(self.pos_idx))
            lam = float(self.rng.beta(c.mixup_alpha, c.mixup_alpha))
            lam = max(lam, 1 - lam)                       # keep this sample dominant
            xb = self.a["X"][b]
            x = lam * x + (1 - lam) * xb
            ya = np.maximum(ya, self.a["y_atk"][b])       # OR the attack labels
            if tp is not None and "teacher" in self.a:
                tp = lam * tp + (1 - lam) * self.a["teacher"][b]
            if lam < 0.75:                                # progression: take the
                yp = np.where(ya > 0, np.maximum(yp, self.a["y_prog"][b]), yp)
        # --- feature jitter ----------------------------------------------- --
        if self.rng.random() < c.p_jitter:
            x = x + self.rng.normal(0, 1, x.shape).astype(np.float32) \
                * self.sigma * c.jitter_scale
        # --- window dropout (collection gap) ---------------------------------
        if self.L > 3 and self.rng.random() < c.p_window_dropout:
            r = int(self.rng.integers(1, c.window_dropout_max + 1))
            for _ in range(r):
                k = int(self.rng.integers(1, self.L))
                x[k] = x[k - 1]
                dt = dt.copy(); dt[k] = 0.0
        # --- small temporal roll of the history ---------------------------- -
        if c.roll_max and self.rng.random() < c.p_time_roll:
            s = int(self.rng.integers(-c.roll_max, c.roll_max + 1))
            if s:
                x = np.roll(x, s, axis=0)
                if s > 0:
                    x[:s] = x[s]
                else:
                    x[s:] = x[s - 1]
        return x.astype(np.float32), dt.astype(np.float32), ya, yp, tp

    def __getitem__(self, i):
        x = self.a["X"][i].copy()
        dt = self.a["dt"][i].copy()
        ya = self.a["y_atk"][i].copy()
        yp = self.a["y_prog"][i].copy()
        tp = self.a["teacher"][i].copy() if "teacher" in self.a else None
        x, dt, ya, yp, tp = self._augment(x, dt, ya, yp, tp)
        out = [torch.from_numpy(x), torch.from_numpy(dt)]
        if "x_next" in self.a:
            out.append(torch.from_numpy(self.a["x_next"][i]))
        out += [torch.from_numpy(ya), torch.from_numpy(yp)]
        out.append(torch.from_numpy(tp) if tp is not None
                   else torch.zeros(ya.shape[0], dtype=torch.float32))
        return tuple(out)


def summarise_positive_counts(sw, sp, w) -> str:
    """quick diagnostic: attack windows per (family, split)."""
    import pandas as pd
    from sentinel_wm.sequences import assign_split
    sw = sw.copy()
    sw["split"] = assign_split(sw, sp, w, verbose=False)
    a = sw[sw["y_attack"] == 1]
    piv = a.pivot_table(index="dominant_family", columns="split",
                        values="window_index", aggfunc="count", fill_value=0)
    return piv.to_string()
