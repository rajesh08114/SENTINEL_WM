#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  pretrain.py   -  self-supervised encoder pre-training
# -----------------------------------------------------------------------------
# The encoder never has enough LABELLED positives (~1.2k). But it has ~8.5k
# unlabelled TRAIN sequences of network state. Masked-window reconstruction
# teaches the encoder the manifold of "normal + attack" state dynamics before a
# single label is seen; `train.py` then loads these weights into stage A.
#
#   input   X [B, L, F]  (RobustScaler-normalised), dt [B, L]
#   mask    ~mask_ratio of the L windows replaced by a learned [MASK] vector
#   target  reconstruct the masked windows' feature vectors      (MSE)
#
# Output: artifacts/encoder_pretrained.pt  {state_dict, n_features, encoder_kind}
# =============================================================================
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sentinel_wm import config as C
from sentinel_wm.models import build_encoder
from sentinel_wm.sequences import load_sequences

OUT = os.path.join(C.ARTIFACTS, "encoder_pretrained.pt")


class _MaskedReconstructor(nn.Module):
    def __init__(self, n_features: int, m: C.ModelConfig):
        super().__init__()
        self.encoder = build_encoder(n_features, m)
        self.mask_vec = nn.Parameter(torch.zeros(n_features))
        self.head = nn.Sequential(
            nn.Linear(m.d_model, m.d_model), nn.GELU(),
            nn.Linear(m.d_model, n_features))

    def forward(self, x, dt, mask):                     # mask [B, L] bool
        xm = torch.where(mask.unsqueeze(-1), self.mask_vec.view(1, 1, -1), x)
        seq = self.encoder(xm, dt)["seq"]               # [B, L, d]
        return self.head(seq)


def pretrain_encoder(cfg: C.Config = None, device: str = None,
                     epochs: int = 40, mask_ratio: float = 0.25,
                     verbose: bool = True) -> str:
    cfg = cfg or C.CONFIG
    device = C.resolve_device(device or cfg.train.device)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    seq = load_sequences()
    tr = seq["split"] == "train"
    X = torch.as_tensor(seq["X"][tr], dtype=torch.float32)
    DT = torch.as_tensor(seq["dt"][tr], dtype=torch.float32)
    L = X.shape[1]
    n_feat = X.shape[-1]

    # per-feature std of the (already scaled) train windows -> normalise the
    # reconstruction target so every feature contributes comparably to the MSE
    fstd = X.reshape(-1, n_feat).std(0).clamp(1e-2, None).to(device)

    model = _MaskedReconstructor(n_feat, cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr * 3,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(TensorDataset(X, DT), batch_size=cfg.train.batch_size,
                        shuffle=True)

    if verbose:
        print(f"[pretrain] {len(X)} train seqs, L={L} F={n_feat}, "
              f"encoder={cfg.model.encoder}, mask_ratio={mask_ratio}")
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for xb, dtb in loader:
            xb, dtb = xb.to(device), dtb.to(device)
            # never mask the last (anchor) window - it defines the read-out
            m = torch.rand(xb.shape[0], L, device=device) < mask_ratio
            m[:, -1] = False
            if m.sum() == 0:
                m[:, 0] = True
            pred = model(xb, dtb, m)
            loss = F.mse_loss(pred[m] / fstd, xb[m] / fstd)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            tot += float(loss) * xb.shape[0]; n += xb.shape[0]
        sched.step()
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  ep{ep:03d} {time.time()-t0:4.1f}s  recon_mse={tot/n:.4f}")

    torch.save(dict(state_dict=model.encoder.state_dict(),
                    n_features=n_feat, encoder_kind=cfg.model.encoder), OUT)
    if verbose:
        print(f"[pretrain] encoder -> {OUT}")
    return OUT


def load_pretrained_into(encoder: nn.Module, path: str = OUT,
                         verbose: bool = True) -> bool:
    if not os.path.exists(path):
        return False
    ck = torch.load(path, map_location="cpu", weights_only=False)
    try:
        missing, unexpected = encoder.load_state_dict(ck["state_dict"], strict=False)
        if verbose:
            print(f"[pretrain] loaded encoder weights "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        return True
    except Exception as e:                               # pragma: no cover
        print(f"[pretrain] could not load: {e}")
        return False


if __name__ == "__main__":
    pretrain_encoder()
