#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  nn_zoo.py   -  neural sequence baselines
# -----------------------------------------------------------------------------
# Deep baselines that consume the SAME sequence tensor as the world model
# (X [B, L, F], dt [B, L]) and emit the SAME head outputs
# (attack_logits_k [B, K], prog_logits_k [B, K, S]) so that
# `sentinel_wm.train.evaluate_split` scores them unchanged.
#
# These are pure sequence->multi-horizon classifiers - NO state-transition
# network, NO Monte-Carlo rollout. They isolate "does temporal architecture X
# beat a flat model / the Temporal Transformer world model?".
#
#   mlp    - flatten the whole window, ignore order            (ANN baseline)
#   lstm   - 2-layer LSTM over the window                      (proposal baseline #3)
#   gru    - 2-layer GRU
#   tcn    - dilated temporal convolution stack
# =============================================================================
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from sentinel_wm import config as C


# -----------------------------------------------------------------------------
def _with_dt(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """append the (already log1p) elapsed-time channel -> [B, L, F+1]."""
    return torch.cat([x, dt.unsqueeze(-1)], dim=-1)


class MultiHorizonHead(nn.Module):
    def __init__(self, d: int, K: int, S: int, dropout: float = 0.1):
        super().__init__()
        self.K, self.S = K, S
        self.body = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.attack = nn.Linear(d, K)
        self.prog = nn.Linear(d, K * S)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.body(h)
        B = h.shape[0]
        return dict(attack_logits_k=self.attack(h),
                    prog_logits_k=self.prog(h).view(B, self.K, self.S))


# -----------------------------------------------------------------------------
class MLPForecaster(nn.Module):
    """Flatten [L,(F+1)] -> MLP. No notion of order beyond concatenation."""

    def __init__(self, n_features: int, L: int, K: int, S: int,
                 hidden=(512, 256, 128), dropout: float = 0.15):
        super().__init__()
        d_in = (n_features + 1) * L
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.head = MultiHorizonHead(d, K, S, dropout)

    def forward(self, x, dt) -> Dict[str, torch.Tensor]:
        z = self.net(_with_dt(x, dt).flatten(1))
        return self.head(z)


class RNNForecaster(nn.Module):
    def __init__(self, kind: str, n_features: int, K: int, S: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(n_features + 1, hidden, num_layers=layers,
                       batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = MultiHorizonHead(hidden, K, S, dropout)

    def forward(self, x, dt) -> Dict[str, torch.Tensor]:
        out, _ = self.rnn(_with_dt(x, dt))
        return self.head(self.norm(out[:, -1, :]))


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def _crop(self, y):
        return y[..., :-self.pad] if self.pad else y

    def forward(self, x):
        y = self.drop(self.act(self._crop(self.conv1(x))))
        y = self.drop(self.act(self._crop(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return self.act(y + res)


class TCNForecaster(nn.Module):
    """Causal dilated 1-D conv stack over the time axis."""

    def __init__(self, n_features: int, K: int, S: int,
                 channels=(96, 96, 96), kernel: int = 3, dropout: float = 0.15):
        super().__init__()
        blocks, c_in = [], n_features + 1
        for i, c_out in enumerate(channels):
            blocks.append(_TCNBlock(c_in, c_out, kernel, 2 ** i, dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(c_in)
        self.head = MultiHorizonHead(c_in, K, S, dropout)

    def forward(self, x, dt) -> Dict[str, torch.Tensor]:
        h = self.tcn(_with_dt(x, dt).transpose(1, 2))    # [B, C, L]
        return self.head(self.norm(h[:, :, -1]))


# -----------------------------------------------------------------------------
def build_nn_model(kind: str, n_features: int, L: int, cfg: C.Config = None
                   ) -> nn.Module:
    cfg = cfg or C.CONFIG
    K = cfg.sequence.horizon
    S = len(C.PROGRESSION_STATES)
    kind = kind.lower()
    if kind == "mlp":
        return MLPForecaster(n_features, L, K, S)
    if kind in ("lstm", "gru"):
        return RNNForecaster(kind, n_features, K, S)
    if kind == "tcn":
        return TCNForecaster(n_features, K, S)
    raise ValueError(f"unknown nn kind '{kind}' (mlp|lstm|gru|tcn)")


NN_KINDS = ["mlp", "lstm", "gru", "tcn"]
