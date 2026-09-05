#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  models.py   (the world model - proposal 6.4 / 6.5)
# -----------------------------------------------------------------------------
# A learned model of network-state transition dynamics, NOT a static classifier.
#
#   encoder   : [S_{t-L+1..t}] (+ real elapsed dt)  --Temporal Transformer-->  z_t
#   STN       : z_t  --probabilistic MLP-->  mu_{t+1}, logvar_{t+1}
#               z_{t+1} ~ N(mu, sigma)                (reparameterised sample)
#   attack    : z  -->  P(A = 1)                      (binary, per horizon)
#   progress  : z  -->  P(Z = c) over 5 states        (per horizon)
#
# Two prediction paths share the heads:
#   TRAIN  - direct multi-horizon heads on z_t give stable gradients; the STN is
#            trained to reproduce next-step latents (z_{t+1}_target = encoder of
#            the shifted window, detached).
#   INFER  - `rollout()` runs the STN autoregressively for K steps and applies
#            the heads to each simulated latent -> the "forward simulation".
#
# Attention weights from the final transformer block are retained for the
# explainability layer (proposal 6.8).
# =============================================================================
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel_wm import config as C


# -----------------------------------------------------------------------------
# elapsed-time positional encoding  (proposal 6.4: real time, not index)
# -----------------------------------------------------------------------------
class ElapsedTimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.idx = nn.Parameter(self._sinusoid(512, d_model), requires_grad=False)

    @staticmethod
    def _sinusoid(n, d):
        pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # x  [B, L, D] ;  dt [B, L]  (already log1p-compressed in sequences.py)
        B, L, D = x.shape
        te = self.mlp(dt.unsqueeze(-1))                # [B, L, D]
        pe = self.idx[:L].unsqueeze(0).to(x.dtype)     # [1, L, D]
        return x + te + pe


# -----------------------------------------------------------------------------
# a transformer block that exposes its attention weights
# -----------------------------------------------------------------------------
class AttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, attn_mask=attn_mask,
                         need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()                    # [B, L, L]
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# -----------------------------------------------------------------------------
# temporal encoder
# -----------------------------------------------------------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, n_features: int, m: C.ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(n_features, m.d_model)
        self.pos = ElapsedTimePositionalEncoding(m.d_model)
        self.blocks = nn.ModuleList([
            AttnBlock(m.d_model, m.n_heads, m.ff_mult, m.dropout)
            for _ in range(m.n_layers)])
        self.ln = nn.LayerNorm(m.d_model)

    def forward(self, x, dt) -> Dict[str, torch.Tensor]:
        B, L, _ = x.shape
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.pos(self.in_proj(x), dt)
        for blk in self.blocks:
            h = blk(h, attn_mask=causal)
        h = self.ln(h)
        return dict(seq=h, z=h[:, -1, :], attn=self.blocks[-1].last_attn)


# -----------------------------------------------------------------------------
# probabilistic State-Transition Network
# -----------------------------------------------------------------------------
class StateTransitionNet(nn.Module):
    def __init__(self, m: C.ModelConfig):
        super().__init__()
        d = m.d_model
        self.body = nn.Sequential(
            nn.Linear(d, m.stn_hidden), nn.LayerNorm(m.stn_hidden), nn.ReLU(),
            nn.Linear(m.stn_hidden, m.stn_hidden), nn.LayerNorm(m.stn_hidden), nn.ReLU())
        self.mu = nn.Linear(m.stn_hidden, d)
        self.logvar = nn.Linear(m.stn_hidden, d)

    def forward(self, z):
        h = self.body(z)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 4.0)
        return mu, logvar

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


# -----------------------------------------------------------------------------
# full model
# -----------------------------------------------------------------------------
class SentinelWorldModel(nn.Module):
    def __init__(self, n_features: int, n_states: int = 5,
                 horizon: int = 6, m: Optional[C.ModelConfig] = None):
        super().__init__()
        self.m = m or C.CONFIG.model
        self.n_features = n_features
        self.n_states = n_states
        self.horizon = horizon

        self.encoder = TemporalEncoder(n_features, self.m)
        self.stn = StateTransitionNet(self.m)

        d = self.m.d_model
        self.attack_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(self.m.dropout),
            nn.Linear(d // 2, 1))
        self.prog_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(self.m.dropout),
            nn.Linear(d // 2, n_states))
        # direct multi-horizon projection from z_t (training-time stability)
        self.horizon_attack = nn.Linear(d, horizon)
        self.horizon_prog = nn.Linear(d, horizon * n_states)

    # -- shared heads on an arbitrary latent -------------------------------
    def apply_heads(self, z):
        return dict(attack_logit=self.attack_head(z).squeeze(-1),
                    prog_logits=self.prog_head(z))

    # -- training forward -------------------------------------------------
    def forward(self, x, dt) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x, dt)
        z = enc["z"]
        mu, logvar = self.stn(z)
        z_next = self.stn.sample(mu, logvar)

        B = x.shape[0]
        out = dict(
            z=z, z_next_mu=mu, z_next_logvar=logvar, z_next=z_next,
            attn=enc["attn"],
            # direct multi-horizon predictions  [B, K] / [B, K, S]
            attack_logits_k=self.horizon_attack(z),
            prog_logits_k=self.horizon_prog(z).view(B, self.horizon, self.n_states),
        )
        # shared heads on the CURRENT latent (nowcast) and on the STN-simulated
        # next latent (world-model 1-step path). Both are supervised so that
        # `rollout()` - which calls apply_heads on every simulated latent - is
        # actually trained, not just the direct multi-horizon projection.
        h0 = self.apply_heads(z)
        h1 = self.apply_heads(z_next)
        out["attack_logit_now"] = h0["attack_logit"]
        out["prog_logits_now"] = h0["prog_logits"]
        out["attack_logit_sim1"] = h1["attack_logit"]
        out["prog_logits_sim1"] = h1["prog_logits"]
        return out

    # -- inference: K-step Monte-Carlo forward simulation (proposal 6.6) --
    @torch.no_grad()
    def rollout(self, x, dt, K: Optional[int] = None, M: int = 50
               ) -> Dict[str, np.ndarray]:
        self.eval()
        K = K or self.horizon
        enc = self.encoder(x, dt)
        z0 = enc["z"]                                   # [B, D]
        B, D = z0.shape
        z = z0.unsqueeze(1).expand(B, M, D).reshape(B * M, D).clone()

        atk = np.zeros((B, K, M), np.float32)
        prg = np.zeros((B, K, M, self.n_states), np.float32)
        for k in range(K):
            mu, logvar = self.stn(z)
            z = self.stn.sample(mu, logvar)
            h = self.apply_heads(z)
            atk[:, k, :] = torch.sigmoid(h["attack_logit"]).reshape(B, M).cpu().numpy()
            prg[:, k, :, :] = F.softmax(h["prog_logits"], -1).reshape(
                B, M, self.n_states).cpu().numpy()

        return dict(
            attack_prob=atk.mean(-1),                           # [B, K]
            attack_ci_lo=np.percentile(atk, 2.5, axis=-1),
            attack_ci_hi=np.percentile(atk, 97.5, axis=-1),
            attack_std=atk.std(-1),
            prog_prob=prg.mean(-2),                             # [B, K, S]
            prog_state=prg.mean(-2).argmax(-1),                 # [B, K]
            z0=z0.cpu().numpy(),
            attn=enc["attn"].cpu().numpy() if enc["attn"] is not None else None,
        )

    # -- convenience ------------------------------------------------------
    def config_dict(self) -> dict:
        return dict(n_features=self.n_features, n_states=self.n_states,
                   horizon=self.horizon, model=asdict(self.m))


def build_model(n_features: int, cfg: Optional[C.Config] = None
                ) -> SentinelWorldModel:
    cfg = cfg or C.CONFIG
    return SentinelWorldModel(n_features, n_states=len(C.PROGRESSION_STATES),
                              horizon=cfg.sequence.horizon, m=cfg.model)


# -----------------------------------------------------------------------------
# joint loss  (proposal 6.5)
# -----------------------------------------------------------------------------
def joint_loss(out: Dict[str, torch.Tensor],
               batch: Dict[str, torch.Tensor],
               z_next_target: torch.Tensor,
               m: C.ModelConfig,
               attack_pos_weight: Optional[torch.Tensor] = None,
               prog_class_weight: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
    y_atk = batch["y_atk"].float()                       # [B, K]
    y_prog = batch["y_prog"].long()                      # [B, K]

    l_attack = F.binary_cross_entropy_with_logits(
        out["attack_logits_k"], y_atk, pos_weight=attack_pos_weight)

    B, K, S = out["prog_logits_k"].shape
    l_prog = F.cross_entropy(
        out["prog_logits_k"].reshape(B * K, S), y_prog.reshape(B * K),
        weight=prog_class_weight)

    # supervise the shared heads (used by rollout) on the 1-step target
    y1, yp1 = y_atk[:, 0], y_prog[:, 0]
    l_attack = l_attack + 0.5 * (
        F.binary_cross_entropy_with_logits(out["attack_logit_now"], y1,
                                           pos_weight=attack_pos_weight)
        + F.binary_cross_entropy_with_logits(out["attack_logit_sim1"], y1,
                                             pos_weight=attack_pos_weight))
    l_prog = l_prog + 0.5 * (
        F.cross_entropy(out["prog_logits_now"], yp1, weight=prog_class_weight)
        + F.cross_entropy(out["prog_logits_sim1"], yp1, weight=prog_class_weight))

    # STN reproduces the next-step latent
    l_next = F.mse_loss(out["z_next_mu"], z_next_target)

    # light KL(N(mu,sigma) || N(0,1)) - keeps sigma meaningful
    mu, lv = out["z_next_mu"], out["z_next_logvar"]
    l_kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean()

    total = (m.w_attack * l_attack + m.w_progression * l_prog
             + m.w_next_state * l_next + m.w_kl * l_kl)
    return dict(total=total, attack=l_attack.detach(), prog=l_prog.detach(),
               next_state=l_next.detach(), kl=l_kl.detach())


if __name__ == "__main__":
    # smoke test on random tensors
    md = build_model(41)
    x = torch.randn(8, 10, 41)
    dt = torch.rand(8, 10)
    o = md(x, dt)
    print("forward keys:", list(o.keys()))
    print("attack_logits_k", o["attack_logits_k"].shape,
          "prog_logits_k", o["prog_logits_k"].shape,
          "attn", None if o["attn"] is None else tuple(o["attn"].shape))
    r = md.rollout(x, dt, K=6, M=16)
    print("rollout attack_prob", r["attack_prob"].shape,
          "prog_state", r["prog_state"].shape)
    print("params:", sum(p.numel() for p in md.parameters()))
