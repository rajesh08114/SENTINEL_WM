"""The nets. Vendored from research/sentinel_wm/models.py (world model) +
nn_zoo.py (the TCN/LSTM/GRU blend members). Training-only code (joint_loss,
focal_bce, the __main__ smoke tests) is NOT vendored. Architectures must match
the checkpoints - keep in sync with research/sentinel_wm/{models,nn_zoo}.py.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import schema as C


# =============================================================================
# world model  (models.py)
# =============================================================================
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

    def forward(self, x, dt):
        B, L, D = x.shape
        te = self.mlp(dt.unsqueeze(-1))
        pe = self.idx[:L].unsqueeze(0).to(x.dtype)
        return x + te + pe


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
        self.last_attn = w.detach()
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, n_features: int, m: C.ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(n_features, m.d_model)
        self.pos = ElapsedTimePositionalEncoding(m.d_model)
        self.blocks = nn.ModuleList([
            AttnBlock(m.d_model, m.n_heads, m.ff_mult, m.dropout)
            for _ in range(m.n_layers)])
        self.ln = nn.LayerNorm(m.d_model)

    def forward(self, x, dt):
        B, L, _ = x.shape
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.pos(self.in_proj(x), dt)
        for blk in self.blocks:
            h = blk(h, attn_mask=causal)
        h = self.ln(h)
        return dict(seq=h, z=h[:, -1, :], attn=self.blocks[-1].last_attn)


class GRUEncoder(nn.Module):
    def __init__(self, n_features: int, m: C.ModelConfig):
        super().__init__()
        d = m.d_model
        self.in_proj = nn.Linear(n_features + 1, d)
        self.norm_in = nn.LayerNorm(d)
        self.gru = nn.GRU(d, d // 2, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=m.dropout)
        self.attn = nn.MultiheadAttention(d, m.n_heads, dropout=m.dropout,
                                          batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(m.dropout)
        self.last_attn = None

    def forward(self, x, dt):
        h0 = self.norm_in(self.in_proj(torch.cat([x, dt.unsqueeze(-1)], -1)))
        seq, _ = self.gru(h0)
        q = seq[:, -1:, :]
        ctx, w = self.attn(q, seq, seq, need_weights=True,
                           average_attn_weights=True)
        self.last_attn = w.detach()
        z = self.ln(seq[:, -1, :] + self.drop(ctx.squeeze(1)))
        return dict(seq=seq, z=z, attn=self.last_attn)


def build_encoder(n_features: int, m: C.ModelConfig) -> nn.Module:
    kind = getattr(m, "encoder", "gru").lower()
    if kind in ("transformer", "attn", "tt"):
        return TemporalEncoder(n_features, m)
    return GRUEncoder(n_features, m)


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
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class SentinelWorldModel(nn.Module):
    def __init__(self, n_features: int, n_states: int = 5,
                 horizon: int = 6, m: Optional[C.ModelConfig] = None):
        super().__init__()
        self.m = m or C.CONFIG.model
        self.n_features = n_features
        self.n_states = n_states
        self.horizon = horizon
        self.encoder = build_encoder(n_features, self.m)
        self.stn = StateTransitionNet(self.m)
        d = self.m.d_model
        self.attack_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(self.m.dropout),
            nn.Linear(d // 2, 1))
        self.prog_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(self.m.dropout),
            nn.Linear(d // 2, n_states))
        self.horizon_attack = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(self.m.dropout),
            nn.Linear(d, horizon))
        self.horizon_prog = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(self.m.dropout),
            nn.Linear(d, horizon * n_states))

    def apply_heads(self, z):
        return dict(attack_logit=self.attack_head(z).squeeze(-1),
                    prog_logits=self.prog_head(z))

    def forward(self, x, dt) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x, dt)
        z = enc["z"]
        mu, logvar = self.stn(z)
        z_next = self.stn.sample(mu, logvar)
        B = x.shape[0]
        out = dict(
            z=z, z_next_mu=mu, z_next_logvar=logvar, z_next=z_next,
            attn=enc["attn"],
            attack_logits_k=self.horizon_attack(z),
            prog_logits_k=self.horizon_prog(z).view(B, self.horizon, self.n_states),
        )
        h0 = self.apply_heads(z)
        h1 = self.apply_heads(z_next)
        out["attack_logit_now"] = h0["attack_logit"]
        out["prog_logits_now"] = h0["prog_logits"]
        out["attack_logit_sim1"] = h1["attack_logit"]
        out["prog_logits_sim1"] = h1["prog_logits"]
        return out

    @torch.no_grad()
    def rollout(self, x, dt, K: Optional[int] = None, M: int = 50
               ) -> Dict[str, np.ndarray]:
        self.eval()
        K = K or self.horizon
        enc = self.encoder(x, dt)
        z0 = enc["z"]
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
            attack_prob=atk.mean(-1),
            attack_ci_lo=np.percentile(atk, 2.5, axis=-1),
            attack_ci_hi=np.percentile(atk, 97.5, axis=-1),
            attack_std=atk.std(-1),
            prog_prob=prg.mean(-2),
            prog_state=prg.mean(-2).argmax(-1),
            z0=z0.cpu().numpy(),
            attn=enc["attn"].cpu().numpy() if enc["attn"] is not None else None,
        )

    def config_dict(self) -> dict:
        return dict(n_features=self.n_features, n_states=self.n_states,
                   horizon=self.horizon, model=asdict(self.m))


def build_model(n_features: int, cfg: Optional[C.InferConfig] = None
                ) -> SentinelWorldModel:
    cfg = cfg or C.CONFIG
    return SentinelWorldModel(n_features, n_states=len(C.PROGRESSION_STATES),
                              horizon=cfg.sequence.horizon, m=cfg.model)


@torch.no_grad()
def wm_predict(model: SentinelWorldModel, X, dt, device="cpu",
               snapshots=None, self_ensemble=False, mc=24):
    """Direct multi-horizon head (+ snapshot average). Returns
    (attack_prob_k [N,K], prog_k [N,K]). Snapshot paths must be absolute."""
    model.eval().to(device)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    dt = torch.as_tensor(np.asarray(dt), dtype=torch.float32, device=device)
    K = getattr(model, "horizon", 6)

    def _one(mdl):
        pa, pk, roll = [], [], []
        for i in range(0, len(X), 512):
            o = mdl(X[i:i + 512], dt[i:i + 512])
            pa.append(torch.sigmoid(o["attack_logits_k"]).cpu().numpy())
            pk.append(o["prog_logits_k"].argmax(-1).cpu().numpy())
        direct = np.concatenate(pa)
        prog = np.concatenate(pk)
        w = float(getattr(C.CONFIG, "self_ensemble_direct_w", 1.0))
        if self_ensemble and w < 0.999:
            for i in range(0, len(X), 512):
                r = mdl.rollout(X[i:i + 512], dt[i:i + 512], K=K, M=mc)
                roll.append(r["attack_prob"])
            direct = w * direct + (1.0 - w) * np.concatenate(roll)
        return direct, prog

    probs, prog = _one(model)
    n = 1
    for sp in (snapshots or []):
        if not (sp and os.path.exists(sp)):
            continue
        try:
            sd = torch.load(sp, map_location=device, weights_only=False)
            snap = SentinelWorldModel(model.n_features, model.n_states,
                                      model.horizon, model.m).to(device)
            snap.load_state_dict(sd)
            probs = probs + _one(snap)[0]
            n += 1
        except Exception as e:                               # pragma: no cover
            print(f"[wm] snapshot {sp} skipped: {e}")
    return probs / n, prog


# =============================================================================
# blend members  (nn_zoo.py)
# =============================================================================
def _with_dt(x, dt):
    return torch.cat([x, dt.unsqueeze(-1)], dim=-1)


class MultiHorizonHead(nn.Module):
    def __init__(self, d: int, K: int, S: int, dropout: float = 0.1):
        super().__init__()
        self.K, self.S = K, S
        self.body = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.attack = nn.Linear(d, K)
        self.prog = nn.Linear(d, K * S)

    def forward(self, h):
        h = self.body(h)
        B = h.shape[0]
        return dict(attack_logits_k=self.attack(h),
                    prog_logits_k=self.prog(h).view(B, self.K, self.S))


class MLPForecaster(nn.Module):
    def __init__(self, n_features: int, L: int, K: int, S: int,
                 hidden=(512, 256, 128), dropout: float = 0.15):
        super().__init__()
        d_in = (n_features + 1) * L
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.head = MultiHorizonHead(d, K, S, dropout)

    def forward(self, x, dt):
        return self.head(self.net(_with_dt(x, dt).flatten(1)))


class RNNForecaster(nn.Module):
    def __init__(self, kind: str, n_features: int, K: int, S: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(n_features + 1, hidden, num_layers=layers,
                       batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = MultiHorizonHead(hidden, K, S, dropout)

    def forward(self, x, dt):
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

    def forward(self, x, dt):
        h = self.tcn(_with_dt(x, dt).transpose(1, 2))
        return self.head(self.norm(h[:, :, -1]))


def build_nn_model(kind: str, n_features: int, L: int,
                   cfg: Optional[C.InferConfig] = None) -> nn.Module:
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
