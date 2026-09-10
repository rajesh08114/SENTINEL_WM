#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  gat.py   -  Graph Attention Network baseline (from scratch)
# -----------------------------------------------------------------------------
# The proposal's Advanced-tier spatial encoder (6.4 / 12 Tier-3), implemented
# WITHOUT torch-geometric so it stays offline-installable.
#
#   per window t :  GAT over the host graph G_t  ->  window embedding g_t
#   over the L windows :  GRU(g_{t-L+1..t})      ->  temporal state
#   -> MultiHorizonHead -> attack_logits_k [B,K], prog_logits_k [B,K,S]
#
# Trains through `sentinel_wm.nn_common.train_nn` via a `batch_forward` hook, so
# it gets the exact same class-weighting / early-stop / threshold-calibration /
# checkpoint protocol as every other model.
# =============================================================================
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sentinel_wm import config as C
from sentinel_wm import metrics as M
from sentinel_wm.graph_windows import load_graph_windows
from sentinel_wm.sequences import load_sequences


# -----------------------------------------------------------------------------
# GAT layer  (masked multi-head attention over nodes, GATv2-style scoring)
# -----------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """GAT (Velickovic 2018) additive attention - O(N^2) not O(N^2 D) in memory."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj_mask):
        # h [B,N,in] ; adj_mask [B,N,N] (1 where an edge/self exists)
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.out_dim).permute(0, 2, 1, 3)
        # e_ij = LeakyReLU(a_src.Wh_i + a_dst.Wh_j)   -> [B,H,N,N]
        e_src = (Wh * self.a_src.view(1, self.heads, 1, -1)).sum(-1, keepdim=True)
        e_dst = (Wh * self.a_dst.view(1, self.heads, 1, -1)).sum(-1).unsqueeze(2)
        e = F.leaky_relu(e_src + e_dst, 0.2)
        e = e.masked_fill(adj_mask.unsqueeze(1) == 0, float("-inf"))
        alpha = torch.nan_to_num(torch.softmax(e, dim=-1))          # isolated -> 0
        alpha = self.drop(alpha)
        out = torch.einsum("bhij,bhjd->bhid", alpha, Wh)            # [B,H,N,D]
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_dim)
        else:
            out = out.mean(1)
        return out + self.bias


class GATWindowEncoder(nn.Module):
    def __init__(self, node_feat_dim, d_model=96, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_in = node_feat_dim
        for i in range(layers):
            last = i == layers - 1
            self.blocks.append(GraphAttentionLayer(
                d_in, d_model if last else d_model // heads, heads=heads,
                dropout=dropout, concat=not last))
            d_in = d_model
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, node_feat, adj, node_mask):
        # add self-loops so isolated nodes still attend to themselves
        eye = torch.eye(node_feat.shape[1], device=node_feat.device).unsqueeze(0)
        adj_mask = ((adj > 0).float() + eye).clamp(max=1.0)
        adj_mask = adj_mask * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        h = node_feat
        for blk in self.blocks:
            h = self.act(blk(h, adj_mask))
        h = self.norm(h) * node_mask.unsqueeze(-1)
        # masked mean pool over nodes
        return h.sum(1) / node_mask.sum(1, keepdim=True).clamp(min=1.0)


class GATForecaster(nn.Module):
    def __init__(self, node_feat_dim, L, K, S, d_model=96, gru_hidden=128,
                 gat_layers=2, heads=4, dropout=0.12):
        super().__init__()
        self.L, self.K, self.S = L, K, S
        self.win_encoder = GATWindowEncoder(node_feat_dim, d_model, gat_layers,
                                            heads, dropout)
        self.gru = nn.GRU(d_model, gru_hidden, num_layers=2, batch_first=True,
                          dropout=dropout)
        self.norm = nn.LayerNorm(gru_hidden)
        from sentinel_wm.nn_zoo import MultiHorizonHead
        self.head = MultiHorizonHead(gru_hidden, K, S, dropout)

    def forward(self, node_feat, adj, node_mask) -> Dict[str, torch.Tensor]:
        # node_feat [B,L,N,Fn] ; adj [B,L,N,N] ; node_mask [B,L,N]
        B, L, N, Fn = node_feat.shape
        g = self.win_encoder(node_feat.reshape(B * L, N, Fn),
                             adj.reshape(B * L, N, N),
                             node_mask.reshape(B * L, N))
        g = g.reshape(B, L, -1)
        out, _ = self.gru(g)
        return self.head(self.norm(out[:, -1, :]))


# -----------------------------------------------------------------------------
# data plumbing
# -----------------------------------------------------------------------------
class _GraphCtx:
    def __init__(self, L: int):
        gw = load_graph_windows()
        self.node_feat = gw["node_feat"].astype(np.float32)
        self.adj = gw["adj"].astype(np.float32)
        self.node_mask = gw["node_mask"].astype(np.float32)
        self.L = L
        self.row_of = {(str(d), int(w)): i
                       for i, (d, w) in enumerate(zip(gw["day"], gw["window_index"]))}
        self.N = int(gw["n_max"]); self.Fn = int(gw["n_node_feat"])

    def gather(self, day: str, wi: int):
        rows = []
        for k in range(self.L - 1, -1, -1):
            rows.append(self.row_of.get((str(day), int(wi) - k)))
        rows = [r if r is not None else -1 for r in rows]
        nf = np.stack([self.node_feat[r] if r >= 0 else np.zeros((self.N, self.Fn), np.float32)
                       for r in rows])
        aj = np.stack([self.adj[r] if r >= 0 else np.zeros((self.N, self.N), np.float32)
                       for r in rows])
        mk = np.stack([self.node_mask[r] if r >= 0 else np.zeros(self.N, np.float32)
                       for r in rows])
        return nf, aj, mk


class _GraphDataset(Dataset):
    def __init__(self, ctx: _GraphCtx, day, wi, y_atk, y_prog):
        self.ctx, self.day, self.wi = ctx, day, wi
        self.y_atk, self.y_prog = y_atk, y_prog

    def __len__(self):
        return len(self.wi)

    def __getitem__(self, i):
        nf, aj, mk = self.ctx.gather(self.day[i], int(self.wi[i]))
        return (torch.from_numpy(nf), torch.from_numpy(aj), torch.from_numpy(mk),
                torch.tensor(self.y_atk[i]), torch.tensor(self.y_prog[i]))


def make_gat_batch_forward(ctx: _GraphCtx):
    def bf(mode, payload, device):
        if mode == "loader":
            d, bs = payload
            ds = _GraphDataset(ctx, d["day"], d["window_index"],
                               d["y_atk"].cpu().numpy(), d["y_prog"].cpu().numpy())
            return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
        if mode == "evaluate":
            model, d, cfg, dev, seq, split, thr = payload
            return _gat_eval(model, ctx, d, cfg, dev, thr)
        # training step: `mode` is the model, `payload` is the batch tuple
        model = mode
        nf, aj, mk, ya, yp = [t.to(device) for t in payload]
        return model(nf, aj, mk), ya, yp
    return bf


# -----------------------------------------------------------------------------
@torch.no_grad()
def _gat_eval(model, ctx: _GraphCtx, d, cfg, device, threshold=None) -> Dict:
    model.eval()
    ds = _GraphDataset(ctx, d["day"], d["window_index"],
                       d["y_atk"].cpu().numpy(), d["y_prog"].cpu().numpy())
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    probs_k, prog_k = [], []
    for nf, aj, mk, _ya, _yp in loader:
        out = model(nf.to(device), aj.to(device), mk.to(device))
        probs_k.append(torch.sigmoid(out["attack_logits_k"]).cpu().numpy())
        prog_k.append(out["prog_logits_k"].argmax(-1).cpu().numpy())
    probs_k = np.concatenate(probs_k, 0)
    prog_k = np.concatenate(prog_k, 0)
    y_atk = d["y_atk"].cpu().numpy().astype(int)
    y_prog = d["y_prog"].cpu().numpy().astype(int)
    y_now = d["y_now"].cpu().numpy().astype(int)
    wi = d["window_index"]
    W = cfg.window.window_seconds
    K = cfg.sequence.horizon
    if threshold is None:
        threshold = M.calibrate_threshold(y_atk.max(1), probs_k.max(1),
                                          cfg.train.target_fpr)
    cal = M.expected_calibration_error(y_atk[:, 0], probs_k[:, 0])
    return dict(threshold=float(threshold),
                any_horizon=M.binary_scores(y_atk.max(1), probs_k.max(1), threshold),
                per_horizon=M.horizon_table(y_atk, probs_k, W, threshold),
                brier_k1=M.brier_score(y_atk[:, 0], probs_k[:, 0]),
                ece_k1=cal["ece"], reliability=cal["bins"],
                progression_acc=float((prog_k == y_prog).mean()),
                progression_acc_k1=float((prog_k[:, 0] == y_prog[:, 0]).mean()),
                lead_time=M.lead_time(wi, y_now, probs_k.max(1), W, threshold, K))


# -----------------------------------------------------------------------------
def run_gat(cfg: C.Config = None, device: str = None, epochs: int = 60,
            d_model: int = 96, verbose: bool = True) -> Dict:
    from sentinel_wm.nn_common import train_nn
    cfg = cfg or C.CONFIG
    seq = load_sequences()
    L, K = int(seq["L"]), int(seq["K"])
    ctx = _GraphCtx(L)
    S = len(C.PROGRESSION_STATES)
    model = GATForecaster(ctx.Fn, L, K, S, d_model=d_model)
    bf = make_gat_batch_forward(ctx)
    return train_nn(model, "gat", kind="gat", family="graph", cfg=cfg,
                    device=device, epochs=epochs, seq=seq,
                    batch_forward=bf,
                    extra_meta=dict(n_node_feat=ctx.Fn, n_max=ctx.N,
                                    d_model=d_model),
                    verbose=verbose)


if __name__ == "__main__":
    run_gat()

