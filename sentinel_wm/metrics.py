#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  metrics.py
# -----------------------------------------------------------------------------
# Shared metric functions used by the baselines and the world model so the
# benchmark table (proposal section 10) is computed identically for both.
#
#   * classification : F1 / precision / recall / FPR / AUROC / accuracy
#   * calibration    : Brier score, Expected Calibration Error, reliability bins
#   * forecasting    : per-horizon table, Mean Lead Time, Time-to-Warning
#
# Pure numpy + sklearn. No torch.
# =============================================================================
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:                                   # pragma: no cover
    roc_auc_score = None


# -----------------------------------------------------------------------------
# classification
# -----------------------------------------------------------------------------
def binary_scores(y_true: Sequence[int], y_prob: Sequence[float],
                  threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / max(1, len(y_true))

    auroc = float("nan")
    if roc_auc_score is not None and len(np.unique(y_true)) == 2:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    return dict(threshold=float(threshold), f1=f1, precision=prec, recall=rec,
                fpr=fpr, auroc=auroc, accuracy=acc,
                tp=tp, fp=fp, tn=tn, fn=fn, n=int(len(y_true)),
                positives=int(y_true.sum()))


def calibrate_threshold(y_true: Sequence[int], y_prob: Sequence[float],
                        target_fpr: float = 0.05,
                        grid: Optional[np.ndarray] = None) -> float:
    """Smallest threshold whose FPR on this set is <= target_fpr (proposal 6.6)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    grid = grid if grid is not None else np.linspace(0.01, 0.99, 99)
    best = 0.5
    for thr in grid:
        s = binary_scores(y_true, y_prob, thr)
        if s["fpr"] <= target_fpr:
            best = float(thr)
            break
    return best


# -----------------------------------------------------------------------------
# calibration
# -----------------------------------------------------------------------------
def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_prob = np.asarray(y_prob, float).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> Dict:
    y_true = np.asarray(y_true, float).ravel()
    y_prob = np.asarray(y_prob, float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not m.any():
            rows.append(dict(bin_lo=lo, bin_hi=hi, n=0, conf=0.0, acc=0.0))
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        ece += (m.sum() / len(y_prob)) * abs(acc - conf)
        rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=int(m.sum()),
                         conf=conf, acc=acc))
    return dict(ece=float(ece), bins=rows)


# -----------------------------------------------------------------------------
# forecasting  --  per-horizon table
# -----------------------------------------------------------------------------
def horizon_table(y_true_k: np.ndarray, y_prob_k: np.ndarray,
                  window_seconds: int, threshold: float = 0.5) -> List[Dict]:
    """
    y_true_k / y_prob_k : [N, K]  (column k = horizon step k+1)
    returns one metrics dict per horizon step.
    """
    y_true_k = np.asarray(y_true_k)
    y_prob_k = np.asarray(y_prob_k, float)
    K = y_true_k.shape[1]
    rows = []
    for k in range(K):
        s = binary_scores(y_true_k[:, k], y_prob_k[:, k], threshold)
        s["k"] = k + 1
        s["horizon_seconds"] = (k + 1) * window_seconds
        s["brier"] = brier_score(y_true_k[:, k], y_prob_k[:, k])
        rows.append(s)
    return rows


# -----------------------------------------------------------------------------
# Mean Lead Time  --  the core operational metric (proposal 10.1 / 11.5)
# -----------------------------------------------------------------------------
def lead_time(window_index: np.ndarray,
              y_now: np.ndarray,
              y_prob_any_k: np.ndarray,
              window_seconds: int,
              threshold: float,
              horizon_k: int) -> Dict:
    """
    For every attack episode in an ordered stream of anchor windows:
      t_onset       = first window whose CURRENT label (y_now) is attack
      t_first_warn  = first PRECEDING window whose forecast P(attack in next K)
                      crossed `threshold`
      lead          = (t_onset - t_first_warn) * window_seconds     [>= 0]
    Warnings with no attack within `horizon_k` windows are false alarms.

    window_index : [N]   window id of each anchor (assumed one day, sorted)
    y_now        : [N]   current-window attack label of the anchor
    y_prob_any_k : [N]   max_k P(attack at t+k)
    """
    order = np.argsort(window_index)
    wi = np.asarray(window_index)[order]
    now = np.asarray(y_now)[order].astype(int)
    warn = (np.asarray(y_prob_any_k)[order] >= threshold).astype(int)

    # onsets = attack window whose predecessor was benign
    onsets = [i for i in range(len(now))
              if now[i] == 1 and (i == 0 or now[i - 1] == 0)]

    leads, warned = [], 0
    for oi in onsets:
        # A K-step forecaster can only legitimately claim lead time up to K
        # windows before onset - cap the backward search there, then walk back
        # over the contiguous warning run that touches onset.
        first_warn = None
        lookback_floor = max(0, oi - horizon_k)
        j = oi - 1
        while j >= lookback_floor and now[j] == 0:
            if warn[j] == 1:
                first_warn = j
            else:
                break            # warning must be contiguous up to onset
            j -= 1
        if first_warn is not None:
            warned += 1
            leads.append((wi[oi] - wi[first_warn]) * window_seconds)
        else:
            leads.append(0.0)

    # false-alarm rate: warnings on benign windows with no attack within K
    fa = 0
    benign_idx = np.where(now == 0)[0]
    for i in benign_idx:
        if warn[i] == 1:
            fut = now[i + 1: i + 1 + horizon_k]
            if fut.size == 0 or fut.max() == 0:
                fa += 1
    n_benign = int((now == 0).sum())

    leads = np.asarray(leads, float)
    return dict(
        n_episodes=len(onsets),
        n_episodes_warned=int(warned),
        detection_rate=float(warned / len(onsets)) if onsets else 0.0,
        mean_lead_time_s=float(leads[leads > 0].mean()) if (leads > 0).any() else 0.0,
        median_lead_time_s=float(np.median(leads[leads > 0])) if (leads > 0).any() else 0.0,
        max_lead_time_s=float(leads.max()) if leads.size else 0.0,
        false_alarm_rate=float(fa / n_benign) if n_benign else 0.0,
        per_episode_lead_s=leads.tolist())


def summarise(tag: str, s: Dict) -> str:
    return (f"{tag:22s} F1={s['f1']:.3f} P={s['precision']:.3f} "
            f"R={s['recall']:.3f} FPR={s['fpr']:.3f} AUROC={s['auroc']:.3f} "
            f"(n={s['n']}, pos={s['positives']})")
