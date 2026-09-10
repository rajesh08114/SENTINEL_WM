"""raw flow records / CSV  ->  per-anchor attack-progression forecast JSON.

Reuses the research package end to end:
  preprocessing.clean_flow_frame  ->  state_windows.build_state_windows
  ->  (persisted RobustScaler)     ->  forward_sim.simulate_anchor
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.inference.loader import Engine, get_engine

# ---- the columns state_windows._agg_windows indexes without a guard ----------
REQUIRED = [
    "flow_start_epoch",
    "Source IP", "Destination IP", "Destination Port", "Protocol",
    "Flow Duration", "Flow IAT Mean",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
]
# common CICFlowMeter-v4 / lower-case header variants -> our canonical names
ALIASES = {
    "flow id": "Flow ID", "src ip": "Source IP", "dst ip": "Destination IP",
    "src port": "Source Port", "dst port": "Destination Port",
    "protocol": "Protocol", "timestamp": "Timestamp",
    "tot fwd pkts": "Total Fwd Packets", "tot bwd pkts": "Total Backward Packets",
    "total fwd packet": "Total Fwd Packets",
    "total bwd packet": "Total Backward Packets",
    "totlen fwd pkts": "Total Length of Fwd Packets",
    "totlen bwd pkts": "Total Length of Bwd Packets",
    "total length of fwd packet": "Total Length of Fwd Packets",
    "total length of bwd packet": "Total Length of Bwd Packets",
    "flow duration": "Flow Duration", "flow iat mean": "Flow IAT Mean",
    "fwd iat tot": "Fwd IAT Total", "bwd iat tot": "Bwd IAT Total",
}
_FLAG_SRC = {  # flag_true_x  <-  "X Flag Count"
    "flag_true_fin": "FIN Flag Count", "flag_true_syn": "SYN Flag Count",
    "flag_true_rst": "RST Flag Count", "flag_true_psh": "PSH Flag Count",
    "flag_true_ack": "ACK Flag Count", "flag_true_urg": "URG Flag Count",
    "flag_true_cwr": "CWE Flag Count", "flag_true_ece": "ECE Flag Count",
}


class BadUpload(ValueError):
    """422 - the upload is missing columns the pipeline cannot synthesise."""


# ---------------------------------------------------------------------------
def normalise_upload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # alias lower-cased headers
    lower = {c.lower(): c for c in df.columns}
    ren = {lower[a]: canon for a, canon in ALIASES.items()
           if a in lower and canon not in df.columns}
    if ren:
        df = df.rename(columns=ren)

    # time axis
    if "flow_start_epoch" not in df.columns:
        if "Timestamp" in df.columns:
            ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df["flow_start_epoch"] = ts.view("int64") / 1e9
        else:
            raise BadUpload("need a `flow_start_epoch` (float unix seconds) or a "
                            "parseable `Timestamp` column")
    df["flow_start_epoch"] = pd.to_numeric(df["flow_start_epoch"], errors="coerce")
    df = df[df["flow_start_epoch"].notna()].copy()
    if df.empty:
        raise BadUpload("no rows with a valid flow_start_epoch")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise BadUpload(f"missing required columns: {missing}. "
                        f"Provide a CIC-IDS-2017 / CICFlowMeter flow schema.")

    # honest flag counters: synthesise from "* Flag Count" if absent
    for true_col, src in _FLAG_SRC.items():
        if true_col not in df.columns:
            if src in df.columns:
                df[true_col] = (pd.to_numeric(df[src], errors="coerce").fillna(0) > 0).astype("int8")
            else:
                df[true_col] = np.int8(0)

    # inference stand-ins for the training-only columns
    df["Label"] = "BENIGN"
    df["is_attack"] = np.int8(0)
    df["attack_family"] = "BENIGN"
    df["day"] = "upload"
    return df


def flows_to_windows(df: pd.DataFrame) -> pd.DataFrame:
    from sentinel_wm import preprocessing
    from sentinel_wm.state_windows import build_state_windows

    clean = preprocessing.clean_flow_frame(df, verbose=False)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmp = tf.name
    try:
        sw = build_state_windows(flows=clean, out_path=tmp, verbose=False)
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
    return sw.sort_values(["day", "window_index"]).reset_index(drop=True)


def windows_to_tensors(sw: pd.DataFrame, eng: Engine):
    """contiguous L-window slices -> (X_scaled [N,L,F], dt [N,L], anchor_meta)."""
    L, feat_cols = eng.L, [c for c in eng.feat_cols if c in sw.columns]
    Xr, DTr, meta = [], [], []
    for day, g in sw.groupby("day", sort=False):
        g = g.sort_values("window_index")
        F = g[feat_cols].to_numpy(np.float32)
        dt = g["time_since_prev_window"].to_numpy(np.float32)
        wi = g["window_index"].to_numpy(np.int64)
        for t in range(L - 1, len(g)):
            if wi[t] - wi[t - L + 1] != L - 1:          # must be contiguous
                continue
            Xr.append(F[t - L + 1:t + 1])
            DTr.append(dt[t - L + 1:t + 1])
            meta.append({"day": str(day), "window_index": int(wi[t]),
                         "window_start_epoch": float(g["window_start"].to_numpy()[t])
                         if "window_start" in g.columns else None})
    if not Xr:
        return (np.empty((0, L, len(feat_cols)), np.float32),
                np.empty((0, L), np.float32), [])
    Xr = np.stack(Xr)
    X = eng.scaler.transform(Xr.reshape(-1, Xr.shape[-1])).reshape(Xr.shape).astype(np.float32)
    DT = np.log1p(np.clip(np.stack(DTr), 0, None)).astype(np.float32)
    return X, DT, meta


def _summarise(anchors: list[dict], thr: float) -> dict:
    if not anchors:
        return {"n_alerts": 0, "max_attack_prob": 0.0, "phases": []}
    phases, mx, n_alert = set(), 0.0, 0
    for a in anchors:
        mx = max(mx, a.get("max_detection_prob", a["max_attack_prob"]))
        n_alert += int(a["alert"])
        for h in a["horizon"]:
            ph = h["attck"]["kill_chain_phase"]
            if ph not in ("None", "Attack (unspecified)"):
                phases.add(ph)
    return {"n_alerts": n_alert, "max_attack_prob": round(mx, 4),
            "phases": sorted(phases), "alert_threshold": thr}


# ---------------------------------------------------------------------------
def forecast(df: pd.DataFrame, family_hint: Optional[str] = None,
             explain: bool = True, eng: Optional[Engine] = None) -> dict:
    from sentinel_wm import forward_sim
    from app.settings import settings

    eng = eng or get_engine()
    n_flows = len(df)
    sw = flows_to_windows(normalise_upload(df))
    X, DT, meta = windows_to_tensors(sw, eng)

    explainer = eng.explainer if explain else None
    anchors = []
    for i, m in enumerate(meta):
        res = forward_sim.simulate_anchor(
            eng.model, X[i], DT[i], eng.ckpt,
            meta={"dominant_family_hint": (family_hint or "BENIGN"),
                  "window_index": m["window_index"],
                  "window_start_epoch": m["window_start_epoch"]},
            M_samples=settings.mc_samples, device=eng.device,
            feature_vec=X[i][-1], explainer=explainer)
        anchors.append(res)

    used_model = "SENTINEL-WM"
    if eng.members and anchors:
        used_model = "SENTINEL-WM (system)"
        _apply_system_blend(anchors, X, DT, eng)

    return {
        "meta": {"n_flows": n_flows, "n_windows": int(len(sw)),
                 "n_anchors": len(anchors), "window_seconds": eng.window_seconds,
                 "history_windows": eng.L, "horizon_steps": eng.K,
                 "feature_dim": eng.n_features, "model": used_model,
                 "blend_weight": (round(eng.blend_weight, 3)
                                  if used_model.endswith("(system)") else None),
                 "family_hint": family_hint or "BENIGN"},
        "summary": _summarise(anchors, eng.alert_threshold),
        "anchors": anchors,
    }


def _apply_system_blend(anchors: list[dict], X, DT, eng: Engine) -> None:
    """Blend the val-tuned SENTINEL-WM (system) probability onto each anchor.

    The world-model rollout (attack_prob + 95% CI, progression state,
    progression_dist, ATT&CK phase) is kept as the forecast narrative; the
    sharper blended probability drives `detection_prob` and the alert / lead-time.
    """
    K = eng.K
    wm = np.array([[h["attack_prob"] for h in a["horizon"]] for a in anchors],
                  dtype=np.float32)                          # [N, K]
    mps = []
    for _nm, pred in eng.members:
        try:
            mps.append(np.asarray(pred.predict(X, DT)["attack_prob_k"], np.float32))
        except Exception:                                    # pragma: no cover
            pass
    if not mps:
        return
    member = np.mean(mps, axis=0)                            # [N, K]
    w = float(eng.blend_weight)
    sysp = w * wm + (1.0 - w) * member                       # [N, K]
    thr = eng.alert_threshold
    Wsec = eng.window_seconds
    for i, a in enumerate(anchors):
        first = None
        for k, h in enumerate(a["horizon"]):
            p = float(sysp[i, k])
            h["detection_prob"] = round(p, 6)
            if first is None and p >= thr:
                first = k + 1
        a["detection_model"] = "SENTINEL-WM (system)"
        a["max_detection_prob"] = float(sysp[i].max())
        a["alert"] = first is not None
        a["first_alert_k"] = first
        a["lead_time_seconds"] = 0 if first is None else (K - first + 1) * Wsec
