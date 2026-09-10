"""The inference engine: a model bundle directory -> forecast JSON.

    eng = load_bundle("/models", device="cpu", serve_mode="auto")
    result = eng.forecast(flows_dataframe, family_hint="PortScan")

Depends only on: this package, torch, numpy, pandas, scikit-learn (for the
pickled RobustScaler). Nothing in the research tree.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch

from . import attack_stages as A
from . import explain as _explain
from . import preprocess
from . import schema as C
from . import windows as _windows
from .net import build_model, build_nn_model, wm_predict


class BundleNotFound(RuntimeError):
    pass


class BundleContractError(RuntimeError):
    pass


# ---- the ~15 columns windows._agg_windows indexes without a guard ------------
REQUIRED = [
    "flow_start_epoch",
    "Source IP", "Destination IP", "Destination Port", "Protocol",
    "Flow Duration", "Flow IAT Mean",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
]
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
_FLAG_SRC = {
    "flag_true_fin": "FIN Flag Count", "flag_true_syn": "SYN Flag Count",
    "flag_true_rst": "RST Flag Count", "flag_true_psh": "PSH Flag Count",
    "flag_true_ack": "ACK Flag Count", "flag_true_urg": "URG Flag Count",
    "flag_true_cwr": "CWE Flag Count", "flag_true_ece": "ECE Flag Count",
}


class BadUpload(ValueError):
    """422 - the upload is missing columns the pipeline cannot synthesise."""


def normalise_upload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    ren = {lower[a]: canon for a, canon in ALIASES.items()
           if a in lower and canon not in df.columns}
    if ren:
        df = df.rename(columns=ren)

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

    for true_col, src in _FLAG_SRC.items():
        if true_col not in df.columns:
            if src in df.columns:
                df[true_col] = (pd.to_numeric(df[src], errors="coerce").fillna(0) > 0).astype("int8")
            else:
                df[true_col] = np.int8(0)

    df["Label"] = "BENIGN"
    df["is_attack"] = np.int8(0)
    df["attack_family"] = "BENIGN"
    df["day"] = "upload"
    return df


# ---------------------------------------------------------------------------
def simulate_anchor(model, x: np.ndarray, dt: np.ndarray, ckpt: dict,
                    meta: Optional[dict] = None, M_samples: int = 50,
                    device: str = "cpu", feature_vec: Optional[np.ndarray] = None,
                    explainer=None) -> dict:
    """x: [L, F]  dt: [L]  -> one forecast dict. Vendored from
    research/sentinel_wm/forward_sim.simulate_anchor."""
    K = ckpt["sequence"]["K"]
    W = C.CONFIG.window.window_seconds
    xb = torch.as_tensor(x[None], dtype=torch.float32, device=device)
    dtb = torch.as_tensor(dt[None], dtype=torch.float32, device=device)
    roll = model.rollout(xb, dtb, K=K, M=M_samples)

    thr = ckpt.get("alert_threshold", 0.7)
    meta = meta or {}
    fam_hint = meta.get("dominant_family_hint", "BENIGN")
    prev_fam = meta.get("prev_family")

    horizon, first_alert_k = [], None
    for k in range(K):
        p = float(roll["attack_prob"][0, k])
        state_probs = roll["prog_prob"][0, k].tolist()
        stage = A.assess_forecast(state_probs, horizon_k=k + 1,
                                  dominant_family_hint=fam_hint,
                                  prev_family=prev_fam, attack_prob=p)
        if first_alert_k is None and p >= thr:
            first_alert_k = k + 1
        horizon.append(dict(
            k=k + 1, horizon_seconds=(k + 1) * W, attack_prob=p,
            attack_ci=[float(roll["attack_ci_lo"][0, k]),
                       float(roll["attack_ci_hi"][0, k])],
            attack_std=float(roll["attack_std"][0, k]),
            progression_state=C.IDX_TO_STATE[int(roll["prog_state"][0, k])],
            progression_dist={C.IDX_TO_STATE[i]: round(v, 4)
                              for i, v in enumerate(state_probs)},
            attck=stage.to_dict()))

    result = dict(
        meta=meta, alert_threshold=thr,
        alert=bool(first_alert_k is not None),
        lead_time_seconds=(0 if first_alert_k is None
                           else (K - first_alert_k + 1) * W),
        first_alert_k=first_alert_k,
        max_attack_prob=float(roll["attack_prob"][0].max()),
        horizon=horizon)
    if explainer is not None and feature_vec is not None:
        try:
            result["driving_features"] = explainer(x, dt, feature_vec)
        except Exception as e:                              # pragma: no cover
            result["driving_features"] = {"error": str(e)}
    return result


# ---------------------------------------------------------------------------
@dataclass
class InferenceEngine:
    model: Any
    ckpt: dict
    scaler: Any
    feat_cols: list
    explainer: Callable
    L: int
    K: int
    n_features: int
    window_seconds: int
    alert_threshold: float
    device: str
    manifest: dict = field(default_factory=dict)
    members: list = field(default_factory=list)       # [(name, nn.Module)]
    blend_weight: float = 1.0
    serve_mode: str = "world_model"
    mc_samples: int = 50

    # -- flows -> state windows -> L-window tensors --------------------------
    def flows_to_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = preprocess.clean_flow_frame(normalise_upload(df), verbose=False)
        sw = _windows.build_state_windows(clean, C.CONFIG)
        return sw.sort_values(["day", "window_index"]).reset_index(drop=True)

    def windows_to_tensors(self, sw: pd.DataFrame):
        L = self.L
        feat_cols = [c for c in self.feat_cols if c in sw.columns]
        Xr, DTr, meta = [], [], []
        for day, g in sw.groupby("day", sort=False):
            g = g.sort_values("window_index")
            Fm = g[feat_cols].to_numpy(np.float32)
            dtc = g["time_since_prev_window"].to_numpy(np.float32)
            wi = g["window_index"].to_numpy(np.int64)
            ws = (g["window_start"].to_numpy()
                  if "window_start" in g.columns else np.full(len(g), np.nan))
            for t in range(L - 1, len(g)):
                if wi[t] - wi[t - L + 1] != L - 1:
                    continue
                Xr.append(Fm[t - L + 1:t + 1])
                DTr.append(dtc[t - L + 1:t + 1])
                meta.append({"day": str(day), "window_index": int(wi[t]),
                             "window_start_epoch": float(ws[t])
                             if np.isfinite(ws[t]) else None})
        if not Xr:
            return (np.empty((0, L, len(feat_cols)), np.float32),
                    np.empty((0, L), np.float32), [])
        Xr = np.stack(Xr)
        X = self.scaler.transform(
            Xr.reshape(-1, Xr.shape[-1])).reshape(Xr.shape).astype(np.float32)
        DT = np.log1p(np.clip(np.stack(DTr), 0, None)).astype(np.float32)
        return X, DT, meta

    # -- the SENTINEL-WM (system) blend -----------------------------------
    def _apply_system_blend(self, anchors: list, X, DT) -> None:
        if not self.members or not anchors:
            return
        wm = np.array([[h["attack_prob"] for h in a["horizon"]] for a in anchors],
                      np.float32)
        mps = []
        for _nm, m in self.members:
            with torch.no_grad():
                xb = torch.as_tensor(X, dtype=torch.float32, device=self.device)
                db = torch.as_tensor(DT, dtype=torch.float32, device=self.device)
                pa = []
                for i in range(0, len(xb), 512):
                    o = m(xb[i:i + 512], db[i:i + 512])
                    pa.append(torch.sigmoid(o["attack_logits_k"]).cpu().numpy())
                mps.append(np.concatenate(pa))
        if not mps:
            return
        member = np.mean(mps, axis=0)
        w = float(self.blend_weight)
        sysp = w * wm + (1.0 - w) * member
        thr, Wsec = self.alert_threshold, self.window_seconds
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
            a["lead_time_seconds"] = 0 if first is None else (self.K - first + 1) * Wsec

    # -- public entry point --------------------------------------------
    def forecast(self, df: pd.DataFrame, family_hint: Optional[str] = None,
                 explain: bool = True) -> dict:
        n_flows = len(df)
        sw = self.flows_to_windows(df)
        X, DT, meta = self.windows_to_tensors(sw)
        expl = self.explainer if explain else None
        anchors = []
        for i, m in enumerate(meta):
            anchors.append(simulate_anchor(
                self.model, X[i], DT[i], self.ckpt,
                meta={"dominant_family_hint": family_hint or "BENIGN",
                      "window_index": m["window_index"],
                      "window_start_epoch": m["window_start_epoch"]},
                M_samples=self.mc_samples, device=self.device,
                feature_vec=X[i][-1], explainer=expl))
        used = "SENTINEL-WM"
        if self.members and anchors:
            used = "SENTINEL-WM (system)"
            self._apply_system_blend(anchors, X, DT)

        alerts = sum(int(a["alert"]) for a in anchors)
        mx = max((a.get("max_detection_prob", a["max_attack_prob"])
                  for a in anchors), default=0.0)
        phases = sorted({h["attck"]["kill_chain_phase"]
                         for a in anchors for h in a["horizon"]
                         if h["attck"]["kill_chain_phase"] not in
                         ("None", "Attack (unspecified)")})
        return {
            "meta": {"n_flows": n_flows, "n_windows": int(len(sw)),
                     "n_anchors": len(anchors), "window_seconds": self.window_seconds,
                     "history_windows": self.L, "horizon_steps": self.K,
                     "feature_dim": self.n_features, "model": used,
                     "blend_weight": (round(self.blend_weight, 3)
                                      if used.endswith("(system)") else None),
                     "family_hint": family_hint or "BENIGN"},
            "summary": {"n_alerts": alerts, "max_attack_prob": round(mx, 4),
                        "phases": phases, "alert_threshold": self.alert_threshold},
            "anchors": anchors,
        }


# ---------------------------------------------------------------------------
def load_bundle(bundle_dir: str, device: str = "cpu",
                serve_mode: str = "auto", mc_samples: int = 50) -> InferenceEngine:
    bd = Path(bundle_dir)
    wm = bd / "world_model.pt"
    if not wm.exists():
        raise BundleNotFound(
            f"no model bundle at {bd} (missing world_model.pt). Build it with: "
            f"cd research && python -m sentinel_wm.research all && "
            f"python -m sentinel_wm.cli bundle {bd}")

    manifest = {}
    if (bd / "bundle.json").exists():
        manifest = json.loads((bd / "bundle.json").read_text())

    ckpt = torch.load(wm, map_location=device, weights_only=False)
    n_feat = int(ckpt["config"]["n_features"])
    C.CONFIG.model.encoder = (ckpt.get("config", {}).get("model", {})
                              .get("encoder", C.CONFIG.model.encoder))
    C.CONFIG.sequence.horizon = int(ckpt["sequence"]["K"])
    C.CONFIG.mc_samples = mc_samples

    model = build_model(n_feat, C.CONFIG)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    snaps = [str(bd / sp) for sp in ckpt.get("snapshots", []) if (bd / sp).exists()]
    ckpt["snapshots"] = snaps

    with open(bd / "state_scaler.pkl", "rb") as fh:
        sc = pickle.load(fh)
    scaler = sc["scaler"]
    feat_cols = list(sc.get("feature_names") or _windows.STATE_FEATURE_COLS)

    # --- CONTRACT: the bundle's feature schema must match this vendored copy --
    bundle_feats = manifest.get("feature_names")
    if bundle_feats and list(bundle_feats) != list(_windows.STATE_FEATURE_COLS):
        raise BundleContractError(
            "bundle feature schema != sentinel_infer.windows.STATE_FEATURE_COLS. "
            "The bundle was built from a different research revision - re-vendor "
            "sentinel_infer (see backend/app/sentinel_infer/README.md).")
    if len(feat_cols) != n_feat:
        raise BundleContractError(
            f"scaler has {len(feat_cols)} features but the checkpoint expects {n_feat}")

    explainer = _explain.make_explainer(
        model, {"feature_names": np.array(feat_cols, dtype=object)}, device)

    members = []
    if serve_mode in ("auto", "system"):
        for nm in manifest.get("system_members", ["tcn", "lstm", "gru"]):
            mp = bd / "models" / "nn" / f"{nm}.pt"
            if not mp.exists():
                continue
            try:
                mck = torch.load(mp, map_location=device, weights_only=False)
                mm = build_nn_model(mck.get("kind", nm),
                                    int(mck.get("n_features", n_feat)),
                                    int(mck["sequence"]["L"]), C.CONFIG)
                mm.load_state_dict(mck["state_dict"])
                mm.eval().to(device)
                members.append((nm, mm))
            except Exception as e:                          # pragma: no cover
                print(f"[bundle] member {nm} skipped: {e}")
    serve = "system" if members else "world_model"
    if serve_mode == "system" and not members:
        print("[bundle] serve_mode=system but no member models present -> world_model")

    thr = float(manifest.get("system_threshold")
                or ckpt.get("alert_threshold", 0.7))
    ckpt["alert_threshold"] = thr

    eng = InferenceEngine(
        model=model, ckpt=ckpt, scaler=scaler, feat_cols=feat_cols,
        explainer=explainer, L=int(ckpt["sequence"]["L"]),
        K=int(ckpt["sequence"]["K"]), n_features=n_feat,
        window_seconds=int(C.CONFIG.window.window_seconds),
        alert_threshold=thr, device=device, manifest=manifest,
        members=members,
        blend_weight=float(manifest.get("system_blend_weight", 0.5)) if members else 1.0,
        serve_mode=serve, mc_samples=mc_samples)

    # warm pass
    try:
        z = np.zeros((eng.L, len(feat_cols)), np.float32)
        simulate_anchor(model, z, np.zeros(eng.L, np.float32), ckpt,
                        meta={}, M_samples=4, device=device)
    except Exception as e:                                  # pragma: no cover
        print(f"[bundle] warm pass failed (non-fatal): {e}")

    print(f"[sentinel_infer] engine ready: mode={eng.serve_mode} "
          f"members={[n for n, _ in members]} blend_w={eng.blend_weight:.2f} "
          f"L={eng.L} K={eng.K} F={n_feat} device={device} bundle={bd}")
    return eng
