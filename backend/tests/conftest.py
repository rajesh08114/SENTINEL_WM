"""Test fixtures.

`tiny_bundle` builds a throwaway model bundle from RANDOM weights so the whole
CSV / streaming plumbing (windowing -> scaler -> simulate_anchor -> ATT&CK ->
explain -> JSON) is exercised without a 10-minute training run. The forecast
*numbers* are meaningless; the *shapes and contract* are what these tests check.
"""
from __future__ import annotations

import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory) -> str:
    dst = tmp_path_factory.mktemp("bundle")
    import torch
    from sklearn.preprocessing import RobustScaler

    from sentinel_wm import config as C
    from sentinel_wm.models import build_model
    from sentinel_wm.state_windows import STATE_FEATURE_COLS

    F = len(STATE_FEATURE_COLS)
    C.CONFIG.sequence.horizon = 6
    model = build_model(F, C.CONFIG)
    torch.save(
        dict(state_dict=model.state_dict(),
             config={"n_features": F, "model": {"encoder": C.CONFIG.model.encoder}},
             sequence={"L": 12, "K": 6},
             alert_threshold=0.5, snapshots=[], self_ensemble=False),
        dst / "world_model.pt")

    sc = RobustScaler().fit(np.random.default_rng(0).normal(size=(512, F)))
    with open(dst / "state_scaler.pkl", "wb") as fh:
        pickle.dump({"scaler": sc, "feature_names": list(STATE_FEATURE_COLS)}, fh)

    # random-weight blend members so the SENTINEL-WM (system) path is exercised
    from sentinel_wm.nn_zoo import build_nn_model
    nn_dir = dst / "models" / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("tcn", "lstm", "gru"):
        mm = build_nn_model(kind, F, 12, C.CONFIG)
        torch.save(dict(state_dict=mm.state_dict(), kind=kind, family="nn",
                        n_features=F, sequence={"L": 12, "K": 6},
                        alert_threshold=0.5, feature_names=list(STATE_FEATURE_COLS)),
                   nn_dir / f"{kind}.pt")
        (nn_dir / f"{kind}.meta.json").write_text(json.dumps(
            {"name": kind, "kind": kind, "family": "nn", "threshold": 0.5,
             "params": sum(p.numel() for p in mm.parameters()), "metrics": {}}))

    os.environ["SENTINEL_WM_MODEL_DIR"] = str(dst)
    from sentinel_wm import registry
    registry.build_registry(verbose=False, base_dir=str(dst))

    (dst / "bundle.json").write_text(json.dumps(
        {"created": "test", "git_sha": "test", "L": 12, "K": 6, "n_features": F,
         "window_seconds": 10, "encoder": C.CONFIG.model.encoder,
         "alert_threshold": 0.5, "progression_states": list(C.PROGRESSION_STATES),
         "snapshots": 0, "classical_models": 0, "registry": ["SENTINEL-WM"],
         "system_members": ["tcn", "lstm", "gru"], "system_blend_weight": 0.5,
         "system_threshold": 0.5}))
    return str(dst)


@pytest.fixture()
def client(tiny_bundle, monkeypatch):
    monkeypatch.setenv("SENTINEL_WM_MODEL_DIR", tiny_bundle)
    monkeypatch.setenv("SENTINEL_MC_SAMPLES", "8")          # fast
    # rebuild the cached singletons against the test env
    import app.settings as S
    S.get_settings.cache_clear()
    S.settings = S.get_settings()
    import app.inference.loader as L
    L.get_engine.cache_clear()
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------------------
def _synth_flows(n_benign=260, burst=140, t0=1_700_000_000.0) -> pd.DataFrame:
    """steady benign traffic for ~300 s + a SYN scan burst from ~100-220 s."""
    rng = np.random.default_rng(7)
    rows = []
    # benign: ~1 flow/s, mixed ports, balanced fwd/bwd
    for i in range(n_benign):
        t = t0 + i * (300.0 / n_benign) + rng.uniform(0, 0.3)
        fp, bp = int(rng.integers(4, 40)), int(rng.integers(4, 40))
        rows.append(dict(
            flow_start_epoch=t, **{"Source IP": "10.0.0.9"},
            **{"Destination IP": f"10.0.1.{rng.integers(2, 60)}"},
            **{"Source Port": int(rng.integers(1024, 65000))},
            **{"Destination Port": int(rng.choice([80, 443, 53, 22, 8080]))},
            Protocol=6, **{"Flow Duration": float(rng.uniform(1e4, 5e6))},
            **{"Flow IAT Mean": float(rng.uniform(50, 5000))},
            **{"Total Fwd Packets": fp}, **{"Total Backward Packets": bp},
            **{"Total Length of Fwd Packets": fp * 500.0},
            **{"Total Length of Bwd Packets": bp * 600.0},
            **{"SYN Flag Count": 1}, **{"RST Flag Count": 0}, **{"ACK Flag Count": fp + bp}))
    # burst: many 1-2 packet SYN flows, one src -> sequential dst ports
    for j in range(burst):
        t = t0 + 100.0 + j * (120.0 / burst) + rng.uniform(0, 0.05)
        rows.append(dict(
            flow_start_epoch=t, **{"Source IP": "10.0.0.66"},
            **{"Destination IP": "10.0.1.5"},
            **{"Source Port": int(rng.integers(40000, 60000))},
            **{"Destination Port": int(1000 + j)},
            Protocol=6, **{"Flow Duration": float(rng.uniform(10, 500))},
            **{"Flow IAT Mean": float(rng.uniform(1, 50))},
            **{"Total Fwd Packets": 1}, **{"Total Backward Packets": 0},
            **{"Total Length of Fwd Packets": 40.0},
            **{"Total Length of Bwd Packets": 0.0},
            **{"SYN Flag Count": 1}, **{"RST Flag Count": 1}, **{"ACK Flag Count": 0}))
    df = pd.DataFrame(rows).sort_values("flow_start_epoch").reset_index(drop=True)
    return df


@pytest.fixture(scope="session")
def synth_csv_bytes() -> bytes:
    buf = io.BytesIO()
    _synth_flows().to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture()
def synth_flows_df() -> pd.DataFrame:
    return _synth_flows()
