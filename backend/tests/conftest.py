"""Test fixtures.

`tiny_bundle` builds a throwaway model bundle from RANDOM weights (world model +
tcn/lstm/gru members) so the whole CSV / streaming plumbing is exercised without
a training run. Built entirely from `app.sentinel_infer` - the backend has no
dependency on the research tree.
"""
from __future__ import annotations

import io
import json
import pickle

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory) -> str:
    dst = tmp_path_factory.mktemp("bundle")
    import torch
    from sklearn.preprocessing import RobustScaler

    from app.sentinel_infer import schema as C
    from app.sentinel_infer.net import build_model, build_nn_model
    from app.sentinel_infer.windows import STATE_FEATURE_COLS

    F = len(STATE_FEATURE_COLS)
    C.CONFIG.sequence.horizon = 6
    wm = build_model(F, C.CONFIG)
    torch.save(dict(
        state_dict=wm.state_dict(),
        config={"n_features": F, "model": {"encoder": C.CONFIG.model.encoder}},
        sequence={"L": 12, "K": 6},
        alert_threshold=0.5, snapshots=[], self_ensemble=False),
        dst / "world_model.pt")

    sc = RobustScaler().fit(np.random.default_rng(0).normal(size=(512, F)))
    with open(dst / "state_scaler.pkl", "wb") as fh:
        pickle.dump({"scaler": sc, "feature_names": list(STATE_FEATURE_COLS)}, fh)

    nn_dir = dst / "models" / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("tcn", "lstm", "gru"):
        mm = build_nn_model(kind, F, 12, C.CONFIG)
        torch.save(dict(state_dict=mm.state_dict(), kind=kind, family="nn",
                        n_features=F, sequence={"L": 12, "K": 6}),
                   nn_dir / f"{kind}.pt")

    (dst / "bundle.json").write_text(json.dumps({
        "created": "test", "git_sha": "test", "L": 12, "K": 6, "n_features": F,
        "window_seconds": 10, "encoder": C.CONFIG.model.encoder,
        "progression_states": list(C.PROGRESSION_STATES),
        "feature_names": list(STATE_FEATURE_COLS),
        "system_members": ["tcn", "lstm", "gru"],
        "system_blend_weight": 0.5, "system_threshold": 0.5,
        "system_metrics": {"pr_auc": 0.99, "f1_best": 0.97},
    }))
    return str(dst)


@pytest.fixture()
def client(tiny_bundle, monkeypatch):
    monkeypatch.setenv("SENTINEL_WM_MODEL_DIR", tiny_bundle)
    monkeypatch.setenv("SENTINEL_MC_SAMPLES", "8")
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
    rng = np.random.default_rng(7)
    rows = []
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
    return pd.DataFrame(rows).sort_values("flow_start_epoch").reset_index(drop=True)


@pytest.fixture(scope="session")
def synth_csv_bytes() -> bytes:
    buf = io.BytesIO()
    _synth_flows().to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture()
def synth_flows_df() -> pd.DataFrame:
    return _synth_flows()
