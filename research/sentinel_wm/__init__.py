"""
SENTINEL-WM — Spatio-temporal Enemy Network Intelligence with a Learned World Model.

A fully-offline research prototype that learns 10-second network-state transition
dynamics from CIC-IDS-2017 flow telemetry and forecasts attacker progression K
windows ahead, with calibrated probabilities, MITRE ATT&CK phase mapping (with
explicit confidence), and per-prediction explainability.

Pipeline modules (run in order, or via `python -m sentinel_wm.cli all`):

    config          hyper-parameters, paths, locked feature tiers
    preprocessing   PHASE 1  raw unified CSV  -> artifacts/clean_flows.parquet
    state_windows   PHASE 2  clean flows      -> artifacts/state_windows.parquet
    attack_stages   MITRE ATT&CK phase mapping (Layer A static + Layer B context)
    sequences       PHASE 2c windows          -> artifacts/sequences.npz (+ scaler)
    baselines       PHASE 3  LogisticRegression / RandomForest comparison floor
    models          the world model: Temporal Transformer + probabilistic STN + heads
    train           PHASE 4  train / --test the world model
    forward_sim     PHASE 5  K-step Monte-Carlo forward simulation
    explain         PHASE 6  SHAP (or fallback) + attention + gradient saliency
    evaluate        benchmark table: world model vs baselines
    metrics         shared metric functions
    cli             offline command-line interface
"""

__version__ = "0.1.0"
__all__ = [
    "config", "preprocessing", "state_windows", "attack_stages", "sequences",
    "baselines", "metrics", "models", "train", "forward_sim", "explain",
    "evaluate", "cli",
]
