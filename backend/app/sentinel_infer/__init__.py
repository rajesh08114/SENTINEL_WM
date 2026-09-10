"""sentinel_infer - self-contained SENTINEL-WM inference library.

Vendored from `research/sentinel_wm/` so the backend depends on NOTHING in the
research tree - only on a model bundle directory (`SENTINEL_WM_MODEL_DIR`).

Contract: the bundle's `bundle.json["feature_names"]` must equal
`windows.STATE_FEATURE_COLS` here. `forecast.load_world_model` enforces it and
fails loudly on drift - re-vendor (see `README.md`) if it does.

Modules
  schema        LOCKED feature schema + slim inference config (no paths, no I/O)
  attack_stages MITRE ATT&CK phase mapping  (verbatim copy - pure python)
  preprocess    clean_flow_frame  (label/day/numeric normalisation, winsor)
  windows       build_state_windows  (10-s state-vector aggregation, in memory)
  net           SentinelWorldModel + STN + encoders + the TCN/LSTM/GRU members
  explain       per-anchor gradient x input + attention saliency
  forecast      load_bundle -> InferenceEngine -> forecast(flows_df) -> JSON
"""
__version__ = "0.1.0"
