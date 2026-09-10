"""Response models. `extra="allow"` on the forecast bodies so a change in
`forward_sim.simulate_anchor` never breaks the API contract."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class AttckAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")
    progression_state: str
    mitre_tactic: str
    technique_ids: list[str] = []
    kill_chain_phase: str
    confidence: str
    rationale: str = ""
    dominant_family: str = "BENIGN"
    family_transition: Optional[str] = None


class HorizonStep(BaseModel):
    model_config = ConfigDict(extra="allow")
    k: int
    horizon_seconds: int
    attack_prob: float                    # world-model rollout P(attack) (has the CI)
    attack_ci: list[float]
    attack_std: float
    detection_prob: Optional[float] = None  # SENTINEL-WM (system) blended prob (drives the alert)
    progression_state: str
    progression_dist: dict[str, float]
    attck: AttckAssessment


class AnchorForecast(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: dict[str, Any] = {}
    alert_threshold: float
    alert: bool
    lead_time_seconds: int
    first_alert_k: Optional[int] = None
    max_attack_prob: float
    max_detection_prob: Optional[float] = None
    detection_model: Optional[str] = None   # "SENTINEL-WM (system)" when the blend is active
    horizon: list[HorizonStep]
    driving_features: Optional[dict[str, Any]] = None


class ForecastSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    n_alerts: int
    max_attack_prob: float
    phases: list[str] = []
    alert_threshold: Optional[float] = None


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: dict[str, Any]
    summary: ForecastSummary
    anchors: list[AnchorForecast]


class JobCreated(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    id: str
    kind: str
    status: str                       # queued | running | done | error
    progress: float = 0.0
    error: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None
    has_result: bool = False


class MetaResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    history_windows: int
    horizon_steps: int
    feature_dim: int
    window_seconds: int
    progression_states: list[str]
    feature_names: list[str]
    alert_threshold: float
    device: str
    serve_mode: str = "world_model"        # "system" | "world_model"
    blend_members: list[str] = []
    blend_weight: Optional[float] = None
    bundle: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None
