"""Backend configuration. No dependency on the research tree - the model bundle
directory is the only interface (see app.sentinel_infer)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve().parent.parent          # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_HERE / ".env"), env_prefix="SENTINEL_", extra="ignore",
        protected_namespaces=(),
    )

    # path to the model bundle (`sentinel-wm bundle`). Ships inside backend/ so
    # the service is self-contained (code + models). Accepts SENTINEL_WM_MODEL_DIR.
    bundle_dir: Path = Field(default=_HERE / "models",
                             validation_alias="SENTINEL_WM_MODEL_DIR")
    device: str = "cpu"
    # "auto" (system blend if the bundle has the members, else world_model) |
    # "system" (force blend) | "world_model" (world model only)
    serve_mode: str = "auto"
    max_sync_flows: int = 20_000
    mc_samples: int = 50
    workers: int = 2
    job_dir: Path = _HERE / "data" / "jobs"
    stream_grace_seconds: float = 3.0
    cors_origins: str = "*"

    # --- real-time (live sessions, synthetic test-bed, capture agent) ---
    live_max_sessions: int = Field(default=4, validation_alias="SENTINEL_LIVE_MAX_SESSIONS")
    live_idle_timeout_s: int = Field(default=900, validation_alias="SENTINEL_LIVE_IDLE_TIMEOUT_S")
    synth_max_rate: int = Field(default=500, validation_alias="SENTINEL_SYNTH_MAX_RATE")
    agent_enabled: bool = Field(default=True, validation_alias="SENTINEL_AGENT_ENABLED")
    log_json: bool = Field(default=False, validation_alias="SENTINEL_LOG_JSON")
    metrics: bool = Field(default=False, validation_alias="SENTINEL_METRICS")

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.bundle_dir = s.bundle_dir.resolve()
    s.job_dir = s.job_dir.resolve()
    s.job_dir.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
