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

    # path to the model bundle (`sentinel-wm bundle`). Accepts SENTINEL_WM_MODEL_DIR.
    bundle_dir: Path = Field(default=_HERE.parent / "models",
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
