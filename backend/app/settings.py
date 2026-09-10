"""Backend configuration.

This module has NO `sentinel_wm` import and MUST be imported before anything that
touches `sentinel_wm.config` - it exports `SENTINEL_WM_MODEL_DIR` into the
environment so the research package loads models from the bundle, not the tree.
"""
from __future__ import annotations

import os
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

    # accepts SENTINEL_WM_MODEL_DIR (canonical) or SENTINEL_MODEL_DIR
    bundle_dir: Path = Field(default=_HERE.parent / "models",
                             validation_alias="SENTINEL_WM_MODEL_DIR")
    device: str = "cpu"
    # "auto" (system blend if the bundle has the member models, else world_model),
    # "system" (force blend), "world_model" (world model only)
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
    # hand the bundle path to sentinel_wm.config BEFORE it is imported
    os.environ["SENTINEL_WM_MODEL_DIR"] = str(s.bundle_dir)
    return s


settings = get_settings()
