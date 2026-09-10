"""Build the singleton inference engine from the model bundle.

Thin wrapper over `app.sentinel_infer` (the self-contained vendored inference
library). The backend depends on NOTHING in the research tree - only on the
bundle directory `settings.bundle_dir`.
"""
from __future__ import annotations

from functools import lru_cache

from app.sentinel_infer.forecast import (BundleContractError, BundleNotFound,
                                         InferenceEngine, load_bundle)
from app.settings import settings

__all__ = ["get_engine", "InferenceEngine", "BundleNotFound", "BundleContractError"]


@lru_cache
def get_engine() -> InferenceEngine:
    return load_bundle(str(settings.bundle_dir), device=settings.device,
                       serve_mode=settings.serve_mode,
                       mc_samples=settings.mc_samples)
