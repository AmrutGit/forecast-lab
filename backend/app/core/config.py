"""Backend service configuration. Reuses ml.config for paths shared with the
training pipeline (registry dir, metadata file) so the serving layer and
training pipeline never disagree about where artifacts live."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.config.settings import CONFIG as ML_CONFIG


class BackendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORECAST_LAB_API_")

    cors_allow_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    top_k: int = ML_CONFIG.serving.top_k


settings = BackendSettings()

MODEL_REGISTRY_DIR = ML_CONFIG.paths.model_registry_dir
REGISTRY_METADATA_FILE = ML_CONFIG.registry.metadata_file
KEEP_LAST_N_VERSIONS = ML_CONFIG.registry.keep_last_n_versions
