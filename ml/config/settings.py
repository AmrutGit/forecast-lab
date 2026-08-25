"""Typed access to config.yaml. Import `CONFIG` from this module everywhere
instead of re-reading or hardcoding config values."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class PathsConfig:
    raw_data: Path
    processed_dir: Path
    train_features: Path
    val_features: Path
    test_features: Path
    lookup_artifacts: Path
    eda_report_dir: Path
    model_registry_dir: Path
    mlruns_dir: Path


@dataclass(frozen=True)
class DataGenerationConfig:
    seed: int
    n_rows: int
    start_date: str
    end_date: str
    output_path: Path


@dataclass(frozen=True)
class SplitConfig:
    train_end_date: str
    val_end_date: str


@dataclass(frozen=True)
class FeaturesConfig:
    rolling_windows_days: list[int]
    min_history_days: int


@dataclass(frozen=True)
class TrainingConfig:
    target: str
    model_type: str
    cv_folds: int
    n_search_trials: int
    early_stopping_rounds: int
    random_state: int


@dataclass(frozen=True)
class RegistryConfig:
    keep_last_n_versions: int
    metadata_file: Path


@dataclass(frozen=True)
class ServingConfig:
    top_k: int


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    data_generation: DataGenerationConfig
    split: SplitConfig
    features: FeaturesConfig
    training: TrainingConfig
    registry: RegistryConfig
    serving: ServingConfig


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: Path | None = None) -> Config:
    raw = _load_yaml(path or CONFIG_PATH)

    paths_raw = raw["paths"]
    paths = PathsConfig(
        raw_data=_resolve(paths_raw["raw_data"]),
        processed_dir=_resolve(paths_raw["processed_dir"]),
        train_features=_resolve(paths_raw["train_features"]),
        val_features=_resolve(paths_raw["val_features"]),
        test_features=_resolve(paths_raw["test_features"]),
        lookup_artifacts=_resolve(paths_raw["lookup_artifacts"]),
        eda_report_dir=_resolve(paths_raw["eda_report_dir"]),
        model_registry_dir=_resolve(paths_raw["model_registry_dir"]),
        mlruns_dir=_resolve(paths_raw["mlruns_dir"]),
    )

    dg_raw = raw["data_generation"]
    data_generation = DataGenerationConfig(
        seed=dg_raw["seed"],
        n_rows=dg_raw["n_rows"],
        start_date=dg_raw["start_date"],
        end_date=dg_raw["end_date"],
        output_path=_resolve(dg_raw["output_path"]),
    )

    split = SplitConfig(**raw["split"])
    features = FeaturesConfig(**raw["features"])
    training = TrainingConfig(**raw["training"])

    registry_raw = raw["registry"]
    registry = RegistryConfig(
        keep_last_n_versions=registry_raw["keep_last_n_versions"],
        metadata_file=_resolve(registry_raw["metadata_file"]),
    )

    serving = ServingConfig(**raw["serving"])

    return Config(
        paths=paths,
        data_generation=data_generation,
        split=split,
        features=features,
        training=training,
        registry=registry,
        serving=serving,
    )


# Allow overriding the config file location via env var (useful for tests).
CONFIG = load_config(Path(os.environ["FORECAST_LAB_CONFIG"]) if os.environ.get("FORECAST_LAB_CONFIG") else None)
