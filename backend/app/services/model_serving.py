"""Serving-layer model access.

Loads the currently active model version (per the file-based registry) and
its paired lookup artifacts, and builds prediction responses purely from
those persisted artifacts. This module never re-runs feature engineering
from raw data and never retrains — that separation is what keeps serving
fast and keeps serving-time features identical to training-time features
(see ml/pipeline/features.py's LookupArtifacts).

The loaded model/lookup pair is cached in-process and only reloaded when the
registry's active version changes (checked on each request via a cheap
metadata read, not a full artifact reload).
"""

from __future__ import annotations

import threading

import joblib
import pandas as pd

from backend.app.core.config import KEEP_LAST_N_VERSIONS, MODEL_REGISTRY_DIR, REGISTRY_METADATA_FILE
from ml.config.taxonomy import CATEGORIES, CATEGORY_ATTRIBUTES, REGIONS
from ml.pipeline.drift import compute_drift_report
from ml.pipeline.explainability import top_feature_contributions
from ml.pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from ml.pipeline.registry import ModelRegistry

SHAP_TOP_N_FACTORS = 3


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained model version exists in the registry yet."""


class ServedModel:
    """Wraps a loaded (QuantileModel, lookup_artifacts) pair for one registry
    version."""

    def __init__(self, version: str, model, lookup_artifacts, metadata: dict):
        self.version = version
        self.model = model
        self.lookup_artifacts = lookup_artifacts
        self.metadata = metadata

    def predict_group(self, region: str, category: str, attribute_type: str) -> list[dict]:
        """Predict units_sold (p10/p50/p90) for every attribute_value of the
        given (region, category, attribute_type), using each group's most
        recent persisted feature snapshot (never recomputed ad hoc), plus a
        top-factor SHAP breakdown per prediction."""
        values = CATEGORY_ATTRIBUTES.get(category, {}).get(attribute_type, [])
        rows = []
        for value in values:
            snapshot = self.lookup_artifacts.get_features_for_group(region, category, attribute_type, value)
            if snapshot is None:
                continue
            rows.append({"attribute_value": value, **snapshot})

        if not rows:
            return []

        df = pd.DataFrame(rows)
        X = df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS].copy()
        for col in CATEGORICAL_COLUMNS:
            X[col] = X[col].astype("category")

        quantile_preds = self.model.predict(X)
        df["predicted_units"] = quantile_preds["p50"]
        df["predicted_units_low"] = quantile_preds["p10"]
        df["predicted_units_high"] = quantile_preds["p90"]
        df["historical_avg_units"] = df["rolling_mean_12"]

        top_factors = top_feature_contributions(
            self.model.primary, X, top_n=SHAP_TOP_N_FACTORS, reportable_columns=FEATURE_COLUMNS
        )
        df["top_factors"] = top_factors

        df = df.sort_values("predicted_units", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        columns = [
            "attribute_value",
            "predicted_units",
            "predicted_units_low",
            "predicted_units_high",
            "historical_avg_units",
            "rank",
            "top_factors",
        ]
        return df[columns].to_dict("records")

    def drift_report(self, current_df: pd.DataFrame | None = None):
        """Compare the training-time reference distribution (bundled in the
        lookup artifact) against a "current" sample — defaults to the
        lookup's own latest-snapshot rows, which stand in for the most
        recent known feature values per group."""
        current = current_df if current_df is not None else self.lookup_artifacts.latest_by_group
        return compute_drift_report(self.lookup_artifacts.drift_reference, current)


class ModelServer:
    """Thread-safe accessor for the currently active served model, backed by
    the file-based ModelRegistry. Lazily (re)loads artifacts when the active
    registry version changes."""

    def __init__(self):
        self._registry = ModelRegistry(
            registry_dir=MODEL_REGISTRY_DIR,
            metadata_file=REGISTRY_METADATA_FILE,
            keep_last_n=KEEP_LAST_N_VERSIONS,
        )
        self._lock = threading.Lock()
        self._served: ServedModel | None = None

    def _load(self, metadata: dict) -> ServedModel:
        model = joblib.load(metadata["model_path"])
        lookup_artifacts = joblib.load(metadata["lookup_artifacts_path"])
        return ServedModel(version=metadata["version"], model=model, lookup_artifacts=lookup_artifacts, metadata=metadata)

    def get_active(self) -> ServedModel:
        active_meta = self._registry.get_active()
        if active_meta is None:
            raise ModelNotAvailableError(
                "No trained model is registered yet. Run the training pipeline "
                "(python -m ml.pipeline.run_training) before serving predictions."
            )

        with self._lock:
            if self._served is None or self._served.version != active_meta["version"]:
                self._served = self._load(active_meta)
            return self._served

    def list_versions(self) -> list[dict]:
        return self._registry.list_versions()

    def valid_region(self, region: str) -> bool:
        return region in REGIONS

    def valid_category(self, category: str) -> bool:
        return category in CATEGORIES

    def attribute_types_for_category(self, category: str) -> list[str]:
        return list(CATEGORY_ATTRIBUTES.get(category, {}).keys())


# Module-level singleton, mirroring how a real deployment would keep one
# warm in-process model server per worker process.
model_server = ModelServer()
