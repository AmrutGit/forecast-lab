"""End-to-end training pipeline entrypoint.

    python -m ml.pipeline.run_training

Stages: load raw data -> aggregate to weekly -> engineer features ->
time-based split -> hyperparameter search + CV (train only) -> fit final
model (train+val) -> evaluate on held-out test -> log to MLflow -> persist
versioned model + lookup artifacts to the registry.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import joblib

from ml.config.settings import CONFIG
from ml.pipeline.data_loading import aggregate_to_weekly, load_raw_orders
from ml.pipeline.evaluation import evaluate_predictions
from ml.pipeline.features import (
    build_lookup_artifacts,
    engineer_features,
    save_lookup_artifacts,
)
from ml.pipeline.registry import ModelRegistry, ModelVersionMetadata
from ml.pipeline.splitting import time_based_split
from ml.pipeline.training import (
    log_training_run,
    prepare_feature_matrix,
    search_hyperparameters,
    train_final_model,
)


def _data_version(raw_path) -> str:
    """Content hash of the raw data file, used to detect which data a model
    version was trained on — independent of the file's mtime or path."""
    hasher = hashlib.sha256()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def run_training_pipeline() -> ModelVersionMetadata:
    cfg = CONFIG

    print("Loading raw orders...")
    raw = load_raw_orders(cfg.paths.raw_data)
    data_version = _data_version(cfg.paths.raw_data)

    print("Aggregating to weekly grain...")
    weekly = aggregate_to_weekly(raw)

    print("Engineering features...")
    featured = engineer_features(weekly)

    print("Splitting train/val/test by time...")
    train_df, val_df, test_df = time_based_split(
        featured, cfg.split.train_end_date, cfg.split.val_end_date
    )
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(cfg.paths.train_features, index=False)
    val_df.to_parquet(cfg.paths.val_features, index=False)
    test_df.to_parquet(cfg.paths.test_features, index=False)

    print(f"Searching hyperparameters ({cfg.training.n_search_trials} trials, {cfg.training.cv_folds}-fold CV)...")
    best_params, cv_mae = search_hyperparameters(
        train_df,
        target=cfg.training.target,
        cv_folds=cfg.training.cv_folds,
        n_trials=cfg.training.n_search_trials,
        early_stopping_rounds=cfg.training.early_stopping_rounds,
        random_state=cfg.training.random_state,
    )
    print(f"  best CV MAE: {cv_mae:.3f}")
    print(f"  best params: {best_params}")

    print("Training final model on train+val (early stopping on val)...")
    model = train_final_model(
        train_df, val_df, cfg.training.target, best_params, cfg.training.early_stopping_rounds
    )

    print("Evaluating on held-out test set...")
    X_test = prepare_feature_matrix(test_df)
    test_df = test_df.copy()
    test_df["predicted_units"] = model.predict(X_test, num_iteration=model.best_iteration_)
    metrics = evaluate_predictions(test_df, y_true_col=cfg.training.target, y_pred_col="predicted_units", k=5)
    print(f"  test metrics: {metrics}")

    print("Building serving-time lookup artifacts...")
    lookup = build_lookup_artifacts(featured)

    version = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cfg.paths.model_registry_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.paths.model_registry_dir / f"model_{version}.joblib"
    lookup_path = cfg.paths.model_registry_dir / f"lookup_{version}.joblib"

    joblib.dump(model, model_path)
    save_lookup_artifacts(lookup, lookup_path)

    print("Logging run to MLflow...")
    run_id = log_training_run(
        mlflow_tracking_uri=f"file:{cfg.paths.mlruns_dir}",
        experiment_name="forecast_lab_demand",
        params=best_params,
        cv_mae=cv_mae,
        metrics=metrics,
        model=model,
        data_version=data_version,
    )

    metadata = ModelVersionMetadata(
        version=version,
        model_path=str(model_path),
        lookup_artifacts_path=str(lookup_path),
        mlflow_run_id=run_id,
        data_version=data_version,
        trained_at=dt.datetime.utcnow().isoformat() + "Z",
        metrics=metrics,
        params=best_params,
    )

    registry = ModelRegistry(
        registry_dir=cfg.paths.model_registry_dir,
        metadata_file=cfg.registry.metadata_file,
        keep_last_n=cfg.registry.keep_last_n_versions,
    )
    registry.register(metadata)

    print(f"Registered model version {version} (mlflow run {run_id})")
    return metadata


if __name__ == "__main__":
    run_training_pipeline()
