"""Model training: LightGBM regressor with Optuna hyperparameter search and
time-series cross-validation, tracked via MLflow.

Cross-validation uses expanding-window time splits *within the training set
only* (never touching val/test) so hyperparameter selection doesn't leak
future information either.
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from ml.pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas 'category' dtype for LightGBM's
    native categorical handling, and select the model feature columns."""
    X = df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype("category")
    return X


def _cv_score(params: dict, X: pd.DataFrame, y: pd.Series, n_splits: int, early_stopping_rounds: int) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_maes = []

    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="mae",
            categorical_feature=CATEGORICAL_COLUMNS,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        preds = model.predict(X_valid, num_iteration=model.best_iteration_)
        fold_maes.append(mean_absolute_error(y_valid, preds))

    return float(np.mean(fold_maes))


def search_hyperparameters(
    train_df: pd.DataFrame,
    target: str,
    cv_folds: int,
    n_trials: int,
    early_stopping_rounds: int,
    random_state: int,
) -> dict:
    """Optuna search over LightGBM hyperparameters, scored by expanding-window
    CV MAE on the training set."""
    X = prepare_feature_matrix(train_df)
    y = train_df[target]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "random_state": random_state,
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "verbosity": -1,
        }
        return _cv_score(params, X, y, cv_folds, early_stopping_rounds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update({"objective": "regression", "random_state": random_state, "verbosity": -1})
    return best_params, study.best_value


def train_final_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    params: dict,
    early_stopping_rounds: int,
) -> lgb.LGBMRegressor:
    """Fit the final model on the training set, using the validation set only
    for early stopping (not for hyperparameter selection, which already
    happened via CV on the training set alone)."""
    X_train = prepare_feature_matrix(train_df)
    y_train = train_df[target]
    X_val = prepare_feature_matrix(val_df)
    y_val = val_df[target]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def log_training_run(
    mlflow_tracking_uri: str,
    experiment_name: str,
    params: dict,
    cv_mae: float,
    metrics: dict,
    model: lgb.LGBMRegressor,
    data_version: str,
) -> str:
    """Log params/metrics/model to MLflow and return the run_id."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("data_version", data_version)
        mlflow.log_metric("cv_mae", cv_mae)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.lightgbm.log_model(model, artifact_path="model")
        return run.info.run_id
