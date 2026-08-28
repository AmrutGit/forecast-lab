"""Model training: LightGBM quantile regressors with Optuna hyperparameter
search and time-series cross-validation, tracked via MLflow.

We train three LightGBM models per version — p10, p50 (median), p90 — rather
than a single point-estimate regressor. p50 is used as the primary
prediction everywhere a single number is needed (ranking, MAE/RMSE/NDCG@5
evaluation), so those metrics are unchanged in meaning (now explicitly
"error against the median" rather than "error against a point estimate" —
a standard and defensible substitution). p10/p90 give the product a
prediction *interval* instead of a bare number, which matters for a
stocking decision: "40 units, could be 25-60" is a materially different
signal than "40 units" alone.

Cross-validation uses expanding-window time splits *within the training set
only* (never touching val/test) so hyperparameter selection doesn't leak
future information either. The hyperparameter search optimizes only the p50
model's CV MAE — p10/p90 reuse the same tree structure params (num_leaves,
max_depth, etc.) with just the quantile `alpha` swapped, which is standard
practice (searching three independent hyperparameter spaces would triple
the search cost for marginal gain, since the three quantiles share the same
underlying feature relationships).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from ml.pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

optuna.logging.set_verbosity(optuna.logging.WARNING)

QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
PRIMARY_QUANTILE = "p50"


@dataclass
class QuantileModel:
    """The three quantile regressors trained together for one model version.
    Serialized as a single artifact so serving always loads a matched triple
    (never a p50 from one training run paired with a stale p10/p90)."""

    models: dict[str, lgb.LGBMRegressor]  # keys: "p10", "p50", "p90"

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            name: model.predict(X, num_iteration=getattr(model, "best_iteration_", None))
            for name, model in self.models.items()
        }

    @property
    def primary(self) -> lgb.LGBMRegressor:
        return self.models[PRIMARY_QUANTILE]


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
) -> QuantileModel:
    """Fit the final p10/p50/p90 quantile models on the training set, using
    the validation set only for early stopping (not for hyperparameter
    selection, which already happened via CV on the training set alone).

    All three models share the tree-structure hyperparameters found by
    `search_hyperparameters` (which searched using a plain regression
    objective) — only `objective`/`alpha` differ per quantile. This is the
    standard shortcut for multi-quantile GBMs: the same splits that predict
    the conditional mean well are a good starting point for the conditional
    quantiles, so a full independent search per quantile isn't worth 3x the
    search cost.
    """
    X_train = prepare_feature_matrix(train_df)
    y_train = train_df[target]
    X_val = prepare_feature_matrix(val_df)
    y_val = val_df[target]

    base_params = {k: v for k, v in params.items() if k not in ("objective", "alpha")}

    models: dict[str, lgb.LGBMRegressor] = {}
    for name, alpha in QUANTILES.items():
        model = lgb.LGBMRegressor(**base_params, objective="quantile", alpha=alpha)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            categorical_feature=CATEGORICAL_COLUMNS,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        models[name] = model

    return QuantileModel(models=models)


def log_training_run(
    mlflow_tracking_uri: str,
    experiment_name: str,
    params: dict,
    cv_mae: float,
    metrics: dict,
    model: QuantileModel,
    data_version: str,
) -> str:
    """Log params/metrics/all three quantile models to MLflow and return the
    run_id."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("data_version", data_version)
        mlflow.log_metric("cv_mae", cv_mae)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        for quantile_name, quantile_model in model.models.items():
            mlflow.lightgbm.log_model(quantile_model, artifact_path=f"model_{quantile_name}")
        return run.info.run_id
