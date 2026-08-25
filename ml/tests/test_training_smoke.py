"""Smoke test: train on a tiny fixture dataset end-to-end and check the model
loads and predicts. This is not testing model quality — it's testing that the
full train -> save -> load -> predict path doesn't break.
"""

from __future__ import annotations

import joblib

from ml.data_generator.generate import generate_dataset
from ml.pipeline.data_loading import aggregate_to_weekly
from ml.pipeline.evaluation import evaluate_predictions
from ml.pipeline.features import (
    build_lookup_artifacts,
    engineer_features,
)
from ml.pipeline.splitting import time_based_split
from ml.pipeline.training import (
    prepare_feature_matrix,
    train_final_model,
)


def _tiny_featured_dataset():
    # Small but wide-enough synthetic slice: 1.5 years so lag_52 and yearly
    # seasonality features have at least one full cycle of history.
    raw = generate_dataset(
        n_rows=6000, start_date="2023-01-01", end_date="2024-06-30", seed=123
    )
    weekly = aggregate_to_weekly(raw)
    return engineer_features(weekly)


def test_train_predict_roundtrip(tmp_path):
    featured = _tiny_featured_dataset()

    train_df, val_df, test_df = time_based_split(
        featured, train_end_date="2023-12-31", val_end_date="2024-03-31"
    )

    params = {
        "objective": "regression",
        "random_state": 42,
        "verbosity": -1,
        "n_estimators": 50,
        "learning_rate": 0.1,
        "num_leaves": 15,
    }
    model = train_final_model(train_df, val_df, target="units_sold", params=params, early_stopping_rounds=10)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)

    X_test = prepare_feature_matrix(test_df)
    preds = loaded_model.predict(X_test, num_iteration=loaded_model.best_iteration_)

    assert len(preds) == len(test_df)
    assert not any(p != p for p in preds)  # no NaNs (p != p is true only for NaN)

    test_df = test_df.copy()
    test_df["predicted_units"] = preds
    metrics = evaluate_predictions(test_df, y_true_col="units_sold", y_pred_col="predicted_units", k=5)
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_lookup_artifacts_roundtrip(tmp_path):
    featured = _tiny_featured_dataset()
    lookup = build_lookup_artifacts(featured)

    path = tmp_path / "lookup.joblib"
    joblib.dump(lookup, path)
    loaded = joblib.load(path)

    any_row = featured.iloc[0]
    result = loaded.get_features_for_group(
        any_row["region_code"],
        any_row["product_category"],
        any_row["product_attribute_type"],
        any_row["product_attribute_value"],
    )
    assert result is not None
