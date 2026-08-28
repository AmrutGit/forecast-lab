from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.pipeline.explainability import top_feature_contributions


def _toy_model_and_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, size=n),
            "feature_b": rng.normal(0, 1, size=n),
            "feature_c": rng.normal(0, 1, size=n),
        }
    )
    # y depends strongly on feature_a, weakly on the others — so SHAP should
    # consistently rank feature_a as the top contributor.
    y = 10 * X["feature_a"] + 0.01 * X["feature_b"] + rng.normal(0, 0.1, size=n)

    model = lgb.LGBMRegressor(n_estimators=50, verbosity=-1, random_state=0)
    model.fit(X, y)
    return model, X


def test_top_feature_contributions_shape_matches_input():
    model, X = _toy_model_and_data()
    contributions = top_feature_contributions(model, X, top_n=2)

    assert len(contributions) == len(X)
    for row_contributions in contributions:
        assert len(row_contributions) == 2
        for entry in row_contributions:
            assert set(entry.keys()) == {"feature", "value", "impact"}


def test_dominant_feature_is_ranked_first_for_most_rows():
    model, X = _toy_model_and_data()
    contributions = top_feature_contributions(model, X, top_n=1)

    top_feature_counts = pd.Series([row[0]["feature"] for row in contributions]).value_counts()
    assert top_feature_counts.idxmax() == "feature_a"
    # feature_a should dominate the vast majority of rows, given its much
    # larger coefficient in the data-generating process.
    assert top_feature_counts["feature_a"] / len(X) > 0.8


def test_impact_values_are_floats_and_json_safe():
    model, X = _toy_model_and_data()
    contributions = top_feature_contributions(model, X, top_n=3)

    for row in contributions:
        for entry in row:
            assert isinstance(entry["impact"], float)


def test_reportable_columns_filters_out_excluded_features():
    model, X = _toy_model_and_data()
    # Exclude the dominant feature from reporting — SHAP is still computed
    # against the full X (required by LightGBM's contribution decomposition),
    # but "feature_a" must never appear in the returned top factors.
    contributions = top_feature_contributions(model, X, top_n=2, reportable_columns=["feature_b", "feature_c"])

    for row in contributions:
        reported = {entry["feature"] for entry in row}
        assert "feature_a" not in reported
        assert reported <= {"feature_b", "feature_c"}
