from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.pipeline.evaluation import (
    compute_ndcg_at_k,
    compute_regression_metrics,
    evaluate_predictions,
)


def test_regression_metrics_perfect_prediction_is_zero():
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_ndcg_perfect_ranking_scores_one():
    df = pd.DataFrame(
        {
            "region_code": ["Region A"] * 4,
            "product_category": ["Outerwear"] * 4,
            "product_attribute_type": ["material"] * 4,
            "week_start": [pd.Timestamp("2024-01-01")] * 4,
            "units_sold": [40, 30, 20, 10],
            "predicted": [40, 30, 20, 10],
        }
    )
    result = compute_ndcg_at_k(df, "units_sold", "predicted", k=5)
    assert result["ndcg_at_5"] == pytest.approx(1.0)
    assert result["ndcg_at_5_n_groups"] == 1


def test_ndcg_worst_ranking_scores_below_one():
    df = pd.DataFrame(
        {
            "region_code": ["Region A"] * 4,
            "product_category": ["Outerwear"] * 4,
            "product_attribute_type": ["material"] * 4,
            "week_start": [pd.Timestamp("2024-01-01")] * 4,
            "units_sold": [40, 30, 20, 10],
            "predicted": [10, 20, 30, 40],  # exactly reversed
        }
    )
    result = compute_ndcg_at_k(df, "units_sold", "predicted", k=5)
    assert result["ndcg_at_5"] < 1.0


def test_ndcg_skips_single_item_groups():
    df = pd.DataFrame(
        {
            "region_code": ["Region A"],
            "product_category": ["Outerwear"],
            "product_attribute_type": ["material"],
            "week_start": [pd.Timestamp("2024-01-01")],
            "units_sold": [40],
            "predicted": [40],
        }
    )
    result = compute_ndcg_at_k(df, "units_sold", "predicted", k=5)
    assert result["ndcg_at_5_n_groups"] == 0
    assert np.isnan(result["ndcg_at_5"])


def test_evaluate_predictions_returns_all_metrics():
    df = pd.DataFrame(
        {
            "region_code": ["Region A"] * 4,
            "product_category": ["Outerwear"] * 4,
            "product_attribute_type": ["material"] * 4,
            "week_start": [pd.Timestamp("2024-01-01")] * 4,
            "units_sold": [40, 30, 20, 10],
            "predicted": [38, 31, 22, 9],
        }
    )
    result = evaluate_predictions(df, "units_sold", "predicted", k=5)
    assert set(result.keys()) == {"mae", "rmse", "ndcg_at_5", "ndcg_at_5_n_groups"}
