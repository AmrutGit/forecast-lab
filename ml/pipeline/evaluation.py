"""Evaluation metrics: MAE, RMSE (regression accuracy) and NDCG@5 (ranking
quality within each region/category/attribute_type group), as justified in
docs/problem_framing.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANKING_GROUP_KEYS = ["region_code", "product_category", "product_attribute_type", "week_start"]


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def _dcg_at_k(relevance: np.ndarray, k: int) -> float:
    relevance = relevance[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance / discounts))


def _ndcg_at_k_for_group(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float | None:
    """NDCG@k for one ranking group. Returns None if the group has fewer than
    2 items (ranking is trivial/undefined)."""
    if len(y_true) < 2:
        return None

    pred_order = np.argsort(-y_pred)
    predicted_relevance = y_true[pred_order]
    dcg = _dcg_at_k(predicted_relevance, k)

    ideal_relevance = np.sort(y_true)[::-1]
    idcg = _dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return None

    return dcg / idcg


def compute_ndcg_at_k(df: pd.DataFrame, y_true_col: str, y_pred_col: str, k: int = 5) -> dict:
    """Mean NDCG@k across all (region, category, attribute_type, week)
    groups that have at least 2 candidate attribute values to rank.

    Relevance is the raw units_sold value (graded relevance, not binary),
    which is standard for demand-ranking NDCG.
    """
    scores = []
    for _, group in df.groupby(RANKING_GROUP_KEYS):
        score = _ndcg_at_k_for_group(
            group[y_true_col].to_numpy(dtype=np.float64),
            group[y_pred_col].to_numpy(dtype=np.float64),
            k,
        )
        if score is not None:
            scores.append(score)

    if not scores:
        return {f"ndcg_at_{k}": float("nan"), f"ndcg_at_{k}_n_groups": 0}

    return {f"ndcg_at_{k}": float(np.mean(scores)), f"ndcg_at_{k}_n_groups": len(scores)}


def evaluate_predictions(df: pd.DataFrame, y_true_col: str, y_pred_col: str, k: int = 5) -> dict:
    """Full metric suite used for test-set evaluation and MLflow logging."""
    metrics = compute_regression_metrics(df[y_true_col], df[y_pred_col].to_numpy())
    metrics.update(compute_ndcg_at_k(df, y_true_col, y_pred_col, k=k))
    return metrics
