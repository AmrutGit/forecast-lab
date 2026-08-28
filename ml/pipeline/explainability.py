"""Per-prediction explainability via SHAP.

Uses shap.TreeExplainer against the p50 (median) quantile model — exact and
fast for tree ensembles, no additional training or approximation needed. This
is computed at serving time, on demand, per prediction request; it is not
part of the training pipeline because SHAP values only need to be produced
for predictions a user actually asks to see, not precomputed for every
group in the catalog.
"""

from __future__ import annotations

import shap
import pandas as pd


def top_feature_contributions(
    model, X: pd.DataFrame, top_n: int = 3, reportable_columns: list[str] | None = None
) -> list[list[dict]]:
    """For each row in X, return the top_n features (by absolute SHAP value)
    that pushed that row's prediction up or down, as
    [{"feature": ..., "value": ..., "impact": ...}, ...] per row.

    `X` must be the exact feature matrix the model predicts on (same columns,
    order, and dtypes it was fit with, categoricals included) — LightGBM's
    contribution decomposition requires the full matrix even though we may
    only want to *report* a subset of features (`reportable_columns`, e.g.
    excluding raw identity columns like region_code/product_category from
    the reported "why" — a buyer doesn't need to be told "this is Region A"
    as an explanation for a Region A prediction).

    `impact` is the signed SHAP value in the model's output units (units of
    `units_sold`), so "impact: +4.2" reads as "this feature added ~4 units
    to the prediction relative to the model's baseline."
    """
    reportable = set(reportable_columns) if reportable_columns is not None else set(X.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    contributions = []
    for row_idx in range(len(X)):
        row_shap = shap_values[row_idx]
        row_values = X.iloc[row_idx]

        candidates = [
            (feature, impact)
            for feature, impact in zip(X.columns, row_shap, strict=True)
            if feature in reportable
        ]
        ranked = sorted(candidates, key=lambda item: abs(item[1]), reverse=True)[:top_n]

        contributions.append(
            [
                {
                    "feature": feature,
                    "value": _json_safe(row_values[feature]),
                    "impact": round(float(impact), 3),
                }
                for feature, impact in ranked
            ]
        )

    return contributions


def _json_safe(value):
    """Coerce a pandas Categorical/Timestamp cell to a plain JSON-serializable
    value for the API response."""
    if hasattr(value, "item"):
        return value.item()
    return str(value) if not isinstance(value, (int, float, str, bool)) else value
