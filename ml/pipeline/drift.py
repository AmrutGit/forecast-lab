"""Feature drift detection via Population Stability Index (PSI).

PSI is the standard industry metric for comparing a reference distribution
(here: a feature's distribution in the training set) against a current
distribution (here: the same feature in the most recent weeks of data), used
widely in credit risk and demand forecasting to flag when a model's input
distribution has shifted enough that a retrain is warranted, independent of
whether ground-truth labels are even available yet for the recent period
(which is exactly why PSI is useful in a forecasting context — you often
know demand shifted before you know how well the model predicted it).

Interpretation (standard thresholds used across industry):
    PSI < 0.1   -> no significant shift
    0.1 <= PSI < 0.25 -> moderate shift, worth monitoring
    PSI >= 0.25 -> significant shift, retrain recommended

We compute PSI per monitored feature and report the maximum across features
as the headline "should we retrain" signal, since any single badly-drifted
feature can degrade predictions even if others are stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONITORED_FEATURES = [
    "lag_1",
    "rolling_mean_4",
    "rolling_mean_12",
    "group_avg_price",
    "region_category_share",
]

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    max_psi: float
    max_psi_feature: str
    status: str  # "stable" | "moderate_shift" | "significant_shift"
    retrain_recommended: bool


def _psi_for_feature(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """PSI between two 1-D samples of the same feature, binned on the
    reference sample's quantiles so each reference bin starts with ~equal
    weight (standard PSI construction)."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) < n_bins or len(current) == 0:
        return 0.0

    quantile_edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(quantile_edges) < 3:
        return 0.0  # feature has near-zero variance in the reference set; PSI undefined/meaningless

    quantile_edges[0] = -np.inf
    quantile_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=quantile_edges)
    cur_counts, _ = np.histogram(current, bins=quantile_edges)

    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-4, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str] | None = None
) -> DriftReport:
    """Compare `current_df` (e.g. the most recent N weeks) against
    `reference_df` (e.g. the training set) across `features`, returning a
    per-feature PSI and an overall retrain recommendation."""
    features = features or MONITORED_FEATURES

    feature_psi = {}
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        feature_psi[feature] = _psi_for_feature(
            reference_df[feature].to_numpy(dtype=np.float64),
            current_df[feature].to_numpy(dtype=np.float64),
        )

    if not feature_psi:
        return DriftReport(
            feature_psi={}, max_psi=0.0, max_psi_feature="", status="stable", retrain_recommended=False
        )

    max_psi_feature = max(feature_psi, key=feature_psi.get)
    max_psi = feature_psi[max_psi_feature]

    if max_psi >= PSI_SIGNIFICANT_THRESHOLD:
        status = "significant_shift"
    elif max_psi >= PSI_MODERATE_THRESHOLD:
        status = "moderate_shift"
    else:
        status = "stable"

    return DriftReport(
        feature_psi=feature_psi,
        max_psi=max_psi,
        max_psi_feature=max_psi_feature,
        status=status,
        retrain_recommended=max_psi >= PSI_SIGNIFICANT_THRESHOLD,
    )
