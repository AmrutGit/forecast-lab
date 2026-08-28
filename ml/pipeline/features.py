"""Feature engineering for the weekly demand model.

Design principle — no train/serve skew:
    Every feature that depends on historical data (lags, rolling means,
    group-level rate baselines) is computed here from a full weekly panel and
    is available at both training time and serving time from the *same*
    lookup artifact (see `build_lookup_artifacts` / `LookupArtifacts`). At
    serving time we never recompute these from scratch against a different
    (shorter, live) window, and we never fill them with placeholders — the
    lookup artifact IS the source of truth for "as of last known week" values
    for every (region, category, attribute_type, attribute_value) group.

Feature list and justification:
    - calendar features (month, quarter, week_of_year, is_holiday_quarter):
      captures seasonality directly; cheap, no leakage since they're
      functions of the date alone.
    - lag_1, lag_4, lag_52 (units_sold N weeks ago): captures short-term
      momentum (lag_1/lag_4) and year-over-year seasonality (lag_52).
    - rolling_mean_4, rolling_mean_12 (trailing average units_sold): smooths
      noise, captures recent trend level.
    - rolling_std_4: captures recent volatility, helps the model distinguish
      a stable riser from a noisy one.
    - group_avg_price: trailing average unit price for the group, since
      price level can shift demand.
    - region_category_share: this attribute value's trailing share of its
      (region, category, attribute_type) group's total demand — captures
      relative popularity, which is exactly what the ranking product surface
      needs (NDCG@5 within a group).
    - trend_slope_12: slope of a linear fit over the trailing 12 weeks,
      an explicit trend-direction signal distinct from the level captured by
      rolling means.

All of the above are computed strictly from weeks *before* the target week
(shift(1) before any rolling/lag operation), so no feature ever peeks at the
label it's predicting.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from ml.pipeline.data_loading import GROUP_KEYS

LAGS = [1, 4, 52]
ROLLING_WINDOWS = [4, 12]

FEATURE_COLUMNS = [
    "month",
    "quarter",
    "week_of_year",
    "is_holiday_quarter",
    "lag_1",
    "lag_4",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_12",
    "rolling_std_4",
    "group_avg_price",
    "region_category_share",
    "trend_slope_12",
]

CATEGORICAL_COLUMNS = [
    "region_code",
    "product_category",
    "product_attribute_type",
    "product_attribute_value",
]


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["week_start"].dt.month
    out["quarter"] = out["week_start"].dt.quarter
    out["week_of_year"] = out["week_start"].dt.isocalendar().week.astype(int)
    out["is_holiday_quarter"] = out["quarter"].isin([4]).astype(int)
    return out


def _trend_slope(values: pd.Series) -> float:
    """Slope of an OLS fit of values against a 0..n-1 index. Returns 0 for
    fewer than 2 points."""
    n = len(values)
    if n < 2 or values.isnull().all():
        return 0.0
    x = np.arange(n)
    y = values.to_numpy(dtype=np.float64)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    slope = np.polyfit(x[mask], y[mask], 1)[0]
    return float(slope)


def _lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag/rolling/trend features per group, using only data strictly
    before the current week (shift(1) applied before any window op)."""
    out = df.sort_values(GROUP_KEYS + ["week_start"]).copy()
    grouped = out.groupby(GROUP_KEYS, group_keys=False)["units_sold"]

    for lag in LAGS:
        out[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    out["_shifted_units"] = shifted

    for window in ROLLING_WINDOWS:
        roll = out.groupby(GROUP_KEYS, group_keys=False)["_shifted_units"].rolling(
            window=window, min_periods=1
        ).mean()
        out[f"rolling_mean_{window}"] = roll.reset_index(level=list(range(len(GROUP_KEYS))), drop=True)

    roll_std = out.groupby(GROUP_KEYS, group_keys=False)["_shifted_units"].rolling(
        window=4, min_periods=2
    ).std()
    out["rolling_std_4"] = roll_std.reset_index(level=list(range(len(GROUP_KEYS))), drop=True)

    price_grouped = out.groupby(GROUP_KEYS, group_keys=False)["avg_unit_price"].shift(1)
    out["group_avg_price"] = (
        out.assign(_shifted_price=price_grouped)
        .groupby(GROUP_KEYS, group_keys=False)["_shifted_price"]
        .rolling(window=12, min_periods=1)
        .mean()
        .reset_index(level=list(range(len(GROUP_KEYS))), drop=True)
    )

    trend = (
        out.groupby(GROUP_KEYS, group_keys=False)["_shifted_units"]
        .rolling(window=12, min_periods=2)
        .apply(_trend_slope, raw=False)
    )
    out["trend_slope_12"] = trend.reset_index(level=list(range(len(GROUP_KEYS))), drop=True)

    out = out.drop(columns=["_shifted_units"])
    return out


def _region_category_share(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing share of demand this attribute_value holds within its
    (region, category, attribute_type) parent group, as of the prior week."""
    out = df.copy()
    parent_keys = ["region_code", "product_category", "product_attribute_type", "week_start"]

    shifted_units = out.groupby(GROUP_KEYS, group_keys=False)["units_sold"].shift(1)
    out["_shifted_units_for_share"] = shifted_units

    parent_total = out.groupby(parent_keys)["_shifted_units_for_share"].transform("sum")
    out["region_category_share"] = np.where(
        parent_total > 0, out["_shifted_units_for_share"] / parent_total, 0.0
    )
    out = out.drop(columns=["_shifted_units_for_share"])
    return out


def engineer_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline applied to a weekly-aggregated panel. Returns
    the input augmented with all FEATURE_COLUMNS, ready for train/predict."""
    out = _calendar_features(weekly_df)
    out = _lag_and_rolling_features(out)
    out = _region_category_share(out)
    return out


@dataclass
class LookupArtifacts:
    """Persisted historical-aggregate lookups, keyed for O(1) serving-time
    feature construction. Built once from the full training-period panel and
    reused identically at inference — this is what prevents train/serve skew
    for any feature derived from history.

    Also carries a reference sample of training-set feature values
    (`drift_reference`), used by ml.pipeline.drift to detect when live data
    has shifted away from what the model was trained on — bundled here
    rather than as a separate artifact since it travels with the same model
    version and is irrelevant once that version is retired."""

    latest_by_group: pd.DataFrame  # one row per group: most recent known feature snapshot
    as_of_week: pd.Timestamp
    drift_reference: pd.DataFrame  # sample of training-set feature values, for PSI comparison

    def get_features_for_group(
        self, region_code: str, product_category: str, product_attribute_type: str, product_attribute_value: str
    ) -> dict | None:
        mask = (
            (self.latest_by_group["region_code"] == region_code)
            & (self.latest_by_group["product_category"] == product_category)
            & (self.latest_by_group["product_attribute_type"] == product_attribute_type)
            & (self.latest_by_group["product_attribute_value"] == product_attribute_value)
        )
        rows = self.latest_by_group[mask]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()


def build_lookup_artifacts(featured_df: pd.DataFrame, drift_reference_df: pd.DataFrame | None = None) -> LookupArtifacts:
    """From a fully-featured historical panel, extract the most recent row
    per group — this becomes the frozen lookup used to build serving-time
    feature vectors for "predict next period" requests.

    `drift_reference_df` should be the *training-set* slice of the featured
    panel (passed explicitly by the caller, since this function also gets
    called with the full historical panel for the lookup half); it's stored
    verbatim for later PSI comparison against live data. Defaults to
    `featured_df` itself if not given.
    """
    latest_idx = featured_df.groupby(GROUP_KEYS)["week_start"].idxmax()
    latest = featured_df.loc[latest_idx].reset_index(drop=True)
    as_of_week = featured_df["week_start"].max()
    drift_reference = (drift_reference_df if drift_reference_df is not None else featured_df)[
        FEATURE_COLUMNS
    ].copy()
    return LookupArtifacts(latest_by_group=latest, as_of_week=as_of_week, drift_reference=drift_reference)


def save_lookup_artifacts(artifacts: LookupArtifacts, path) -> None:
    joblib.dump(artifacts, path)


def load_lookup_artifacts(path) -> LookupArtifacts:
    return joblib.load(path)
