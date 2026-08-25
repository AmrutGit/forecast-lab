from __future__ import annotations

import numpy as np
import pytest

from ml.pipeline.data_loading import aggregate_to_weekly
from ml.pipeline.features import (
    FEATURE_COLUMNS,
    build_lookup_artifacts,
    engineer_features,
)


def test_engineer_features_adds_all_columns(tiny_raw_orders):
    weekly = aggregate_to_weekly(tiny_raw_orders)
    featured = engineer_features(weekly)
    for col in FEATURE_COLUMNS:
        assert col in featured.columns


def test_lag_features_no_leakage(tiny_raw_orders):
    """A lag_1 feature for week N must equal units_sold from week N-1 for the
    same group — and must never equal week N's own units_sold (no leakage)."""
    weekly = aggregate_to_weekly(tiny_raw_orders)
    featured = engineer_features(weekly)

    group = featured[
        (featured["region_code"] == "Region A")
        & (featured["product_attribute_value"] == "wool")
    ].sort_values("week_start")

    units = group["units_sold"].to_numpy()
    lag_1 = group["lag_1"].to_numpy()

    # lag_1[i] should equal units[i-1], and be NaN for the first row.
    assert np.isnan(lag_1[0])
    np.testing.assert_array_equal(lag_1[1:], units[:-1])


def test_rolling_mean_uses_only_prior_weeks(tiny_raw_orders):
    weekly = aggregate_to_weekly(tiny_raw_orders)
    featured = engineer_features(weekly)

    group = featured[
        (featured["region_code"] == "Region A")
        & (featured["product_attribute_value"] == "wool")
    ].sort_values("week_start").reset_index(drop=True)

    # rolling_mean_4 at row i must be computable purely from units_sold[0:i]
    # (strictly before row i), never including units_sold[i] itself.
    units = group["units_sold"].to_numpy(dtype=float)
    for i in range(1, len(group)):
        window = units[max(0, i - 4):i]
        expected = window.mean()
        assert group.loc[i, "rolling_mean_4"] == pytest.approx(expected)


def test_region_category_share_sums_to_one_within_group(tiny_raw_orders):
    weekly = aggregate_to_weekly(tiny_raw_orders)
    featured = engineer_features(weekly)

    # For weeks after the first (where shares are defined), shares across
    # attribute values within the same (region, category, attribute_type,
    # week) must sum to ~1.0 (since both values had nonzero prior sales).
    later = featured[featured["week_start"] > featured["week_start"].min()]
    grouped = later.groupby(["region_code", "product_category", "product_attribute_type", "week_start"])[
        "region_category_share"
    ].sum()
    assert (grouped.round(6) == 1.0).all()


def test_build_lookup_artifacts_returns_latest_row_per_group(tiny_raw_orders):
    weekly = aggregate_to_weekly(tiny_raw_orders)
    featured = engineer_features(weekly)
    lookup = build_lookup_artifacts(featured)

    result = lookup.get_features_for_group("Region A", "Outerwear", "material", "wool")
    assert result is not None
    assert result["week_start"] == featured["week_start"].max()

    missing = lookup.get_features_for_group("Region A", "Outerwear", "material", "nonexistent")
    assert missing is None
