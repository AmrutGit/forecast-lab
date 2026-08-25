from __future__ import annotations

import pandas as pd
import pytest

from ml.pipeline.data_loading import (
    GROUP_KEYS,
    REQUIRED_COLUMNS,
    aggregate_to_weekly,
    load_raw_orders,
)


def test_load_raw_orders_validates_schema(tmp_path):
    df = pd.DataFrame({"order_id": [1], "order_date": ["2024-01-01"]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_orders(path)


def test_load_raw_orders_rejects_non_positive_quantity(tmp_path):
    df = pd.DataFrame(
        {col: [1] for col in REQUIRED_COLUMNS}
    )
    df["order_date"] = "2024-01-01"
    df["region_code"] = "Region A"
    df["product_category"] = "Outerwear"
    df["product_attribute_type"] = "material"
    df["product_attribute_value"] = "wool"
    df["quantity"] = 0
    df["unit_price"] = 100.0
    path = tmp_path / "bad_qty.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-positive quantity"):
        load_raw_orders(path)


def test_aggregate_to_weekly_sums_quantity(tiny_raw_orders):
    weekly = aggregate_to_weekly(tiny_raw_orders)

    assert "units_sold" in weekly.columns
    assert "week_start" in weekly.columns
    assert (weekly["units_sold"] > 0).all()

    # Each group's total units_sold across weeks must equal the raw total.
    raw_total = tiny_raw_orders.groupby(GROUP_KEYS)["quantity"].sum()
    weekly_total = weekly.groupby(GROUP_KEYS)["units_sold"].sum()
    pd.testing.assert_series_equal(raw_total.sort_index(), weekly_total.sort_index(), check_names=False)


def test_aggregate_to_weekly_revenue_matches_price_times_quantity(tiny_raw_orders):
    weekly = aggregate_to_weekly(tiny_raw_orders)
    expected_revenue = tiny_raw_orders["quantity"].sum() * 100.0
    assert weekly["revenue"].sum() == pytest.approx(expected_revenue)
