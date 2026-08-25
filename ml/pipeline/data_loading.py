"""Load and validate raw order-line data, and aggregate it to the weekly
(region, category, attribute_type, attribute_value) grain the model trains on."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "region_code",
    "product_category",
    "product_attribute_type",
    "product_attribute_value",
    "quantity",
    "unit_price",
]

GROUP_KEYS = [
    "region_code",
    "product_category",
    "product_attribute_type",
    "product_attribute_value",
]


def load_raw_orders(path: Path) -> pd.DataFrame:
    """Load the raw order-line CSV and validate its schema."""
    df = pd.read_csv(path, parse_dates=["order_date"])

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data at {path} is missing required columns: {sorted(missing)}")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        null_cols = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isnull().any()].tolist()
        raise ValueError(f"Raw data contains nulls in required columns: {null_cols}")

    if (df["quantity"] <= 0).any():
        raise ValueError("Raw data contains non-positive quantity values")
    if (df["unit_price"] <= 0).any():
        raise ValueError("Raw data contains non-positive unit_price values")

    return df


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw order lines to weekly units_sold per group.

    The week label is the Monday that starts each ISO week, giving stable,
    sortable period boundaries for the time-based split downstream.
    """
    working = df.copy()
    working["week_start"] = working["order_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    agg = (
        working.groupby(GROUP_KEYS + ["week_start"])
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("unit_price", lambda s: (s * working.loc[s.index, "quantity"]).sum()),
            avg_unit_price=("unit_price", "mean"),
            n_orders=("order_id", "nunique"),
        )
        .reset_index()
    )
    return agg
