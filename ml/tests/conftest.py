from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def tiny_raw_orders() -> pd.DataFrame:
    """A hand-built raw order-line dataset spanning 6 weeks, 2 regions,
    1 category, 1 attribute type with 2 values — enough to exercise
    aggregation, feature lags, and splitting without needing real data."""
    rows = []
    order_id = 1
    dates = pd.date_range("2024-01-01", periods=42, freq="D")  # 6 weeks
    for d in dates:
        for region in ["Region A", "Region B"]:
            for value, qty in [("wool", 5), ("down", 3)]:
                rows.append(
                    {
                        "order_id": order_id,
                        "order_date": d,
                        "region_code": region,
                        "product_category": "Outerwear",
                        "product_attribute_type": "material",
                        "product_attribute_value": value,
                        "quantity": qty,
                        "unit_price": 100.0,
                    }
                )
                order_id += 1
    return pd.DataFrame(rows)
