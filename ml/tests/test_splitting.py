from __future__ import annotations

import pandas as pd
import pytest

from ml.pipeline.splitting import time_based_split


def test_time_based_split_has_no_boundary_leakage():
    df = pd.DataFrame(
        {
            "week_start": pd.date_range("2024-01-01", periods=20, freq="W-MON"),
            "value": range(20),
        }
    )
    train, val, test = time_based_split(df, train_end_date="2024-03-04", val_end_date="2024-04-08")

    assert train["week_start"].max() <= pd.Timestamp("2024-03-04")
    assert val["week_start"].min() > pd.Timestamp("2024-03-04")
    assert val["week_start"].max() <= pd.Timestamp("2024-04-08")
    assert test["week_start"].min() > pd.Timestamp("2024-04-08")

    assert len(train) + len(val) + len(test) == len(df)


def test_time_based_split_raises_on_empty_partition():
    df = pd.DataFrame(
        {"week_start": pd.date_range("2024-01-01", periods=5, freq="W-MON"), "value": range(5)}
    )
    with pytest.raises(ValueError, match="empty partition"):
        time_based_split(df, train_end_date="2030-01-01", val_end_date="2031-01-01")
