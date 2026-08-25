"""Time-based train/validation/test split for weekly-aggregated demand data.

Splitting is strictly chronological on `week_start` — no shuffling, no
group-based leakage across the boundary. Rows are partitioned by cutoff date
only, so a (region, category, attribute_type, attribute_value) group can and
will appear in all three splits, just at different weeks — which is correct
for a forecasting task where we want to predict *future* weeks for groups
we've already seen history for.
"""

from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
    date_col: str = "week_start",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)

    dates = df[date_col]
    train = df[dates <= train_end].copy()
    val = df[(dates > train_end) & (dates <= val_end)].copy()
    test = df[dates > val_end].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            "Time-based split produced an empty partition "
            f"(train={len(train)}, val={len(val)}, test={len(test)}). "
            "Check train_end_date/val_end_date against the data's date range."
        )

    return train, val, test
