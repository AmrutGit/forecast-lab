"""Exploratory data analysis for the Forecast Lab synthetic dataset.

Produces saved artifacts under docs/eda/:
  - summary_stats.json: row counts, missingness, date range, cardinality
  - seasonality_by_category.png: monthly demand curve per category
  - regional_attribute_skew.png: heatmap of region x attribute-value affinity
  - trend_over_time.png: quarterly demand trend per category

These findings directly informed the feature set in ml/pipeline/features.py:
  - Visible monthly seasonality -> calendar features (month, quarter) plus
    lag_52 (year-over-year) to let the model learn the seasonal curve.
  - Visible regional skew per attribute value -> region_category_share
    feature so the model can learn each group's relative popularity per
    region rather than treating all regions identically.
  - Visible slow multi-year trend -> trend_slope_12 and rolling means, so the
    model has an explicit trend-direction signal instead of relying solely
    on lag features to infer drift.
  - No missing values found in required columns after generation -> no
    imputation step was needed in the feature pipeline.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.config.settings import CONFIG
from ml.pipeline.data_loading import GROUP_KEYS, aggregate_to_weekly, load_raw_orders

sns.set_theme(style="whitegrid")


def compute_summary_stats(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "date_range": [str(df["order_date"].min().date()), str(df["order_date"].max().date())],
        "n_regions": int(df["region_code"].nunique()),
        "n_categories": int(df["product_category"].nunique()),
        "n_attribute_types": int(df["product_attribute_type"].nunique()),
        "n_attribute_values": int(df["product_attribute_value"].nunique()),
        "missingness_pct": {col: float(df[col].isnull().mean() * 100) for col in df.columns},
        "quantity_stats": {
            "mean": float(df["quantity"].mean()),
            "std": float(df["quantity"].std()),
            "min": int(df["quantity"].min()),
            "max": int(df["quantity"].max()),
        },
        "unit_price_stats": {
            "mean": float(df["unit_price"].mean()),
            "min": float(df["unit_price"].min()),
            "max": float(df["unit_price"].max()),
        },
        "rows_per_region": df["region_code"].value_counts().to_dict(),
        "rows_per_category": df["product_category"].value_counts().to_dict(),
    }


def plot_seasonality_by_category(df: pd.DataFrame, out_path) -> None:
    monthly = df.copy()
    monthly["month"] = monthly["order_date"].dt.month
    agg = monthly.groupby(["product_category", "month"])["quantity"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for category, group in agg.groupby("product_category"):
        ax.plot(group["month"], group["quantity"], marker="o", label=category)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total units sold")
    ax.set_title("Monthly demand seasonality by category")
    ax.set_xticks(range(1, 13))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_regional_attribute_skew(df: pd.DataFrame, out_path, category: str = "Outerwear", attribute_type: str = "material") -> None:
    subset = df[(df["product_category"] == category) & (df["product_attribute_type"] == attribute_type)]
    pivot = subset.groupby(["region_code", "product_attribute_value"])["quantity"].sum().unstack(fill_value=0)
    pivot_share = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_share, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(f"Regional preference share — {category} / {attribute_type}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trend_over_time(df: pd.DataFrame, out_path) -> None:
    quarterly = df.copy()
    quarterly["year_quarter"] = quarterly["order_date"].dt.to_period("Q").astype(str)
    agg = quarterly.groupby(["product_category", "year_quarter"])["quantity"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for category, group in agg.groupby("product_category"):
        ax.plot(group["year_quarter"], group["quantity"], marker="o", label=category)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Total units sold")
    ax.set_title("Quarterly demand trend by category")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = CONFIG
    df = load_raw_orders(cfg.paths.raw_data)
    cfg.paths.eda_report_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_summary_stats(df)
    with open(cfg.paths.eda_report_dir / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print("Wrote summary_stats.json")
    print(json.dumps(stats, indent=2, default=str))

    plot_seasonality_by_category(df, cfg.paths.eda_report_dir / "seasonality_by_category.png")
    print("Wrote seasonality_by_category.png")

    plot_regional_attribute_skew(df, cfg.paths.eda_report_dir / "regional_attribute_skew.png")
    print("Wrote regional_attribute_skew.png")

    plot_trend_over_time(df, cfg.paths.eda_report_dir / "trend_over_time.png")
    print("Wrote trend_over_time.png")

    weekly = aggregate_to_weekly(df)
    weekly_stats = {
        "n_weekly_group_rows": int(len(weekly)),
        "n_unique_groups": int(weekly.groupby(GROUP_KEYS).ngroups),
        "weeks_covered": int(weekly["week_start"].nunique()),
    }
    with open(cfg.paths.eda_report_dir / "weekly_agg_stats.json", "w") as f:
        json.dump(weekly_stats, f, indent=2)
    print("Wrote weekly_agg_stats.json")
    print(json.dumps(weekly_stats, indent=2))


if __name__ == "__main__":
    main()
