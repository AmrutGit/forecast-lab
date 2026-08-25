"""Synthetic order-line data generator for Forecast Lab.

Produces a realistic, internally-consistent dataset of retail order lines
with deliberately injected structure:

  - Seasonality: each category has a month-of-year demand curve (e.g.
    Outerwear peaks in winter months, driven by each region's climate).
  - Regional preference skew: each region has an affinity multiplier per
    attribute value (e.g. cold regions favor "wool"/"heavy" attributes).
  - Slow trend: a gentle multi-year growth or decline per category so the
    model has a trend signal to learn, not just seasonality.
  - Poisson noise on top of the deterministic signal, so no two runs of the
    underlying process look identical without a fixed seed.

Every source of randomness is drawn from a single seeded `np.random.Generator`
so the output is byte-for-byte reproducible given the same config.

Usage:
    python -m ml.data_generator.generate
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from ml.config.settings import CONFIG
from ml.config.taxonomy import (
    CATEGORIES,
    CATEGORY_ATTRIBUTES,
    REGION_CLIMATE,
    REGIONS,
)

# --- Category base price ranges (unit_price is drawn per line, not fixed) ---
CATEGORY_PRICE_RANGE: dict[str, tuple[float, float]] = {
    "Outerwear": (60.0, 220.0),
    "Footwear": (40.0, 160.0),
    "Tops": (15.0, 60.0),
    "Bottoms": (25.0, 90.0),
    "Accessories": (10.0, 50.0),
}

# --- Category seasonality: relative monthly demand multiplier (1.0 = average) ---
# Cold-peaking categories vs. warm-peaking / flat categories.
CATEGORY_SEASON_PROFILE: dict[str, str] = {
    "Outerwear": "cold_peak",
    "Footwear": "flat",
    "Tops": "warm_peak",
    "Bottoms": "flat",
    "Accessories": "cold_peak",
}

_MONTHS = np.arange(1, 13)


def _stable_salt(*parts: str) -> int:
    """Deterministic 32-bit salt from string parts, independent of
    PYTHONHASHSEED (unlike the built-in hash())."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _seasonal_curve(profile: str, climate: str) -> np.ndarray:
    """Return a 12-length multiplier array (index 0 = January), shaped by both
    the category's seasonal profile and the region's climate (climate widens
    or dampens the swing amplitude)."""
    amplitude_by_climate = {"cold": 0.55, "temperate": 0.35, "warm": 0.15}
    amp = amplitude_by_climate[climate]

    if profile == "cold_peak":
        # Peak in Dec/Jan/Feb, trough in Jun/Jul.
        phase = np.cos((_MONTHS - 1.5) / 12 * 2 * np.pi)
    elif profile == "warm_peak":
        # Peak in Jun/Jul, trough in Dec/Jan.
        phase = -np.cos((_MONTHS - 1.5) / 12 * 2 * np.pi)
    else:  # flat
        phase = np.zeros(12)
        amp = amp * 0.2  # footwear/bottoms still get a faint climate wobble

    return 1.0 + amp * phase


def _trend_multiplier(day_index: np.ndarray, total_days: int, category: str, rng: np.random.Generator) -> np.ndarray:
    """Slow multi-year trend per category: a fixed per-category drift plus a
    small random slope perturbation so categories don't all trend identically."""
    base_drift = {
        "Outerwear": 0.15,
        "Footwear": 0.05,
        "Tops": 0.20,
        "Bottoms": -0.05,
        "Accessories": 0.10,
    }[category]
    jitter = rng.normal(0, 0.03)
    total_growth = base_drift + jitter
    return 1.0 + total_growth * (day_index / max(total_days - 1, 1))


def _region_attribute_affinity(
    region: str, category: str, attribute_type: str, attribute_value: str
) -> float:
    """Deterministic-per-seed affinity multiplier expressing regional taste
    skew for a given attribute value. Built from a stable hash of the
    (region, category, attribute_type, attribute_value) tuple combined with
    a few hand-authored structural skews (climate -> material/weight)."""
    climate = REGION_CLIMATE[region]

    structural = 1.0
    if attribute_type == "weight_class":
        if climate == "cold" and attribute_value == "heavy":
            structural = 1.6
        elif climate == "warm" and attribute_value == "light":
            structural = 1.6
        elif climate == "temperate" and attribute_value == "mid-weight":
            structural = 1.3
    if attribute_type == "material":
        cold_materials = {"wool", "down", "fleece"}
        warm_materials = {"linen", "cotton", "canvas", "synthetic-mesh"}
        if climate == "cold" and attribute_value in cold_materials:
            structural = 1.5
        elif climate == "warm" and attribute_value in warm_materials:
            structural = 1.4

    # Stable pseudo-random per-region flavor so every attribute value has
    # *some* regional skew, not just the structural climate-driven ones.
    salt = _stable_salt(region, category, attribute_type, attribute_value)
    flavor = np.random.default_rng(salt).uniform(0.75, 1.35)

    return structural * flavor


def generate_dataset(
    n_rows: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate the synthetic order-line dataset as a DataFrame.

    Parameters override config defaults; used by tests to build tiny fixtures.
    """
    cfg = CONFIG.data_generation
    n_rows = n_rows or cfg.n_rows
    start_date = start_date or cfg.start_date
    end_date = end_date or cfg.end_date
    seed = cfg.seed if seed is None else seed

    rng = np.random.default_rng(seed)

    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    total_days = (end - start).days + 1
    all_dates = pd.date_range(start, end, freq="D")

    # Build a flat list of (category, attribute_type, attribute_value) combos.
    combos: list[tuple[str, str, str]] = []
    for category in CATEGORIES:
        for attr_type, values in CATEGORY_ATTRIBUTES[category].items():
            for value in values:
                combos.append((category, attr_type, value))

    # Precompute seasonal curves per (category, region).
    seasonal_curves: dict[tuple[str, str], np.ndarray] = {}
    for category in CATEGORIES:
        profile = CATEGORY_SEASON_PROFILE[category]
        for region in REGIONS:
            climate = REGION_CLIMATE[region]
            seasonal_curves[(category, region)] = _seasonal_curve(profile, climate)

    # Precompute regional affinity per (region, category, attr_type, value).
    affinity: dict[tuple[str, str, str, str], float] = {}
    for region in REGIONS:
        for category, attr_type, value in combos:
            affinity[(region, category, attr_type, value)] = _region_attribute_affinity(
                region, category, attr_type, value
            )

    # Precompute trend multiplier series per category (indexed by day offset).
    day_idx = np.arange(total_days)
    trend_by_category = {
        category: _trend_multiplier(day_idx, total_days, category, rng) for category in CATEGORIES
    }

    # Assign a base popularity weight per combo (some attribute values are
    # just more popular everywhere, e.g. "regular" fit vs. "wide-leg").
    base_weight = {}
    for category, attr_type, value in combos:
        salt = _stable_salt("base_weight", category, attr_type, value)
        base_weight[(category, attr_type, value)] = np.random.default_rng(salt).uniform(0.5, 2.0)

    # Relative size of each region's customer base (not all regions equal).
    region_size = {}
    for region in REGIONS:
        salt = _stable_salt("region_size", region)
        region_size[region] = np.random.default_rng(salt).uniform(0.7, 1.4)

    # Compute an expected-lambda weight for every (date, region, combo) and
    # sample proportionally until we hit n_rows order lines. Rather than loop
    # per-day (slow at this scale), we vectorize by sampling combo/region/day
    # indices according to a multinomial built from the structural weights.
    n_combos = len(combos)
    n_regions = len(REGIONS)

    # Build per-day weight matrix lazily: expected weight for each
    # (region, combo) pair varies by day through season + trend, so we sample
    # day first (uniform, since day-level weighting is handled via season/trend
    # applied post-hoc as a rate), then region or combo conditioned on the day's
    # season/trend adjustment via rejection-free categorical sampling.
    #
    # Practical approach: sample day uniformly across the range (orders occur
    # every day), then sample (region, combo) from a weight vector that
    # incorporates that day's seasonal + trend multipliers.
    day_samples = rng.integers(0, total_days, size=n_rows)
    day_month = pd.DatetimeIndex(all_dates[day_samples]).month.to_numpy()

    region_idx_arr = np.empty(n_rows, dtype=np.int64)
    combo_idx_arr = np.empty(n_rows, dtype=np.int64)

    # Group rows by month to batch weight computation (12 unique weight
    # matrices instead of n_rows).
    weight_cache: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        w = np.empty((n_regions, n_combos), dtype=np.float64)
        for ri, region in enumerate(REGIONS):
            for ci, (category, attr_type, value) in enumerate(combos):
                season_mult = seasonal_curves[(category, region)][month - 1]
                w[ri, ci] = (
                    base_weight[(category, attr_type, value)]
                    * affinity[(region, category, attr_type, value)]
                    * season_mult
                    * region_size[region]
                )
        weight_cache[month] = w

    for month in range(1, 13):
        mask = day_month == month
        n_month = mask.sum()
        if n_month == 0:
            continue
        w = weight_cache[month]
        flat_w = w.flatten()
        flat_p = flat_w / flat_w.sum()
        flat_idx = rng.choice(len(flat_p), size=n_month, p=flat_p)
        region_idx_arr[mask] = flat_idx // n_combos
        combo_idx_arr[mask] = flat_idx % n_combos

    # Apply trend on top of quantity (season/affinity already baked into
    # selection probability; trend affects magnitude of each line's quantity).
    categories_arr = np.array([combos[i][0] for i in combo_idx_arr])
    attr_types_arr = np.array([combos[i][1] for i in combo_idx_arr])
    attr_values_arr = np.array([combos[i][2] for i in combo_idx_arr])
    regions_arr = np.array([REGIONS[i] for i in region_idx_arr])

    trend_mult = np.empty(n_rows, dtype=np.float64)
    for category in CATEGORIES:
        cat_mask = categories_arr == category
        trend_mult[cat_mask] = trend_by_category[category][day_samples[cat_mask]]

    base_lambda = 3.0 * trend_mult
    quantities = rng.poisson(lam=np.clip(base_lambda, 0.5, None)) + 1  # at least 1 unit per line

    # Unit price: drawn within category range, with a mild positive
    # correlation to quantity noise removed (price is independent of qty).
    unit_prices = np.empty(n_rows, dtype=np.float64)
    for category in CATEGORIES:
        cat_mask = categories_arr == category
        lo, hi = CATEGORY_PRICE_RANGE[category]
        n_cat = cat_mask.sum()
        prices = rng.uniform(lo, hi, size=n_cat)
        unit_prices[cat_mask] = np.round(prices, 2)

    order_dates = all_dates[day_samples]

    df = pd.DataFrame(
        {
            "order_id": np.arange(1, n_rows + 1),
            "order_date": order_dates,
            "region_code": regions_arr,
            "product_category": categories_arr,
            "product_attribute_type": attr_types_arr,
            "product_attribute_value": attr_values_arr,
            "quantity": quantities.astype(np.int64),
            "unit_price": unit_prices,
        }
    )

    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    df["order_id"] = np.arange(1, len(df) + 1)

    return df


def main() -> None:
    cfg = CONFIG.data_generation
    df = generate_dataset()
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)
    print(f"Wrote {len(df):,} rows to {cfg.output_path}")
    print(df.head())
    print("\nDate range:", df["order_date"].min(), "to", df["order_date"].max())
    print("Regions:", sorted(df["region_code"].unique()))
    print("Categories:", sorted(df["product_category"].unique()))


if __name__ == "__main__":
    main()
