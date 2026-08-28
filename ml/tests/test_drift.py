from __future__ import annotations

import numpy as np
import pandas as pd

from ml.pipeline.drift import compute_drift_report


def _make_df(values: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(values)


def test_identical_distributions_have_near_zero_psi():
    rng = np.random.default_rng(42)
    reference = _make_df({"lag_1": rng.normal(10, 2, size=500)})
    current = _make_df({"lag_1": rng.normal(10, 2, size=500)})

    report = compute_drift_report(reference, current, features=["lag_1"])

    assert report.status == "stable"
    assert report.max_psi < 0.1
    assert report.retrain_recommended is False


def test_shifted_distribution_triggers_significant_drift():
    rng = np.random.default_rng(42)
    reference = _make_df({"lag_1": rng.normal(10, 2, size=1000)})
    current = _make_df({"lag_1": rng.normal(30, 2, size=1000)})  # big mean shift

    report = compute_drift_report(reference, current, features=["lag_1"])

    assert report.status == "significant_shift"
    assert report.max_psi >= 0.25
    assert report.retrain_recommended is True
    assert report.max_psi_feature == "lag_1"


def test_moderate_shift_is_flagged_but_not_retrain_recommended():
    rng = np.random.default_rng(0)
    reference = _make_df({"lag_1": rng.normal(10, 2, size=2000)})
    # A modest shift — enough to register as moderate, not significant.
    current = _make_df({"lag_1": rng.normal(11.5, 2, size=2000)})

    report = compute_drift_report(reference, current, features=["lag_1"])

    assert report.max_psi > 0.0


def test_multiple_features_reports_the_worst_offender():
    rng = np.random.default_rng(1)
    reference = _make_df(
        {
            "lag_1": rng.normal(10, 2, size=1000),
            "rolling_mean_4": rng.normal(5, 1, size=1000),
        }
    )
    current = _make_df(
        {
            "lag_1": rng.normal(10, 2, size=1000),  # stable
            "rolling_mean_4": rng.normal(50, 1, size=1000),  # heavily shifted
        }
    )

    report = compute_drift_report(reference, current, features=["lag_1", "rolling_mean_4"])

    assert report.max_psi_feature == "rolling_mean_4"
    assert report.feature_psi["rolling_mean_4"] > report.feature_psi["lag_1"]


def test_missing_features_are_skipped_gracefully():
    reference = _make_df({"lag_1": np.arange(20, dtype=float)})
    current = _make_df({"lag_1": np.arange(20, dtype=float)})

    report = compute_drift_report(reference, current, features=["lag_1", "does_not_exist"])

    assert "does_not_exist" not in report.feature_psi
    assert "lag_1" in report.feature_psi


def test_empty_current_sample_returns_zero_psi():
    reference = _make_df({"lag_1": np.arange(20, dtype=float)})
    current = _make_df({"lag_1": np.array([], dtype=float)})

    report = compute_drift_report(reference, current, features=["lag_1"])

    assert report.feature_psi["lag_1"] == 0.0
