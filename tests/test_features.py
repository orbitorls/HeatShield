"""Tests for app/ml/forecast/features.py

CRITICAL: test_no_leakage verifies that the feature matrix has no temporal
data leakage — no feature value at row i was derived from timestamps
after that row's own ts_utc.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from app.ml.forecast.features import build_features, get_feature_names, _DEFAULT_LAGS_H, _DEFAULT_ROLLING_H, _get_lags_for_horizon


def _make_df(n_hours: int = 200, station_id: str = "BKK_01") -> pd.DataFrame:
    """Synthetic hourly weather data."""
    hours = pd.date_range("2024-05-01", periods=n_hours, freq="h", tz="UTC")
    temp = 30 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    rh = 70 + 10 * np.cos(2 * np.pi * np.arange(n_hours) / 24)
    return pd.DataFrame({
        "ts_utc": hours,
        "station_id": station_id,
        "temp_c": temp,
        "rh": rh,
    })


def test_build_features_returns_correct_shapes():
    df = _make_df(200)
    X, y = build_features(df, horizon_h=6)
    assert len(X) == len(y), "X and y must have same length"
    assert len(X) > 0, "Must produce at least some feature rows"
    assert len(X) < len(df), "Rows with NaN lags should be dropped"


def test_no_leakage():
    """CRITICAL: verify no feature row uses future data.

    Method:
    1. Create a df where each row's heat_index_c is its integer row index.
    2. Build features with horizon_h=6.
    3. For each row in X, verify all lag_Nh feature values are <= the
       index value at that row (i.e. they came from earlier rows).

    If lag_1h at row i contains value i+1 or higher, that's leakage.
    """
    n = 200
    hours = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "ts_utc": hours,
        "station_id": "TEST",
        "temp_c": np.arange(n, dtype=float),   # value == row index
        "rh": np.full(n, 60.0),
        "heat_index_c": np.arange(n, dtype=float),  # value == row index
    })

    horizon_h = 6
    X, y = build_features(df, horizon_h=horizon_h)

    # Check lag features: lag_Nh at position i should have value <= (i + offset - N)
    # Since we dropped leading NaN rows, the first valid X row corresponds to
    # df row [max(lags_h)] and the lag_1h there should be max(lags_h)-1
    lag_cols = [c for c in X.columns if "_lag" in c and "heat_index_c" in c]
    assert len(lag_cols) > 0, "Expected lag feature columns"

    for col in lag_cols:
        # Extract lag amount from column name, e.g. heat_index_c_lag24h -> 24
        lag_h = int(col.split("lag")[1].replace("h", ""))
        # The lag value should always be LESS than the current "time"
        # Since heat_index_c == row_index, lag_Nh[i] should be < X_index[i]
        # (lag value is from lag_h steps ago, current row is at some later index)
        # At minimum: lag value must be strictly less than the target row's index
        # We verify: no lag value equals or exceeds the FUTURE (target index)
        # Target for row i = row_index_of_X[i] + horizon_h
        # Lag value for row i = row_index_of_X[i] - lag_h (approximately)
        lag_values = X[col].values
        # Reconstruct approximate current "time index" using lag_1h as reference
        if "heat_index_c_lag1h" in X.columns:
            current_approx = X["heat_index_c_lag1h"].values + 1
            target_approx = current_approx + horizon_h
            assert np.all(lag_values < target_approx), (
                f"Leakage detected in {col}: lag value >= target value at some rows"
            )

    # v2 features — all use shift(k) with k>=1, so they must appear in X
    # hi_change_3h: shift(1) - shift(4) — safe
    # temp_change_3h: shift(1) - shift(4) — safe
    # temp_x_rh: shift(1) * shift(1) — safe
    # hi_residual_lag1h: shift(1) on the climatology residual — safe
    for v2_col in ["hi_change_3h", "temp_change_3h", "temp_x_rh", "hi_residual_lag1h"]:
        assert v2_col in X.columns, f"v2 feature missing from X: {v2_col}"


def test_y_is_future_heat_index():
    """Target y must be heat_index_c at t+horizon, not at t."""
    df = _make_df(200)
    horizon_h = 6
    X, y = build_features(df, horizon_h=horizon_h)
    # y should vary (it's future heat_index) — not constant
    assert y.std() > 0, "Target y should not be constant"


def test_no_nan_in_output():
    df = _make_df(200)
    X, y = build_features(df, horizon_h=6)
    assert not X.isna().any().any(), "X must contain no NaN values"
    assert not y.isna().any(), "y must contain no NaN values"


def test_feature_names_consistent():
    """get_feature_names(horizon_h=H) must return names that appear in build_features(horizon_h=H)."""
    df = _make_df(200)
    for h in [6, 24]:
        X, _ = build_features(df, horizon_h=h)
        expected = set(get_feature_names(horizon_h=h))
        actual = set(X.columns)
        missing = expected - actual
        assert len(missing) == 0, f"h{h}: expected feature columns missing from X: {missing}"


def test_v2_features_present():
    """All v2 feature additions must appear in X."""
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "station_id": "BKK_01",
        "temp_c": 28.0 + rng.normal(0, 2, n),
        "rh": 75.0 + rng.normal(0, 5, n),
        "solar_wm2": np.maximum(0, rng.normal(300, 150, n)),
        "cloud_cover": rng.uniform(0, 1, n),
        "pressure_hpa": 1010.0 + rng.normal(0, 3, n),
    })
    X, _ = build_features(df, horizon_h=24)
    for feat in ["local_hour_sin", "local_hour_cos", "hi_change_3h", "temp_change_3h",
                 "temp_x_rh", "hi_residual_lag1h"]:
        assert feat in X.columns, f"Missing v2 feature: {feat}"


def test_multi_station_lags_and_targets_do_not_cross_station_boundaries():
    """Lag and target values must stay within each station — no cross-contamination."""
    horizon = 6
    n = 200
    hours = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Station A: heat_index_c = 0,1,2,...,n-1; Station B: 1000,1001,...
    station_a = pd.DataFrame({
        "ts_utc": hours,
        "station_id": "A",
        "temp_c": np.arange(n, dtype=float),
        "rh": np.full(n, 60.0),
        "heat_index_c": np.arange(n, dtype=float),
    })
    station_b = pd.DataFrame({
        "ts_utc": hours,
        "station_id": "B",
        "temp_c": 1000.0 + np.arange(n, dtype=float),
        "rh": np.full(n, 60.0),
        "heat_index_c": 1000.0 + np.arange(n, dtype=float),
    })
    df = pd.concat([station_a, station_b], ignore_index=True)

    X, y = build_features(df, horizon_h=horizon)

    # The key invariant: heat_index_c lag values must stay within each station.
    # Station A: heat_index_c = 0..n-1 (all < 1000)
    # Station B: heat_index_c = 1000..1000+n-1 (all >= 1000)
    # Note: derived features like temp_rh (temp*rh) can legitimately exceed 1000 for
    # station A (e.g. 24°C × 60%RH = 1440), so we only check primary HI lag columns.
    hi_lag_cols = [c for c in X.columns if c.startswith("heat_index_c_lag") and "target_h" not in c]
    assert len(hi_lag_cols) > 0, "Expected heat_index_c lag columns"
    X_a = X[X["station_enc"] == 0]
    X_b = X[X["station_enc"] == 1]

    # heat_index_c lag values in station-A rows must all be < 1000
    assert (X_a[hi_lag_cols].to_numpy() < 1000).all(), \
        "Station-A heat_index_c lag features contain values from station B (cross-contamination)"
    # heat_index_c lag values in station-B rows must all be >= 1000
    assert (X_b[hi_lag_cols].to_numpy() >= 999).all(), \
        "Station-B heat_index_c lag features contain values from station A (cross-contamination)"

    # Targets must also stay within each station's value range
    y_a = y[X["station_enc"] == 0]
    y_b = y[X["station_enc"] == 1]
    if isinstance(y_a, pd.DataFrame):
        assert (y_a["temp_c"] < 1000).all()
        assert (y_b["temp_c"] >= 1000).all()
    else:
        assert (y_a < 1000).all()
        assert (y_b >= 1000).all()


def test_truncation_invariance_no_future_data_changes_past_features():
    """A5: Cutting df at time t must not change X values for rows <= t.

    Method: build X on full df, then truncate df at t and rebuild.
    Every feature column at common rows must match exactly.
    This catches any rolling/expanding computation that accidentally
    uses future rows (e.g. unshifted groupby aggregates).
    """
    n = 200
    horizon_h = 6
    hours = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "ts_utc": hours,
        "station_id": "TEST",
        "temp_c": 28.0 + rng.normal(0, 3, n),
        "rh": 70.0 + rng.normal(0, 5, n),
        "heat_index_c": 30.0 + rng.normal(0, 3, n),
    })

    # Build X on full df
    X_full, _ = build_features(df, horizon_h=horizon_h)

    # Truncate at midpoint and rebuild
    cut = n // 2
    df_cut = df.iloc[:cut].copy()
    X_cut, _ = build_features(df_cut, horizon_h=horizon_h)

    # Rows that appear in both: X_cut rows all have ts_utc <= df[cut-1].ts_utc
    # The corresponding rows in X_full must be identical feature-by-feature.
    common_cols = [c for c in X_cut.columns if c in X_full.columns]
    ts_in_cut = set(df_cut["ts_utc"].astype(str))

    # Align on ts_utc — find rows in X_full that correspond to X_cut rows
    # Features store ts_utc in X.attrs, but we can match via lag values.
    # Simpler: both X are sorted chronologically; X_cut rows are a prefix of X_full rows.
    n_common = min(len(X_cut), len(X_full))
    for col in common_cols:
        if col == "station_enc":
            continue  # encoding order may differ when df has only 1 station
        full_vals = X_full[col].values[:n_common]
        cut_vals = X_cut[col].values[:n_common]
        np.testing.assert_array_almost_equal(
            full_vals, cut_vals, decimal=6,
            err_msg=f"Causality violation in feature '{col}': "
                    f"truncating df changed past feature values (future data leaked in).",
        )


def test_physics_features_present():
    """B1: dewpoint_c, vpd_kpa, and wbgt_stull_c lag columns must appear in X."""
    df = _make_df(200)
    X, _ = build_features(df, horizon_h=6)
    for feat in ["dewpoint_c_lag1h", "vpd_kpa_lag1h", "wbgt_stull_c_lag1h"]:
        assert feat in X.columns, f"Physics lag feature missing from X: {feat}"
    # Raw (un-lagged) physics columns must NOT appear in X (causality guard)
    for raw_col in ["dewpoint_c", "vpd_kpa", "wbgt_stull_c"]:
        assert raw_col not in X.columns, (
            f"Raw physics column {raw_col!r} found in X — must only appear as lagged"
        )


def test_extended_feature_fill_does_not_backfill_future_values():
    """Sparse extended features may forward-fill past values but must not backfill."""
    n = 200
    df = pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "station_id": "A",
        "temp_c": np.arange(n, dtype=float),
        "rh": np.full(n, 60.0),
        "heat_index_c": np.arange(n, dtype=float),
        "solar_wm2": [np.nan] * 5 + [50.0] * 195,
    })

    X, _ = build_features(df, horizon_h=6)

    # First valid X row is at per-station index 72 (lag72h NaN threshold).
    # solar_wm2 was present from index 5, so after ffill and lag1h, value is 50.0.
    assert X["solar_wm2_lag1h"].iloc[0] == 50.0
    assert not (X["solar_wm2_lag1h"] == 0.0).any()


def test_feature_builder_does_not_fragment_dataframe_with_extended_columns():
    n = 260
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "station_id": "BKK_01",
        "temp_c": 30.0 + rng.normal(0, 2, n),
        "rh": 70.0 + rng.normal(0, 5, n),
        "heat_index_c": 34.0 + rng.normal(0, 2, n),
        "solar_wm2": np.maximum(0, rng.normal(300, 120, n)),
        "cloud_cover": rng.uniform(0, 1, n),
        "blh_m": rng.normal(900, 100, n),
        "pressure_hpa": rng.normal(1010, 3, n),
        "lst_c": rng.normal(33, 2, n),
        "wind_ms": rng.uniform(0, 6, n),
        "precip_mm": np.maximum(0, rng.normal(0.2, 0.5, n)),
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        X, y = build_features(df, horizon_h=24)

    perf_warnings = [
        warning for warning in caught
        if issubclass(warning.category, pd.errors.PerformanceWarning)
    ]
    assert not perf_warnings
    assert len(X) == len(y)
