"""Feature engineering for heat-index forecast model.

CRITICAL: All lag and rolling features must only use data available at
prediction time (t=0). The target y = heat_index_c at t+horizon_h.
No future values are ever used in X.
"""
from __future__ import annotations
from typing import Literal, Sequence
import hashlib as _hashlib
import re
import numpy as np
import pandas as pd

from app.data.stations import STATIONS


# Extended lags/rolling for long-horizon (h48/h72) support.
# Models for short horizons can learn to ignore irrelevant long lags.
_DEFAULT_LAGS_H = [1, 3, 6, 12, 24, 48, 72, 96, 120, 168]
_DEFAULT_ROLLING_H = [3, 6, 24, 48, 72, 168]

# Per-horizon lag/rolling selection — reduces noise for short horizons,
# increases context for long horizons.
_LAGS_BY_HORIZON: dict[int, list[int]] = {
    6:   [1, 2, 3, 6, 12, 24],
    12:  [1, 3, 6, 12, 24, 48],
    24:  [1, 6, 12, 24, 48, 72],
    # h48/h72: short lags carry no signal at multi-day leads in tropical climates;
    # restrict to lags >= horizon to avoid overfitting to uncorrelated short-lag noise.
    48:  [48, 96, 120, 168, 240, 336],
    72:  [72, 96, 120, 168, 240, 336],
}
_ROLLING_BY_HORIZON: dict[int, list[int]] = {
    6:   [3, 6, 12, 24],
    12:  [3, 6, 12, 24, 48],
    24:  [6, 12, 24, 48, 72],
    48:  [48, 168, 240],
    72:  [72, 168, 240],
}


def _get_lags_for_horizon(horizon_h: int, explicit: list[int] | None = None) -> list[int]:
    """Return lag list: explicit if provided, else horizon-aware default."""
    if explicit is not None:
        return explicit
    return _LAGS_BY_HORIZON.get(horizon_h, _DEFAULT_LAGS_H)


def _get_rolling_for_horizon(horizon_h: int, explicit: list[int] | None = None) -> list[int]:
    """Return rolling window list: explicit if provided, else horizon-aware default."""
    if explicit is not None:
        return explicit
    return _ROLLING_BY_HORIZON.get(horizon_h, _DEFAULT_ROLLING_H)


def _subset_features_for_horizon(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    horizon_h: int,
    lags_h: list[int],
    rolling_h: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """Subset X to lag/rolling columns relevant to the given horizon, plus
    extended atmospheric lags, dynamics, and horizon-specific climatology.

    Args:
        X: Full feature matrix built with the union of all-horizon lags
        y: Target values
        horizon_h: Forecast horizon
        lags_h: Lag steps to keep for this horizon
        rolling_h: Rolling windows to keep for this horizon

    Returns:
        (X_subset, y) where X_subset contains horizon-relevant columns
    """
    lags_set = set(lags_h)
    rolls_set = set(rolling_h)

    # Collect matching columns by scanning X directly — this keeps any column
    # (including extended atmospheric, wind/precip, residuals) whose lag/rolling
    # index is in the allowed sets.
    lag_pat = re.compile(r"^(.+)_lag(\d+)h$")
    roll_pat = re.compile(r"^(.+)_roll(\d+)h_")

    matching_lag_roll: list[str] = []
    for col in X.columns:
        m = lag_pat.match(col)
        if m and int(m.group(2)) in lags_set:
            matching_lag_roll.append(col)
            continue
        m = roll_pat.match(col)
        if m and int(m.group(2)) in rolls_set:
            matching_lag_roll.append(col)

    # Core temporal / geo / raw-physics (no-leakage, always present)
    core_features = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
        "month_sin", "month_cos", "weekday",
        "station_enc", "lat", "lon", "elevation_m",
        "solar_wm2", "wind_ms", "cloud_pct", "pressure_hpa",
        "local_hour_sin", "local_hour_cos",
        "consecutive_hot_hours", "diurnal_amplitude_12h",
        # Thai seasonal features
        "monsoon_phase", "is_songkran", "season_transition",
        "is_hot_season", "is_rainy_season", "is_cool_season",
    ]

    # Horizon-specific target-time climatology buckets (diurnal signal for the
    # forecast target hour — strongest single predictor family after short lags)
    horizon_clim = [
        f"target_h{horizon_h}_hi_clim",
        f"target_h{horizon_h}_hour_sin",
        f"target_h{horizon_h}_hour_cos",
    ]

    # Short-window dynamics (non-lagged engineered cols)
    dynamics = [
        "cooling_3h", "trend_6h", "trend_24h",
        "temp_range_24h", "temp_x_evening",
        "hi_residual_lag1h",
    ]

    existing_cols = set(X.columns)
    keep_cols: list[str] = []
    for c in core_features + horizon_clim + dynamics:
        if c in existing_cols:
            keep_cols.append(c)
    keep_cols.extend(matching_lag_roll)
    keep_cols = list(dict.fromkeys(keep_cols))  # stable de-dup

    return X[keep_cols].copy(), y

# 3-entry LRU feature cache — keyed by (df hash, horizon_h)
_FEATURE_CACHE: dict = {}
_FEATURE_CACHE_ORDER: list = []
_FEATURE_CACHE_MAX = 10  # Increased from 3 to 10 for better cache hit rate

# Cross-horizon cache for build_X_once results — keyed by df hash only.
# Allows multiple horizons to reuse the expensive X matrix without recomputation.
_X_ONCE_CACHE: dict = {}
_X_ONCE_CACHE_ORDER: list = []
_X_ONCE_CACHE_MAX = 10  # Increased from 3 to 10 for better cache hit rate


def _df_hash(df: pd.DataFrame) -> str:
    h = _hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    return h


def _feature_cache_key(df: pd.DataFrame, horizon_h: int) -> str:
    return f"{_df_hash(df)}_{horizon_h}"


def add_heat_index_col(df: pd.DataFrame) -> pd.DataFrame:
    """Compute heat_index_c from temp_c + rh using the Rothfusz formula
    if the column is missing. Avoids importing the FastAPI app layer inside training."""
    if "heat_index_c" in df.columns:
        return df
    # Simplified Steadman for feature engineering (close enough for training features)
    T = df["temp_c"]
    R = df["rh"]
    hi = (-8.78469475556 +
           1.61139411 * T +
           2.33854883889 * R +
           -0.14611605 * T * R +
           -0.012308094 * T**2 +
           -0.0164248277778 * R**2 +
           0.002211732 * T**2 * R +
           0.00072546 * T * R**2 +
           -0.000003582 * T**2 * R**2)
    df = df.copy()
    df["heat_index_c"] = hi
    return df


def build_X_once(
    df: pd.DataFrame,
    lags_h: list[int] | None = None,
    rolling_h: list[int] | None = None,
    *,
    horizon_h: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature matrix X (no target) and return (X, df_augmented).

    df_augmented retains all engineered columns plus ts_utc/station_id so
    build_y_for_horizon can compute the target without re-reading parquet.
    X drops rows where any core (non-extended) feature is NaN.

    Args:
        df: DataFrame with columns ts_utc (datetime, UTC), station_id,
            temp_c, rh, heat_index_c (optional — computed if missing).
            Must be sorted by ts_utc, 1-hour frequency.
        lags_h: Lag steps in hours to include as features.
        rolling_h: Rolling window sizes in hours for mean and std features.

    Returns:
        (X, df_augmented) where X is the feature matrix with valid rows only
        and df_augmented retains all rows with all engineered columns.
    """
    if lags_h is None:
        lags_h = list(_DEFAULT_LAGS_H)
    if rolling_h is None:
        rolling_h = list(_DEFAULT_ROLLING_H)

    df = df.copy()
    df = add_heat_index_col(df)

    # Temporal features (derived from observation timestamp only)
    if not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    if "station_id" not in df.columns:
        df["station_id"] = "default"

    df = df.sort_values(["station_id", "ts_utc"]).reset_index(drop=True)
    station_group = df.groupby("station_id", sort=False)

    # --- B1: Physics features (computed per-row from current values, then lagged) ---
    # Magnus formula → dew-point (causal: uses only current row's temp_c and rh)
    _a_mag, _b_mag = 17.625, 243.04
    _gamma = (
        np.log(np.clip(df["rh"] / 100.0, 1e-9, 1.0))
        + _a_mag * df["temp_c"] / (_b_mag + df["temp_c"])
    )
    _dewpoint_c = _b_mag * _gamma / (_a_mag - _gamma)

    # Vapour Pressure Deficit (kPa)
    _es = 0.6108 * np.exp(17.27 * df["temp_c"] / (df["temp_c"] + 237.3))
    _vpd_kpa = _es * (1.0 - np.clip(df["rh"] / 100.0, 0.0, 1.0))

    # Stull (2011) wet-bulb approximation; valid domain T in [-20,50]°C, RH in [0,100].
    # Clamp to prevent NaN from sensor outliers propagating into lag/rolling features.
    _T, _R = np.clip(df["temp_c"], -20.0, 50.0), np.clip(df["rh"], 0.0, 100.0)
    _wbgt_stull_c = (
        _T * np.arctan(0.151977 * np.sqrt(_R + 8.313659))
        + np.arctan(_T + _R)
        - np.arctan(_R - 1.676331)
        + 0.00391838 * _R ** 1.5 * np.arctan(0.023101 * _R)
        - 4.686035
    )
    # NOTE: dewpoint_c, vpd_kpa, wbgt_stull_c are computed from current-row values
    # (causal). They will be lagged below (shift >= 1) and MUST NOT appear as
    # un-lagged raw columns in the output feature matrix X.

    _hour = df["ts_utc"].dt.hour
    _day_of_year = df["ts_utc"].dt.day_of_year
    _hour_sin = np.sin(2 * np.pi * _hour / 24)
    _hour_cos = np.cos(2 * np.pi * _hour / 24)
    local_hour = (_hour + 7) % 24
    _local_hour_sin = np.sin(2 * np.pi * local_hour / 24)
    _local_hour_cos = np.cos(2 * np.pi * local_hour / 24)
    _doy_sin = np.sin(2 * np.pi * _day_of_year / 365)
    _doy_cos = np.cos(2 * np.pi * _day_of_year / 365)
    _month = df["ts_utc"].dt.month
    _month_sin = np.sin(2 * np.pi * _month / 12)
    _month_cos = np.cos(2 * np.pi * _month / 12)
    _weekday = df["ts_utc"].dt.weekday

    # === Thai seasonal features (monsoon phases, seasonal transitions) ===
    # SW Monsoon: May-Oct (wet season), NE Monsoon: Nov-Apr (dry/cool season)
    _monsoon_phase = _month.map({
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1,   # SW Monsoon
        11: 2, 12: 2, 1: 2, 2: 2, 3: 2, 4: 2    # NE Monsoon
    }).fillna(0).astype(int)

    # Songkran period (mid-April) — historically hottest period
    _day = df["ts_utc"].dt.day
    _is_songkran = ((_month == 4) & (_day.between(10, 20))).astype(int)

    # Season transitions (Mar-May: hot→rainy, Oct: rainy→cool)
    _season_transition = _month.isin([3, 4, 5, 10]).astype(int)

    # Thai hot season flag (Mar-May)
    _is_hot_season = _month.isin([3, 4, 5]).astype(int)

    # Rainy season flag (Jun-Oct)
    _is_rainy_season = _month.isin([6, 7, 8, 9, 10]).astype(int)

    # Cool season flag (Nov-Feb)
    _is_cool_season = _month.isin([11, 12, 1, 2]).astype(int)

    # Station encoding (label encode)
    _station_enc = None
    if "station_id" in df.columns:
        _station_enc = df["station_id"].astype("category").cat.codes

    # Station geometry features — constant per-station lat/lon/elevation
    _lat = None
    _lon = None
    _elevation_m = None
    if "station_id" in df.columns:
        _lat = df["station_id"].map(
            lambda sid: STATIONS[sid].lat if sid in STATIONS else 0.0
        )
        _lon = df["station_id"].map(
            lambda sid: STATIONS[sid].lon if sid in STATIONS else 0.0
        )
        _elevation_m = df["station_id"].map(
            lambda sid: STATIONS[sid].elevation_m if sid in STATIONS else 0.0
        )

    # Batch add physics and temporal features
    _new_cols = {
        "dewpoint_c": _dewpoint_c,
        "vpd_kpa": _vpd_kpa,
        "wbgt_stull_c": _wbgt_stull_c,
        "hour": _hour,
        "day_of_year": _day_of_year,
        "hour_sin": _hour_sin,
        "hour_cos": _hour_cos,
        "local_hour_sin": _local_hour_sin,
        "local_hour_cos": _local_hour_cos,
        "doy_sin": _doy_sin,
        "doy_cos": _doy_cos,
        "month": _month,
        "month_sin": _month_sin,
        "month_cos": _month_cos,
        "weekday": _weekday,
        # Thai seasonal features
        "monsoon_phase": _monsoon_phase,
        "is_songkran": _is_songkran,
        "season_transition": _season_transition,
        "is_hot_season": _is_hot_season,
        "is_rainy_season": _is_rainy_season,
        "is_cool_season": _is_cool_season,
    }
    if _station_enc is not None:
        _new_cols["station_enc"] = _station_enc
    if _lat is not None:
        _new_cols["lat"] = _lat
        _new_cols["lon"] = _lon
        _new_cols["elevation_m"] = _elevation_m
    # Batch 1: Physics + temporal features
    df = pd.concat([df, pd.DataFrame(_new_cols, index=df.index)], axis=1)

    # Recreate station_group after adding new columns
    station_group = df.groupby("station_id", sort=False)

    # Batch 2: Lag + cooling + trend + interaction + rolling features
    _sid = df["station_id"]
    _lag_cols = {}
    for col in ["heat_index_c", "temp_c", "rh", "dewpoint_c", "wbgt_stull_c", "vpd_kpa"]:
        grp = station_group[col]
        for lag in lags_h:
            _lag_cols[f"{col}_lag{lag}h"] = grp.shift(lag)

    _hi_grp = station_group["heat_index_c"]
    _tc_grp = station_group["temp_c"]
    _cooling_cols = {
        "hi_change_3h": _hi_grp.shift(1) - _hi_grp.shift(4),
        "temp_change_3h": _tc_grp.shift(1) - _tc_grp.shift(4),
    }

    _trend_cols = {
        "temp_trend_6h": _tc_grp.shift(1) - _tc_grp.shift(7),
        "temp_trend_24h": _tc_grp.shift(1) - _tc_grp.shift(25),
    }

    _lag1_temp = _tc_grp.shift(1)
    _lag1_rh = station_group["rh"].shift(1)
    _interaction_cols_b2 = {
        "temp_rh_lag1": _lag1_temp * _lag1_rh,
        "temp_sq_lag1": _lag1_temp ** 2,
        "rh_sq_lag1": _lag1_rh ** 2,
    }

    _roll_cols = {}
    for col in ["heat_index_c", "temp_c", "rh", "dewpoint_c"]:
        shifted = station_group[col].shift(1)
        grp_shifted = shifted.groupby(_sid, sort=False)
        for w in rolling_h:
            _roll_cols[f"{col}_roll{w}h_mean"] = (
                grp_shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            _roll_cols[f"{col}_roll{w}h_std"] = (
                grp_shifted.rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            )

    # Single concat for all batch 2 columns
    _batch2_cols = {}
    _batch2_cols.update(_lag_cols)
    _batch2_cols.update(_cooling_cols)
    _batch2_cols.update(_trend_cols)
    _batch2_cols.update(_interaction_cols_b2)
    _batch2_cols.update(_roll_cols)
    df = pd.concat([df, pd.DataFrame(_batch2_cols, index=df.index)], axis=1)

    # Batch 3a: temp_range_24h, interaction features, consecutive_hot_hours, diurnal_amplitude
    _temp_range_24h = (
        df.groupby("station_id")["temp_c"]
        .transform(
            lambda x: (
                x.rolling(24, min_periods=1).max()
                - x.rolling(24, min_periods=1).min()
            ).shift(1)
        )
    )

    _tc_lag1 = station_group["temp_c"].shift(1)
    _temp_x_rh = _tc_lag1 * station_group["rh"].shift(1)

    local_hour_num = (df["ts_utc"].dt.hour + 7) % 24
    evening_mask = ((local_hour_num >= 16) & (local_hour_num <= 20)).astype(float)
    _temp_x_evening = _tc_lag1 * evening_mask

    _hot_flag = (station_group["temp_c"].shift(1) > 35.0).astype(float)
    _not_hot = (_hot_flag == 0).astype(float)
    _cumsum_reset = _not_hot.groupby(_sid, sort=False).cumsum()
    _hot_streak = _hot_flag.groupby([_sid, _cumsum_reset], sort=False).cumsum()

    _diurnal_amp = (
        df.groupby("station_id")["temp_c"]
        .transform(
            lambda x: (
                x.rolling(12, min_periods=1).max()
                - x.rolling(12, min_periods=1).min()
            ).shift(1)
        )
    )

    _batch3a_cols = {
        "temp_range_24h": _temp_range_24h,
        "temp_x_rh": _temp_x_rh,
        "temp_x_evening": _temp_x_evening,
        "consecutive_hot_hours": _hot_streak.values,
        "diurnal_amplitude_12h": _diurnal_amp,
    }
    df = pd.concat([df, pd.DataFrame(_batch3a_cols, index=df.index)], axis=1)

    # Extended atmospheric features — included when the column is present AND
    # at least 30% non-null. Missing values are forward-filled then backfilled
    # so lag creation doesn't explode with NaN.
    _EXTENDED_COLS = ["solar_wm2", "cloud_cover", "blh_m", "pressure_hpa", "lst_c"]
    _EXTENDED_LAGS = [1, 3, 6]  # shorter lag set for sparser extended variables
    _extended_cols = {}
    _filled_extended_sources = {}
    for col in _EXTENDED_COLS:
        if col not in df.columns:
            continue
        fill_rate = df[col].notna().mean()
        if fill_rate < 0.30:
            continue  # not enough data to be useful
        # Forward-fill gaps within station only. Do not backfill: that would use
        # future observations to fill earlier feature rows.
        filled = station_group[col].ffill()
        _filled_extended_sources[col] = filled
        filled_group = filled.groupby(_sid, sort=False)
        for lag in _EXTENDED_LAGS:
            _extended_cols[f"{col}_lag{lag}h"] = filled_group.shift(lag)
        # Rolling mean for solar (captures day trend) + solar-temp interaction
        if col == "solar_wm2":
            sol_shifted = filled_group.shift(1)
            _extended_cols["solar_wm2_roll6h_mean"] = (
                sol_shifted.groupby(_sid, sort=False).rolling(6, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
            _extended_cols["temp_x_solar"] = _tc_lag1 * sol_shifted
    if _filled_extended_sources or _extended_cols:
        df = pd.concat(
            [
                df.drop(columns=list(_filled_extended_sources), errors="ignore"),
                pd.DataFrame(_filled_extended_sources, index=df.index),
                pd.DataFrame(_extended_cols, index=df.index),
            ],
            axis=1,
        )

    # Wind and precipitation features — included when present AND ≥20% non-null.
    # Shorter lag set since these are typically sparser than core met variables.
    _WIND_PRECIP_COLS = ["wind_ms", "precip_mm"]
    _WIND_PRECIP_LAGS = [1, 3]
    _wind_precip_cols = {}
    _filled_wind_precip_sources = {}
    station_group = df.groupby("station_id", sort=False)
    for col in _WIND_PRECIP_COLS:
        if col not in df.columns:
            continue
        fill_rate = df[col].notna().mean()
        if fill_rate < 0.20:
            continue  # too sparse
        filled = station_group[col].ffill()
        _filled_wind_precip_sources[col] = filled
        filled_group = filled.groupby(_sid, sort=False)
        for lag in _WIND_PRECIP_LAGS:
            _wind_precip_cols[f"{col}_lag{lag}h"] = filled_group.shift(lag)

    # Always ensure lag columns exist for consistency with get_feature_names().
    # Filled with 0 when the source column was absent or too sparse.
    for col in _WIND_PRECIP_COLS:
        for lag in _WIND_PRECIP_LAGS:
            lag_col = f"{col}_lag{lag}h"
            if lag_col not in df.columns and lag_col not in _wind_precip_cols:
                _wind_precip_cols[lag_col] = 0.0
    if _filled_wind_precip_sources or _wind_precip_cols:
        df = pd.concat(
            [
                df.drop(columns=list(_filled_wind_precip_sources), errors="ignore"),
                pd.DataFrame(_filled_wind_precip_sources, index=df.index),
                pd.DataFrame(_wind_precip_cols, index=df.index),
            ],
            axis=1,
        )

    # Climatology residual — causal expanding mean per (station, month, hour)
    # bucket. For row at time t with bucket key K, _hi_clim[t] is the mean of
    # heat_index_c at all rows <= t that share K. Using transform("mean") here
    # would leak future bucket members into past rows (caught by
    # test_truncation_invariance_no_future_data_changes_past_features).
    # Rows are already sorted by (station_id, ts_utc) above, so a cumulative
    # mean within each bucket is causal. For the first occurrence of a bucket
    # the expanding mean equals heat_index_c itself; that is still leak-free.
    bucket = df.groupby(["station_id", "month", "hour"], sort=False)["heat_index_c"]
    _hi_clim = bucket.transform(lambda s: s.expanding(min_periods=1).mean())
    # Target-time climatology (independent, can batch with _clim_cols)
    _TARGET_HORIZONS = [6, 12, 24, 48, 72]
    _target_cols = {}
    for _h in _TARGET_HORIZONS:
        _target_ts = df["ts_utc"] + pd.Timedelta(hours=_h)
        _t_hour = _target_ts.dt.hour
        _t_month = _target_ts.dt.month
        _tbucket = df.groupby(["station_id", _t_month, _t_hour], sort=False)["heat_index_c"]
        _target_cols[f"target_h{_h}_hi_clim"] = _tbucket.transform(lambda s: s.expanding(min_periods=1).mean())
        _target_cols[f"target_h{_h}_hour_sin"] = np.sin(2 * np.pi * _t_hour / 24)
        _target_cols[f"target_h{_h}_hour_cos"] = np.cos(2 * np.pi * _t_hour / 24)

    _clim_cols = {"_hi_clim": _hi_clim}
    _clim_cols.update(_target_cols)
    df = pd.concat([df, pd.DataFrame(_clim_cols, index=df.index)], axis=1)

    # Recreate station_group after adding _hi_clim
    station_group = df.groupby("station_id", sort=False)

    _residual_cols = {
        "hi_residual_lag1h": (
            station_group["heat_index_c"].shift(1)
            - station_group["_hi_clim"].shift(1)
        ),
    }
    df = pd.concat([df, pd.DataFrame(_residual_cols, index=df.index)], axis=1)

    # Recreate station_group after adding residual columns
    station_group = df.groupby("station_id", sort=False)

    # Columns excluded from X (identifiers, raw targets, or now-promoted wind/precip
    # that appear only through their lag derivatives).
    # Physics intermediates (dewpoint_c, vpd_kpa, wbgt_stull_c) are excluded here
    # because they are current-row values — only their lagged versions belong in X.
    _NON_FEATURE = {"ts_utc", "station_id", "heat_index_c", "temp_c", "rh",
                    "hour", "day_of_year", "local_hour", "source",
                    "dewpoint_c", "vpd_kpa", "wbgt_stull_c"}
    feature_cols = [
        c for c in df.columns
        if c not in _NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])
    ]
    X_all = df[feature_cols].copy()

    # Extended + wind/precip columns can still have NaN after causal forward-fill.
    # Training code imputes these after splitting using train-only medians.
    _ALL_EXTENDED = list(_EXTENDED_COLS) + _WIND_PRECIP_COLS
    ext_feat_cols = [c for c in X_all.columns if any(c.startswith(e) for e in _ALL_EXTENDED)]

    core_valid = X_all[
        [c for c in X_all.columns if c not in ext_feat_cols]
    ].notna().all(axis=1)

    output_meta = df.loc[core_valid, ["ts_utc", "station_id"]].copy()
    X = X_all[core_valid].copy()

    order = output_meta.sort_values(["ts_utc", "station_id"]).index
    # Keep original positional index (subset of 0..N-1) so build_y_for_horizon
    # can use .loc[X.index] to select the exact same rows from df_augmented.
    X = X.loc[order]
    output_meta = output_meta.loc[order]
    X.attrs["ts_utc"] = output_meta["ts_utc"]
    X.attrs["station_id"] = output_meta["station_id"]

    # df_augmented: all rows with all engineered columns. build_y_for_horizon
    # uses X.index (= order) to select the matching target rows via .loc.
    df_augmented = df.copy()

    return X, df_augmented


def build_y_for_horizon(
    df_augmented: pd.DataFrame,
    X_valid_index: pd.Index,
    horizon_h: int,
    target_kind: Literal["hi", "th"] = "hi",
) -> pd.Series | pd.DataFrame:
    """Compute target y for a specific horizon from the cached augmented df.

    Returns y aligned to X_valid_index rows, with additional NaN rows excluded.
    The caller is responsible for intersecting X and y on the final valid mask:
        valid = y.notna().all(axis=1) if isinstance(y, pd.DataFrame) else y.notna()
        X_final = X.loc[valid]
        y_final = y.loc[valid]

    Args:
        df_augmented: The df_augmented returned by build_X_once (all rows, all cols).
        X_valid_index: The index of X returned by build_X_once (core-valid rows).
        horizon_h: How many hours ahead to predict.
        target_kind: "hi" for heat_index_c, "th" for (temp_c, rh) pair.

    Returns:
        y aligned to X_valid_index.
    """
    station_group = df_augmented.groupby("station_id", sort=False)

    if target_kind == "th":
        y_full = pd.DataFrame({
            "temp_c": station_group["temp_c"].shift(-horizon_h),
            "rh": station_group["rh"].shift(-horizon_h),
        })
    else:
        y_full = station_group["heat_index_c"].shift(-horizon_h)

    # df_augmented has a 0..N-1 RangeIndex (from build_X_once's reset_index).
    # X_valid_index contains label values from that same range, so .loc selects
    # the matching rows and preserves those labels as the output index.
    # This keeps y's index aligned with X's index for downstream boolean filtering.
    y = y_full.loc[X_valid_index] if len(X_valid_index) > 0 else y_full.iloc[[]]
    return y


def build_features(
    df: pd.DataFrame,
    horizon_h: int,
    lags_h: list[int] | None = None,
    rolling_h: list[int] | None = None,
    target_kind: Literal["hi", "th"] = "hi",
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """Build lag + rolling features for XGBoost heat-index forecast.

    Backward-compatible wrapper around build_X_once + build_y_for_horizon.
    Results are cached with a 3-entry LRU so multiple backends training on the
    same station share the computed feature matrix without recomputation.

    Args:
        df: DataFrame with columns ts_utc (datetime, UTC), station_id,
            temp_c, rh, heat_index_c (optional — computed if missing).
            Must be sorted by ts_utc, 1-hour frequency.
        horizon_h: How many hours ahead to predict.
        lags_h: Lag steps in hours. None → horizon-aware default.
        rolling_h: Rolling window sizes. None → horizon-aware default.
        target_kind: "hi" for heat_index_c scalar target, "th" for (temp_c, rh).

    Returns:
        (X, y) where y = heat_index_c shifted by -horizon_h.

    CRITICAL invariant: for every row i in X, the feature values only use
    data from df rows with ts_utc <= df.ts_utc[i]. The target y[i] uses
    df.ts_utc[i + horizon_h]. No leakage.
    """
    # F2: 3-entry LRU cache keyed by (df content hash, horizon_h).
    # Only cache when using horizon-aware defaults (no explicit override).
    _use_cache = (lags_h is None and rolling_h is None and target_kind == "hi")
    lags_h = _get_lags_for_horizon(horizon_h, lags_h)
    rolling_h = _get_rolling_for_horizon(horizon_h, rolling_h)
    if _use_cache:
        _ckey = _feature_cache_key(df, horizon_h)
        if _ckey in _FEATURE_CACHE:
            # Move to most-recently-used position
            _FEATURE_CACHE_ORDER.remove(_ckey)
            _FEATURE_CACHE_ORDER.append(_ckey)
            return _FEATURE_CACHE[_ckey]

    # Reuse build_X_once across horizons via cross-horizon cache
    # Cache key includes resolved lag/rolling so different horizons don't
    # incorrectly share the same X matrix (per-horizon lags differ now).
    _config_key = hash(tuple(lags_h) + tuple(rolling_h)) if _use_cache else None
    _xh_key = f"{_df_hash(df)}_{_config_key}" if _use_cache else None
    if _use_cache and _xh_key in _X_ONCE_CACHE:
        X, df_aug = _X_ONCE_CACHE[_xh_key]
    else:
        X, df_aug = build_X_once(df, lags_h, rolling_h)
        if _use_cache:
            _X_ONCE_CACHE[_xh_key] = (X, df_aug)
            _X_ONCE_CACHE_ORDER.append(_xh_key)
            while len(_X_ONCE_CACHE_ORDER) > _X_ONCE_CACHE_MAX:
                _evict = _X_ONCE_CACHE_ORDER.pop(0)
                _X_ONCE_CACHE.pop(_evict, None)

    y = build_y_for_horizon(df_aug, X.index, horizon_h, target_kind)
    valid = y.notna().all(axis=1) if isinstance(y, pd.DataFrame) else y.notna()
    result = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    if _use_cache:
        _FEATURE_CACHE[_ckey] = result
        _FEATURE_CACHE_ORDER.append(_ckey)
        # Evict LRU entry when over capacity
        while len(_FEATURE_CACHE_ORDER) > _FEATURE_CACHE_MAX:
            _evict = _FEATURE_CACHE_ORDER.pop(0)
            _FEATURE_CACHE.pop(_evict, None)

    return result


def get_feature_names(
    lags_h: list[int] | None = None,
    rolling_h: list[int] | None = None,
    *,
    horizon_h: int = 24,
) -> list[str]:
    """Return the expected feature column names (for validation at predict time)."""
    lags_h = _get_lags_for_horizon(horizon_h, lags_h)
    rolling_h = _get_rolling_for_horizon(horizon_h, rolling_h)
    names = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "station_enc",
        # Station geometry — always present (filled with 0.0 for unknown stations)
        "lat", "lon", "elevation_m",
    ]
    # Core lags: heat_index_c, temp_c, rh (B1/B4: also dewpoint_c, wbgt_stull_c, vpd_kpa)
    for col in ["heat_index_c", "temp_c", "rh", "dewpoint_c", "wbgt_stull_c", "vpd_kpa"]:
        for lag in lags_h:
            names.append(f"{col}_lag{lag}h")
    # Rolling: heat_index_c, temp_c (B2: also rh and dewpoint_c)
    for col in ["heat_index_c", "temp_c", "rh", "dewpoint_c"]:
        for w in rolling_h:
            names.append(f"{col}_roll{w}h_mean")
            names.append(f"{col}_roll{w}h_std")
    # B2: temp_range_24h scalar feature
    names.append("temp_range_24h")
    # v2 features — always present regardless of extended column availability
    names += [
        "local_hour_sin", "local_hour_cos",
        "hi_change_3h", "temp_change_3h",
        "temp_x_rh",
        "temp_x_evening",
        "consecutive_hot_hours", "diurnal_amplitude_12h",
    ]
    # Optional/extended features — present only when the source column has ≥20–30% fill
    # These are conditioned on column presence and fill rate at training time.
    names += [
        "wind_ms_lag1h", "wind_ms_lag3h",
        "precip_mm_lag1h", "precip_mm_lag3h",
    ]
    # Target-time climatology — B3: extended to all five horizons.
    for _h in [6, 12, 24, 48, 72]:
        names += [
            f"target_h{_h}_hi_clim",
            f"target_h{_h}_hour_sin",
            f"target_h{_h}_hour_cos",
        ]
    return names
