"""Parquet I/O helpers for weather observations."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from app.data.schemas import StationObservation

_RAW_ROOT = Path(__file__).parents[2] / "data" / "raw"

# Simple LRU cache for read_observations to avoid re-reading parquet across horizons.
# Key: (station_id, start_iso, end_iso, normalize_hourly_grid, dedupe_sources)
_READ_CACHE: dict = {}
_READ_CACHE_ORDER: list = []
_READ_CACHE_MAX = 20  # Covers 5 stations x 4 year-ranges comfortably


def _partition_path(station_id: str, day: date) -> Path:
    return _RAW_ROOT / f"station_id={station_id}" / f"date={day.isoformat()}" / "obs.parquet"


def write_observations(obs: list[StationObservation], station_id: str, day: date) -> Path:
    """Write observations to partitioned parquet. Idempotent (overwrites same partition)."""
    if not obs:
        return _partition_path(station_id, day)

    records = [o.model_dump() for o in obs]
    df = pd.DataFrame(records)
    path = _partition_path(station_id, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return path


def read_observations(
    station_id: str,
    start: date,
    end: date,
    *,
    dedupe_sources: bool = True,
    normalize_hourly_grid: bool = False,
) -> pd.DataFrame:
    """Read all parquet partitions for a station between start and end (inclusive).

    Returns DataFrame with columns matching StationObservation fields.
    Returns empty DataFrame if no data found.

    Args:
        station_id: Station identifier
        start: Start date (inclusive)
        end: End date (inclusive)
        dedupe_sources: If True, keep only highest-priority source per timestamp.
            Set to False for cross-source validation to preserve all sources.
        normalize_hourly_grid: If True, reindex to hourly grid and add gap markers.
            This ensures lag features are time-aware (lag24h = 24 hours ago, not 24 rows).
    """
    # Check LRU cache first
    _cache_key = (station_id, start.isoformat(), end.isoformat(), normalize_hourly_grid, dedupe_sources)
    if _cache_key in _READ_CACHE:
        _READ_CACHE_ORDER.remove(_cache_key)
        _READ_CACHE_ORDER.append(_cache_key)
        return _READ_CACHE[_cache_key].copy()

    if not _RAW_ROOT.exists():
        return pd.DataFrame()

    try:
        dataset = ds.dataset(_RAW_ROOT, format="parquet", partitioning="hive")
        filt = (
            (ds.field("station_id") == station_id)
            & (ds.field("date") >= start.isoformat())
            & (ds.field("date") <= end.isoformat())
        )
        table = dataset.to_table(filter=filt)
        if table.num_rows == 0:
            return pd.DataFrame()
        df = table.to_pandas()
    except Exception:
        # pyarrow dataset can fail with ArrowTypeError when parquet schemas differ
        # (e.g. large_string vs string). Fallback to manual glob in all cases.
        frames: list[pd.DataFrame] = []
        station_dir = _RAW_ROOT / f"station_id={station_id}"
        if not station_dir.exists():
            return pd.DataFrame()
        for path in station_dir.glob("date=*/obs.parquet"):
            day_str = path.parent.name.removeprefix("date=")
            try:
                day = date.fromisoformat(day_str)
            except ValueError:
                continue
            if start <= day <= end:
                frames.append(pd.read_parquet(path, engine="pyarrow"))

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
    df = df.dropna(axis=1, how="all")
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    if dedupe_sources and "source" in df.columns and df["source"].nunique() > 1:
        # TMD > ERA5 > NASA POWER: highest fidelity (ground truth) wins
        source_order = pd.CategoricalDtype(["nasa_power", "era5", "tmd"], ordered=True)
        df["source"] = df["source"].astype(source_order)
        df = (df.sort_values(["ts_utc", "source"])
                .drop_duplicates(subset=["station_id", "ts_utc"], keep="last")
                .reset_index(drop=True))
    else:
        df = df.sort_values("ts_utc").reset_index(drop=True)

    if normalize_hourly_grid and not df.empty and "ts_utc" in df.columns:
        df = _normalize_to_hourly_grid(df)

    # Store in LRU cache (module-level dicts mutated in-place, no global needed)
    _READ_CACHE[_cache_key] = df
    _READ_CACHE_ORDER.append(_cache_key)
    while len(_READ_CACHE_ORDER) > _READ_CACHE_MAX:
        _old = _READ_CACHE_ORDER.pop(0)
        _READ_CACHE.pop(_old, None)

    return df


def _normalize_to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex DataFrame to hourly grid and add gap markers.

    This ensures that lag features are time-aware: lag24h refers to data
    from exactly 24 hours ago, not 24 rows ago when there are gaps.

    Preserves gap markers:
    - is_gap: True for rows with no actual observation (filled with NaN)
    - gap_hours_since_last_obs: Hours since the last real observation
    - source: Preserved for observed rows, set to 'filled' for gap rows

    Args:
        df: DataFrame with ts_utc column, sorted by ts_utc

    Returns:
        DataFrame reindexed to hourly grid with gap markers
    """
    if df.empty:
        return df

    # Ensure datetime and sort
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Create hourly grid for each station
    stations = df["station_id"].unique() if "station_id" in df.columns else ["default"]
    frames: list[pd.DataFrame] = []

    for station in stations:
        if "station_id" in df.columns:
            station_df = df[df["station_id"] == station].copy()
        else:
            station_df = df.copy()

        if station_df.empty:
            continue

        # Determine time range with padding for max lags
        min_ts = station_df["ts_utc"].min()
        max_ts = station_df["ts_utc"].max()
        full_range = pd.date_range(
            start=min_ts.floor("h"),
            end=max_ts.ceil("h"),
            freq="h",
            tz="UTC"
        )

        # Reindex to hourly grid
        station_df = station_df.set_index("ts_utc")
        station_df = station_df.reindex(full_range)

        # Mark gaps
        station_df["is_gap"] = station_df.index.isin(df["ts_utc"]).astype(int)
        station_df["is_gap"] = 1 - station_df["is_gap"]  # 1 if gap, 0 if observed

        # Calculate hours since last observation (vectorized NumPy loop)
        # Matches original semantics exactly: observed rows -> 0, gaps -> count since last obs.
        is_gap_arr = station_df["is_gap"].to_numpy()
        gap_hours = np.empty_like(is_gap_arr, dtype=int)
        counter = 0
        for i in range(len(is_gap_arr)):
            counter = 0 if is_gap_arr[i] == 0 else counter + 1
            gap_hours[i] = counter
        station_df["gap_hours_since_last_obs"] = gap_hours

        # Preserve source for observed rows, mark filled rows
        if "source" in station_df.columns:
            # Handle categorical columns by adding 'filled' to categories
            if pd.api.types.is_categorical_dtype(station_df["source"]):
                if "filled" not in station_df["source"].cat.categories:
                    station_df["source"] = station_df["source"].cat.add_categories("filled")
            station_df["source"] = station_df["source"].fillna("filled")

        # Reset index to get ts_utc back as column
        station_df = station_df.reset_index()
        station_df = station_df.rename(columns={"index": "ts_utc"})

        frames.append(station_df)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["station_id", "ts_utc"]).reset_index(drop=True)
        return result

    return df
