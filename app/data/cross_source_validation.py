"""Cross-source data validation for HeatShield AI.

Compares observations from multiple data sources (TMD, ERA5, NASA POWER)
and flags discrepancies that may indicate sensor errors or data quality issues.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from app.data.loaders import read_observations
from app.data.stations import STATIONS

logger = logging.getLogger(__name__)

# Thresholds for flagging cross-source discrepancies
_TEMP_DIVERGENCE_THRESHOLD_C = 3.0  # Flag if sources differ by > 3°C
_RH_DIVERGENCE_THRESHOLD_PCT = 15.0  # Flag if RH differs by > 15%
_MIN_OVERLAP_HOURS = 6  # Minimum overlapping hours to validate


@dataclass
class SourceComparisonResult:
    """Result of comparing two data sources for a station-day."""

    station_id: str
    date: date
    source_a: str
    source_b: str
    overlap_hours: int
    temp_bias_c: float  # mean(source_a - source_b) for temperature
    rh_bias_pct: float  # mean(source_a - source_b) for RH
    temp_max_diff_c: float  # max absolute difference
    rh_max_diff_pct: float  # max absolute difference
    is_valid: bool  # True if within thresholds
    issues: list[str]  # List of detected issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "date": self.date.isoformat(),
            "source_a": self.source_a,
            "source_b": self.source_b,
            "overlap_hours": self.overlap_hours,
            "temp_bias_c": round(self.temp_bias_c, 2),
            "rh_bias_pct": round(self.rh_bias_pct, 2),
            "temp_max_diff_c": round(self.temp_max_diff_c, 2),
            "rh_max_diff_pct": round(self.rh_max_diff_pct, 2),
            "is_valid": self.is_valid,
            "issues": self.issues,
        }


def _get_source_df(station_id: str, day: date, source: str, *, dedupe_sources: bool = False) -> pd.DataFrame | None:
    """Read observations for a specific source only.

    Args:
        station_id: Station identifier
        day: Date to read
        source: Source name to filter (e.g., "tmd", "era5", "nasa_power")
        dedupe_sources: If False, preserve all sources (for cross-source comparison)
    """
    df = read_observations(station_id, day, day, dedupe_sources=dedupe_sources)
    if df.empty:
        return None
    # Filter to specific source
    if "source" in df.columns:
        df = df[df["source"] == source].copy()
    if df.empty:
        return None
    return df


def compare_sources(
    station_id: str,
    day: date,
    source_a: str = "tmd",
    source_b: str = "nasa_power",
) -> SourceComparisonResult | None:
    """Compare observations from two data sources for the same station and day.

    Returns None if either source has no data for the day.
    Uses raw mode (dedupe_sources=False) to preserve all sources for comparison.
    """
    df_a = _get_source_df(station_id, day, source_a, dedupe_sources=False)
    df_b = _get_source_df(station_id, day, source_b, dedupe_sources=False)

    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return None

    # Align timestamps to the hour
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["ts_utc"] = pd.to_datetime(df_a["ts_utc"], utc=True).dt.floor("h")
    df_b["ts_utc"] = pd.to_datetime(df_b["ts_utc"], utc=True).dt.floor("h")

    merged = pd.merge(
        df_a[["ts_utc", "temp_c", "rh"]],
        df_b[["ts_utc", "temp_c", "rh"]],
        on="ts_utc",
        suffixes=(f"_{source_a}", f"_{source_b}"),
        how="inner",
    )

    if merged.empty:
        return None

    temp_diff = merged[f"temp_c_{source_a}"] - merged[f"temp_c_{source_b}"]
    rh_diff = merged[f"rh_{source_a}"] - merged[f"rh_{source_b}"]

    issues = []
    if temp_diff.abs().max() > 5.0:
        issues.append(f"Max temp diff {temp_diff.abs().max():.2f}°C exceeds threshold")
    if rh_diff.abs().max() > 20.0:
        issues.append(f"Max RH diff {rh_diff.abs().max():.2f}% exceeds threshold")

    return SourceComparisonResult(
        station_id=station_id,
        date=day,
        source_a=source_a,
        source_b=source_b,
        overlap_hours=len(merged),
        temp_bias_c=float(temp_diff.mean()),
        rh_bias_pct=float(rh_diff.mean()),
        temp_max_diff_c=float(temp_diff.abs().max()),
        rh_max_diff_pct=float(rh_diff.abs().max()),
        is_valid=len(issues) == 0,
        issues=issues,
    )


def validate_station_day(
    station_id: str,
    day: date,
) -> dict[str, Any]:
    """Validate all available data sources for a station-day.

    Returns a summary dict with validation results.
    """
    if station_id not in STATIONS:
        return {"error": f"Unknown station {station_id}"}

    results: list[dict] = []
    sources = [("tmd", "nasa_power"), ("tmd", "era5")]

    for a, b in sources:
        result = compare_sources(station_id, day, a, b)
        if result:
            results.append(result.to_dict())

    # Also check for basic data availability
    df_all = read_observations(station_id, day, day)
    total_hours = len(df_all) if not df_all.empty else 0
    expected_hours = 24
    coverage = total_hours / expected_hours if expected_hours > 0 else 0.0

    missing_sources = []
    for src in ["tmd", "era5", "nasa_power"]:
        df_src = _get_source_df(station_id, day, src, dedupe_sources=False)
        if df_src is None or len(df_src) < 6:
            missing_sources.append(src)

    return {
        "station_id": station_id,
        "date": day.isoformat(),
        "total_hours": total_hours,
        "coverage_pct": round(coverage * 100, 1),
        "missing_sources": missing_sources,
        "source_comparisons": results,
        "has_divergence": any(not r["is_valid"] for r in results),
    }


def validate_all_stations(
    day: date,
    stations: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate data quality for all stations on a given day."""
    if stations is None:
        stations = list(STATIONS.keys())

    results = []
    for sid in stations:
        result = validate_station_day(sid, day)
        results.append(result)

        if result.get("has_divergence"):
            logger.warning(
                "Data divergence detected for %s on %s", sid, day.isoformat()
            )
        if result.get("coverage_pct", 100) < 80:
            logger.warning(
                "Low coverage for %s on %s: %.1f%%",
                sid, day.isoformat(), result.get("coverage_pct", 0),
            )

    return results
