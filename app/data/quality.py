"""Data quality gate — validate and filter TMD observations before storage/training."""
from __future__ import annotations

import math

from app.data.schemas import StationObservation


_BOUNDS = {
    "temp_c": (-5.0, 50.0),
    "rh": (1.0, 100.0),
    "wind_ms": (0.0, 80.0),
    "precip_mm": (0.0, 500.0),
}


def validate_observation(obs: StationObservation) -> tuple[bool, list[str]]:
    """Return (is_valid, reasons_if_invalid)."""
    reasons: list[str] = []

    for field, (lo, hi) in _BOUNDS.items():
        val = getattr(obs, field)
        if val is not None and not (lo <= val <= hi):
            reasons.append(f"{field}={val} out of range [{lo}, {hi}]")

    # Dewpoint cannot exceed temperature (physics)
    try:
        dewpoint = _dewpoint(obs.temp_c, obs.rh)
        if dewpoint > obs.temp_c + 0.5:
            reasons.append(f"dewpoint {dewpoint:.1f}°C > temp {obs.temp_c}°C")
    except (ValueError, TypeError) as exc:
        reasons.append(f"Invalid dewpoint calculation: {exc}")

    return len(reasons) == 0, reasons


def filter_observations(
    obs: list[StationObservation],
) -> tuple[list[StationObservation], list[dict]]:
    """Filter out invalid observations. Returns (kept, dropped_with_reasons)."""
    kept: list[StationObservation] = []
    dropped: list[dict] = []
    for o in obs:
        valid, reasons = validate_observation(o)
        if valid:
            kept.append(o)
        else:
            dropped.append(
                {
                    "ts_utc": o.ts_utc.isoformat(),
                    "station_id": o.station_id,
                    "reasons": reasons,
                }
            )
    return kept, dropped


def _dewpoint(temp_c: float, rh: float) -> float:
    """Magnus formula approximation."""
    a, b = 17.27, 237.7
    alpha = (a * temp_c) / (b + temp_c) + math.log(rh / 100.0)
    return (b * alpha) / (a - alpha)
