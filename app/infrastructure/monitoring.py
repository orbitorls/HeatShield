"""Production monitoring for HeatShield AI prediction pipeline.

Tracks prediction drift, data quality, and model performance metrics.
Provides alerting hooks for external monitoring systems.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_METRICS_DIR = Path(__file__).parents[2] / "logs" / "metrics"
_METRICS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionMetrics:
    """Snapshot of a single prediction request for monitoring."""

    station_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_version: str = "unknown"
    horizons_requested: list[int] = field(default_factory=list)
    horizons_predicted: int = 0
    latency_ms: float = 0.0
    low_confidence: bool = False
    confidence_reason: str | None = None
    data_source: str = "unknown"  # tmd, nasa_power, era5, cache
    staleness_min: float = 0.0
    pi_width_max: float = 0.0
    pi_width_avg: float = 0.0
    temp_range: tuple[float, float] | None = None
    hi_max: float | None = None
    hi_min: float | None = None
    fallback_used: bool = False  # NASA POWER fallback

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "horizons_requested": self.horizons_requested,
            "horizons_predicted": self.horizons_predicted,
            "latency_ms": round(self.latency_ms, 1),
            "low_confidence": self.low_confidence,
            "confidence_reason": self.confidence_reason,
            "data_source": self.data_source,
            "staleness_min": round(self.staleness_min, 1),
            "pi_width_max": round(self.pi_width_max, 2),
            "pi_width_avg": round(self.pi_width_avg, 2),
            "fallback_used": self.fallback_used,
            **({
                "temp_min": round(self.temp_range[0], 1),
                "temp_max": round(self.temp_range[1], 1),
            } if self.temp_range else {}),
            **({"hi_max": round(self.hi_max, 1)} if self.hi_max is not None else {}),
            **({"hi_min": round(self.hi_min, 1)} if self.hi_min is not None else {}),
        }


class MonitoringCollector:
    """Collects and stores prediction metrics for drift detection."""

    def __init__(self, window_size: int = 1000) -> None:
        self._window_size = window_size
        self._history: list[PredictionMetrics] = []

    def record(self, metrics: PredictionMetrics) -> None:
        """Record a prediction event."""
        self._history.append(metrics)
        if len(self._history) > self._window_size:
            self._history = self._history[-self._window_size:]

        # Write to daily metrics file
        self._write_daily(metrics)

    def _write_daily(self, metrics: PredictionMetrics) -> None:
        path = _METRICS_DIR / f"metrics_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")

    def get_recent_stats(self, n: int = 100) -> dict[str, Any]:
        """Return summary stats over the last N predictions."""
        recent = self._history[-n:] if n < len(self._history) else self._history
        if not recent:
            return {}

        return {
            "count": len(recent),
            "low_confidence_rate": sum(1 for m in recent if m.low_confidence) / len(recent),
            "avg_latency_ms": np.mean([m.latency_ms for m in recent]),
            "fallback_rate": sum(1 for m in recent if m.fallback_used) / len(recent),
            "avg_staleness_min": np.mean([m.staleness_min for m in recent]),
            "avg_pi_width": np.mean([m.pi_width_avg for m in recent]),
        }

    def check_drift(self) -> list[str]:
        """Check for prediction drift and return alert messages."""
        alerts: list[str] = []
        if len(self._history) < 100:
            return alerts

        # Split into recent and reference windows
        n = len(self._history)
        recent = self._history[n // 2:]
        reference = self._history[:n // 2]

        # PI width drift
        recent_pi = np.mean([m.pi_width_avg for m in recent])
        ref_pi = np.mean([m.pi_width_avg for m in reference])
        if ref_pi > 0 and recent_pi > ref_pi * 1.3:
            alerts.append(
                f"PI width drift: recent={recent_pi:.2f} vs reference={ref_pi:.2f}"
            )

        # Latency drift
        recent_lat = np.mean([m.latency_ms for m in recent])
        ref_lat = np.mean([m.latency_ms for m in reference])
        if ref_lat > 0 and recent_lat > ref_lat * 2:
            alerts.append(
                f"Latency drift: recent={recent_lat:.0f}ms vs reference={ref_lat:.0f}ms"
            )

        # Fallback rate increase
        recent_fb = sum(1 for m in recent if m.fallback_used) / len(recent)
        ref_fb = sum(1 for m in reference if m.fallback_used) / len(reference)
        if recent_fb > ref_fb * 2 and recent_fb > 0.1:
            alerts.append(
                f"Fallback rate increase: recent={recent_fb:.1%} vs reference={ref_fb:.1%}"
            )

        return alerts


# Global collector
_metrics_collector = MonitoringCollector()


def get_collector() -> MonitoringCollector:
    return _metrics_collector


def record_prediction(
    station_id: str,
    model_version: str,
    horizons: list[int],
    forecasts: list,
    latency_ms: float,
    low_confidence: bool,
    confidence_reason: str | None,
    data_source: str = "unknown",
    staleness_min: float = 0.0,
    fallback_used: bool = False,
) -> None:
    """Record a prediction event with all relevant metrics."""
    hi_values = [f.heat_index_c for f in forecasts]
    pi_widths = [
        f.pi_upper - f.pi_lower for f in forecasts
        if hasattr(f, "pi_upper") and hasattr(f, "pi_lower")
    ]
    temps = [f.temp_c for f in forecasts if hasattr(f, "temp_c")]

    metrics = PredictionMetrics(
        station_id=station_id,
        model_version=model_version,
        horizons_requested=horizons,
        horizons_predicted=len(forecasts),
        latency_ms=latency_ms,
        low_confidence=low_confidence,
        confidence_reason=confidence_reason,
        data_source=data_source,
        staleness_min=staleness_min,
        pi_width_max=max(pi_widths) if pi_widths else 0.0,
        pi_width_avg=np.mean(pi_widths) if pi_widths else 0.0,
        temp_range=(min(temps), max(temps)) if temps else None,
        hi_max=max(hi_values) if hi_values else None,
        hi_min=min(hi_values) if hi_values else None,
        fallback_used=fallback_used,
    )
    _metrics_collector.record(metrics)

    # Log any drift alerts
    alerts = _metrics_collector.check_drift()
    for alert in alerts:
        logger.warning("[DRIFT] %s", alert)


def get_health_report() -> dict[str, Any]:
    """Return current system health summary."""
    stats = _metrics_collector.get_recent_stats(n=1000)
    alerts = _metrics_collector.check_drift()
    return {
        "status": "healthy" if not alerts else "degraded",
        "alerts": alerts,
        **stats,
    }
