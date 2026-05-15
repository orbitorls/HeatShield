"""ML Forecast Module

LightGBM-based heat risk forecasting with confidence intervals.
This module provides machine learning predictions for temperature
and humidity patterns, enabling more accurate risk assessments.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class MLForecastResult:
    """Result of ML-based forecasting"""
    temperature: float
    humidity: float
    confidence: float  # 0-1 confidence level
    uncertainty_low: float
    uncertainty_high: float
    model_version: str
    prediction_type: str  # "ml" or "rule_based_fallback"


class MLForecastModel:
    """LightGBM-based forecast model (stub implementation)

    In production, this would load and use a trained LightGBM model.
    For now, provides rule-based fallback with ML-style confidence scores.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_available = False
        self.model_version = "1.0.0-stub"

    def predict(
        self,
        temps: List[float],
        humidities: List[float],
        group_id: str = "elementary"
    ) -> List[MLForecastResult]:
        """Generate ML predictions for hourly data.

        Args:
            temps: Hourly temperatures
            humidities: Hourly humidities
            group_id: Vulnerability group identifier

        Returns:
            List of MLForecastResult for each hour
        """
        results = []
        for temp, humidity in zip(temps, humidities):
            # Calculate confidence based on data quality
            # More extreme values = lower confidence
            confidence = self._calculate_confidence(temp, humidity)

            # Uncertainty range based on confidence
            uncertainty_range = (1 - confidence) * 5  # Max ±5°C at 0 confidence

            results.append(MLForecastResult(
                temperature=round(temp, 1),
                humidity=round(humidity, 1),
                confidence=round(confidence, 2),
                uncertainty_low=round(temp - uncertainty_range, 1),
                uncertainty_high=round(temp + uncertainty_range, 1),
                model_version=self.model_version,
                prediction_type="rule_based_fallback" if not self.is_available else "ml"
            ))

        return results

    def predict_risk(
        self,
        temps: List[float],
        humidities: List[float],
        group_id: str = "elementary",
        activity_level: str = "moderate"
    ) -> List[Dict[str, Any]]:
        """Generate ML-based risk predictions.

        Args:
            temps: Hourly temperatures
            humidities: Hourly humidities
            group_id: Vulnerability group
            activity_level: Activity intensity

        Returns:
            List of risk predictions with ML confidence
        """
        predictions = self.predict(temps, humidities, group_id)

        from src.core.risk_engine import VULNERABILITY_GROUPS, calculate_adaptive_risk

        group = VULNERABILITY_GROUPS.get(group_id, VULNERABILITY_GROUPS["general"])

        results = []
        for pred in predictions:
            # Use adaptive risk calculation for consistency
            risk = calculate_adaptive_risk(
                temp=pred.temperature,
                humidity=pred.humidity,
                group=group,
                exposure_minutes=60,
                activity_level=activity_level
            )

            results.append({
                "hour": len(results),
                "temperature": pred.temperature,
                "humidity": pred.humidity,
                "risk_score": risk["total_score"],
                "level": risk["level"],
                "ml_confidence": pred.confidence,
                "ml_model_version": pred.model_version,
                "uncertainty_range": {
                    "low": pred.uncertainty_low,
                    "high": pred.uncertainty_high
                }
            })

        return results

    def _calculate_confidence(self, temp: float, humidity: float) -> float:
        """Calculate prediction confidence based on conditions.

        Higher confidence for typical conditions,
        lower confidence for extreme conditions.
        """
        # Typical range: 25-35°C, 40-80% humidity
        temp_score = 1.0 - abs(temp - 30) / 20  # Max deviation 20°C
        humidity_score = 1.0 - abs(humidity - 60) / 50  # Max deviation 50%

        confidence = (temp_score * 0.6 + humidity_score * 0.4)
        return max(0.3, min(0.95, confidence))  # Clamp between 0.3 and 0.95


def get_ml_forecast(
    temps: List[float],
    humidities: List[float],
    group_id: str = "elementary",
    activity_level: str = "moderate"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate ML forecast with fallback.

    Args:
        temps: Hourly temperatures
        humidities: Hourly humidities
        group_id: Vulnerability group
        activity_level: Activity intensity

    Returns:
        Tuple of (predictions list, metadata dict)
    """
    model = MLForecastModel()

    predictions = model.predict_risk(temps, humidities, group_id, activity_level)

    avg_confidence = np.mean([p["ml_confidence"] for p in predictions])

    metadata = {
        "model_available": model.is_available,
        "model_version": model.model_version,
        "average_confidence": round(avg_confidence, 2),
        "prediction_type": "ml" if model.is_available else "rule_based_fallback",
        "fallback_reason": None if model.is_available else "ml_model_not_trained"
    }

    return predictions, metadata


def get_fallback_risk_forecast(
    temps: List[float],
    humidities: List[float],
    group_id: str = "elementary",
    activity_level: str = "moderate"
) -> List[Dict[str, Any]]:
    """Fallback rule-based forecast when ML is unavailable.

    Uses existing risk engine with confidence scoring.
    """
    from src.core.risk_engine import VULNERABILITY_GROUPS, get_hourly_forecast_risk

    group = VULNERABILITY_GROUPS.get(group_id, VULNERABILITY_GROUPS["general"])
    hourly_risks = get_hourly_forecast_risk(temps, humidities, group, activity_level)

    # Add confidence scores
    for risk in hourly_risks:
        risk["ml_confidence"] = 0.5  # Lower confidence for rule-based
        risk["ml_model_version"] = "fallback-1.0"
        risk["uncertainty_range"] = {
            "low": risk["temperature"] - 2,
            "high": risk["temperature"] + 2
        }

    return hourly_risks