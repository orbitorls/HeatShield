"""Ensemble Forecast Module

Combines multiple forecasting methods (ML + rule-based) with
weighted blending and confidence scoring for robust predictions.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from src.core.risk_engine import (
    VULNERABILITY_GROUPS,
    VulnerabilityGroup,
    get_hourly_forecast_risk,
    calculate_adaptive_risk
)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble forecasting"""
    ml_weight: float = 0.6  # Weight for ML predictions
    rule_weight: float = 0.4  # Weight for rule-based predictions
    confidence_threshold: float = 0.7  # Min confidence to trust ML
    adaptivity_enabled: bool = True


@dataclass
class TimeSlot:
    """Time slot for activity planning"""
    hour: int
    start_time: str
    end_time: str
    risk_score: float
    level: str
    temperature: float
    humidity: float
    confidence: float
    is_dangerous: bool
    reason: str


@dataclass
class EnsembleResult:
    """Result of ensemble forecasting"""
    predictions: List[Dict[str, Any]]
    blended_forecast: List[Dict[str, Any]]
    confidence_scores: List[float]
    weights_used: Dict[str, float]
    metadata: Dict[str, Any]


def blend_predictions(
    ml_predictions: List[Dict],
    rule_predictions: List[Dict],
    config: Optional[EnsembleConfig] = None
) -> List[Dict[str, Any]]:
    """Blend ML and rule-based predictions.

    Args:
        ml_predictions: ML-based predictions
        rule_predictions: Rule-based predictions
        config: Ensemble configuration

    Returns:
        Blended predictions with confidence scores
    """
    if config is None:
        config = EnsembleConfig()

    blended = []

    for ml_pred, rule_pred in zip(ml_predictions, rule_predictions):
        ml_conf = ml_pred.get("ml_confidence", 0.5)

        # Adjust weights based on ML confidence
        if ml_conf >= config.confidence_threshold:
            effective_ml_weight = config.ml_weight
        else:
            # Reduce ML weight when confidence is low
            factor = ml_conf / config.confidence_threshold
            effective_ml_weight = config.ml_weight * factor

        effective_rule_weight = 1.0 - effective_ml_weight

        # Blend risk scores
        ml_risk = ml_pred.get("risk_score", rule_pred["risk_score"])
        rule_risk = rule_pred.get("risk_score", ml_pred["risk_score"])

        blended_risk = (
            effective_ml_weight * ml_risk +
            effective_rule_weight * rule_risk
        )

        # Average confidence (weighted by effective weights)
        combined_confidence = (
            effective_ml_weight * ml_conf +
            effective_rule_weight * 0.5  # Rule-based has fixed 0.5 confidence
        )

        from src.core.risk_engine import get_risk_level

        blended.append({
            "hour": ml_pred.get("hour", rule_pred.get("hour")),
            "temperature": (ml_pred.get("temperature") + rule_pred.get("temperature")) / 2,
            "humidity": (ml_pred.get("humidity") + rule_pred.get("humidity")) / 2,
            "risk_score": round(blended_risk, 1),
            "level": get_risk_level(blended_risk).value,
            "ml_contribution": round(effective_ml_weight * ml_risk, 1),
            "rule_contribution": round(effective_rule_weight * rule_risk, 1),
            "ensemble_confidence": round(combined_confidence, 2),
            "ml_confidence": ml_conf,
            "use_ml": ml_conf >= config.confidence_threshold
        })

    return blended


def get_ensemble_forecast(
    temps: List[float],
    humidities: List[float],
    group_id: str = "elementary",
    activity_level: str = "moderate",
    config: Optional[EnsembleConfig] = None
) -> EnsembleResult:
    """Generate ensemble forecast combining ML and rule-based methods.

    Args:
        temps: Hourly temperatures
        humidities: Hourly humidities
        group_id: Vulnerability group
        activity_level: Activity intensity
        config: Optional ensemble configuration

    Returns:
        EnsembleResult with blended predictions
    """
    if config is None:
        config = EnsembleConfig()

    # Get ML predictions
    from src.core.ml_forecast import get_ml_forecast, get_fallback_risk_forecast

    try:
        ml_predictions, ml_metadata = get_ml_forecast(temps, humidities, group_id, activity_level)
    except Exception:
        ml_predictions = get_fallback_risk_forecast(temps, humidities, group_id, activity_level)
        ml_metadata = {"prediction_type": "rule_based_fallback"}

    # Get rule-based predictions
    group = VULNERABILITY_GROUPS.get(group_id, VULNERABILITY_GROUPS["general"])
    rule_predictions = get_hourly_forecast_risk(temps, humidities, group, activity_level)

    # Add uncertainty range to rule predictions
    for pred in rule_predictions:
        pred["uncertainty_range"] = {
            "low": pred["temperature"] - 2,
            "high": pred["temperature"] + 2
        }

    # Blend predictions
    blended = blend_predictions(ml_predictions, rule_predictions, config)

    # Calculate confidence scores
    confidence_scores = [p["ensemble_confidence"] for p in blended]

    return EnsembleResult(
        predictions=blended,
        blended_forecast=blended,
        confidence_scores=confidence_scores,
        weights_used={
            "ml_weight": config.ml_weight,
            "rule_weight": config.rule_weight,
            "confidence_threshold": config.confidence_threshold
        },
        metadata={
            "ml_model_available": ml_metadata.get("model_available", False),
            "ml_model_version": ml_metadata.get("model_version", "unknown"),
            "prediction_type": "ensemble",
            "total_hours": len(blended)
        }
    )


def find_optimal_time_slots(
    hourly_temps: List[float],
    hourly_humidity: List[float],
    group_id: str = "elementary",
    activity_level: str = "moderate",
    min_safe_score: float = 40.0,
    dangerous_threshold: float = 70.0
) -> Dict[str, List[TimeSlot]]:
    """Find optimal and dangerous time slots for activities.

    Args:
        hourly_temps: Hourly temperatures
        hourly_humidity: Hourly humidities
        group_id: Vulnerability group
        activity_level: Activity intensity
        min_safe_score: Max risk score for safe activities
        dangerous_threshold: Minimum risk score for dangerous classification

    Returns:
        Dict with "optimal" and "dangerous" time slot lists
    """
    from src.core.risk_engine import get_risk_level, get_risk_level_name

    group = VULNERABILITY_GROUPS.get(group_id, VULNERABILITY_GROUPS["general"])

    optimal_slots = []
    dangerous_slots = []

    for i, (temp, humidity) in enumerate(zip(hourly_temps, hourly_humidity)):
        risk = calculate_adaptive_risk(
            temp=temp,
            humidity=humidity,
            group=group,
            exposure_minutes=60,
            activity_level=activity_level
        )

        score = risk["total_score"]
        level = get_risk_level_name(score)

        # Determine reasons
        reasons = []
        if risk["details"]["nws_heat_index_c"] > 35:
            reasons.append("high_heat_index")
        if risk["details"]["wbgt"] > 30:
            reasons.append("high_wbgt")
        if risk["components"]["exposure_duration"] > 40:
            reasons.append("prolonged_exposure")
        if risk["components"]["percentile_anomaly"] > 40:
            reasons.append("anomalously_hot")

        is_dangerous = score >= dangerous_threshold

        slot = TimeSlot(
            hour=i,
            start_time=f"{i:02d}:00",
            end_time=f"{i+1:02d}:00" if i < 23 else "00:00",
            risk_score=score,
            level=level,
            temperature=round(temp, 1),
            humidity=round(humidity, 1),
            confidence=0.8,  # Rule-based confidence
            is_dangerous=is_dangerous,
            reason=", ".join(reasons) if reasons else "normal"
        )

        if score <= min_safe_score:
            optimal_slots.append(slot)
        if is_dangerous:
            dangerous_slots.append(slot)

    return {
        "optimal": optimal_slots,
        "dangerous": dangerous_slots
    }


def get_time_slots_recommendation(
    temps: List[float],
    humidities: List[float],
    group_id: str = "elementary",
    activity_level: str = "moderate"
) -> Dict[str, Any]:
    """Get comprehensive time slot recommendations.

    Returns:
        Dict with optimal slots, dangerous slots, and summary
    """
    slots = find_optimal_time_slots(temps, humidities, group_id, activity_level)

    # Convert TimeSlots to dicts for JSON serialization
    optimal = [
        {
            "hour": s.hour,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "risk_score": s.risk_score,
            "level": s.level,
            "temperature": s.temperature,
            "humidity": s.humidity,
            "confidence": s.confidence,
            "reason": s.reason
        }
        for s in slots["optimal"]
    ]

    dangerous = [
        {
            "hour": s.hour,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "risk_score": s.risk_score,
            "level": s.level,
            "temperature": s.temperature,
            "humidity": s.humidity,
            "confidence": s.confidence,
            "reason": s.reason
        }
        for s in slots["dangerous"]
    ]

    # Find best and worst hours
    all_hours = list(range(len(temps)))

    return {
        "optimal_slots": optimal,
        "dangerous_slots": dangerous,
        "summary": {
            "total_hours": len(temps),
            "safe_hours": len(optimal),
            "dangerous_hours": len(dangerous),
            "recommended_start_hour": optimal[0]["hour"] if optimal else None,
            "recommended_end_hour": optimal[-1]["hour"] + 1 if optimal else None,
            "avoid_start_hour": dangerous[0]["hour"] if dangerous else None,
            "avoid_end_hour": dangerous[-1]["hour"] + 1 if dangerous else None,
        }
    }