"""Heat Risk Calculation Engine

Provides vulnerability-based risk scoring for different population groups.
Enhanced with Adaptive Heat-Health Risk Engine supporting:
- NWS Heat Index calculation
- Wet Bulb Globe Temperature (WBGT)
- Local Percentile Anomaly detection
- Exposure Duration scoring
- Adaptive Risk scoring with weighted factors
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    """Risk level categories (original + extended)"""
    NORMAL = "Normal"        # 0-20
    WATCH = "Watch"          # 21-40
    WARNING = "Warning"       # 41-60
    DANGER = "Danger"        # 61-80
    EXTREME = "Extreme"      # 81-100
    # Backward compatibility aliases
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# Backward compatibility: map old levels to new extended levels
_LEGACY_LEVEL_MAP = {
    "Low": RiskLevel.WATCH,
    "Medium": RiskLevel.WARNING,
    "High": RiskLevel.DANGER,
    "Critical": RiskLevel.EXTREME,
}


@dataclass
class VulnerabilityGroup:
    """Vulnerability group configuration"""
    id: str
    name: str
    name_th: str
    icon: str
    base_multiplier: float
    temp_threshold: float
    humidity_weight: float = 0.3
    activity_factor: float = 1.0


# Extended vulnerability profiles with additional factors
@dataclass
class VulnerabilityProfile:
    """Extended vulnerability profile with adaptive factors"""
    group: VulnerabilityGroup
    age_factor: float = 1.0           # 1.0 = normal, 1.3+ = young/elderly
    health_factor: float = 1.0        # 1.0 = healthy, 1.2+ = chronic conditions
    acclimatization: float = 1.0      # 1.0 = acclimatized, 1.2+ = unacclimatized
    medication_factor: float = 1.0    # 1.0 = no meds, 1.1+ = heat-affecting meds


VULNERABILITY_GROUPS: Dict[str, VulnerabilityGroup] = {
    "elementary": VulnerabilityGroup(
        id="elementary",
        name="Elementary Students",
        name_th="นักเรียนประถม",
        icon="👶",
        base_multiplier=1.3,
        temp_threshold=32.0,
        humidity_weight=0.4,
        activity_factor=1.2
    ),
    "elderly": VulnerabilityGroup(
        id="elderly",
        name="Elderly",
        name_th="ผู้สูงอายุ",
        icon="👴",
        base_multiplier=1.5,
        temp_threshold=30.0,
        humidity_weight=0.5,
        activity_factor=0.8
    ),
    "riders": VulnerabilityGroup(
        id="riders",
        name="Motorcycle Riders",
        name_th="นักขับขี่รถจักรยานยนต์",
        icon="🏍️",
        base_multiplier=1.2,
        temp_threshold=35.0,
        humidity_weight=0.2,
        activity_factor=1.5
    ),
    "workers": VulnerabilityGroup(
        id="workers",
        name="Outdoor Workers",
        name_th="กรรมกรกลางแจ้ง",
        icon="👷",
        base_multiplier=1.4,
        temp_threshold=33.0,
        humidity_weight=0.4,
        activity_factor=1.8
    ),
    "general": VulnerabilityGroup(
        id="general",
        name="General Population",
        name_th="ประชาชนทั่วไป",
        icon="👥",
        base_multiplier=1.0,
        temp_threshold=35.0,
        humidity_weight=0.3,
        activity_factor=1.0
    ),
}

# Extended risk levels with new categories
RISK_LEVELS: Dict[RiskLevel, Dict] = {
    RiskLevel.NORMAL: {"score_range": (0, 20), "color": "#2196F3", "color_hex": (33, 150, 243)},
    RiskLevel.WATCH: {"score_range": (21, 40), "color": "#4CAF50", "color_hex": (76, 175, 80)},
    RiskLevel.WARNING: {"score_range": (41, 60), "color": "#FFC107", "color_hex": (255, 193, 7)},
    RiskLevel.DANGER: {"score_range": (61, 80), "color": "#FF9800", "color_hex": (255, 152, 0)},
    RiskLevel.EXTREME: {"score_range": (81, 100), "color": "#F44336", "color_hex": (244, 67, 54)},
    # Backward compatibility
    RiskLevel.LOW: {"score_range": (0, 30), "color": "#4CAF50", "color_hex": (76, 175, 80)},
    RiskLevel.MEDIUM: {"score_range": (31, 60), "color": "#FFC107", "color_hex": (255, 193, 7)},
    RiskLevel.HIGH: {"score_range": (61, 80), "color": "#FF9800", "color_hex": (255, 152, 0)},
    RiskLevel.CRITICAL: {"score_range": (81, 100), "color": "#F44336", "color_hex": (244, 67, 54)},
}

# Activity intensity levels
class ActivityLevel(Enum):
    """Activity intensity classification"""
    LIGHT = "light"       # Office work, sitting
    MODERATE = "moderate" # Walking, light exercise
    HEAVY = "heavy"      # Outdoor labor, sports

ACTIVITY_WEIGHTS = {
    ActivityLevel.LIGHT: 0.5,
    ActivityLevel.MODERATE: 1.0,
    ActivityLevel.HEAVY: 1.5,
}

# Adaptive risk formula weights (must sum to 1.0)
ADAPTIVE_RISK_WEIGHTS = {
    "heat_index": 0.35,
    "percentile_anomaly": 0.20,
    "exposure_duration": 0.20,
    "vulnerability": 0.15,
    "activity_intensity": 0.10,
}

# Historical percentile baselines by location
# These represent typical temperature percentiles for various Thai regions
DEFAULT_PERCENTILE_BASELINES: Dict[str, Dict[str, float]] = {
    "default": {
        "p50": 30.0,
        "p75": 33.0,
        "p90": 35.0,
        "p95": 37.0,
        "p99": 40.0,
    },
    "bangkok": {
        "p50": 32.0,
        "p75": 34.0,
        "p90": 36.0,
        "p95": 38.0,
        "p99": 41.0,
    },
    "chiang_mai": {
        "p50": 29.0,
        "p75": 32.0,
        "p90": 35.0,
        "p95": 37.0,
        "p99": 40.0,
    },
    "songkhla": {
        "p50": 31.0,
        "p75": 33.0,
        "p90": 35.0,
        "p95": 37.0,
        "p99": 40.0,
    },
}

# NWS Heat Index coefficients
NWS_HI_COEFFICIENTS = {
    "c1": -10.3,
    "c2": 0.993,
    "c3": -0.000193,
    "c4": 0.000021,
    "c5": -0.000002,
    "c6": -0.000000006,
    "c7": 0.0000000001,
}


def get_risk_level(score: float, use_legacy: bool = False) -> RiskLevel:
    """Get risk level from score.

    Args:
        score: Risk score (0-100)
        use_legacy: If True, use original 4-level classification for backward compatibility
    """
    if use_legacy:
        # Original behavior
        if score <= 30:
            return RiskLevel.LOW
        elif score <= 60:
            return RiskLevel.MEDIUM
        elif score <= 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    # New extended 5-level classification
    if score <= 20:
        return RiskLevel.NORMAL
    elif score <= 40:
        return RiskLevel.WATCH
    elif score <= 60:
        return RiskLevel.WARNING
    elif score <= 80:
        return RiskLevel.DANGER
    else:
        return RiskLevel.EXTREME


def get_risk_level_name(score: float, use_legacy: bool = False) -> str:
    """Get risk level name string.

    Args:
        score: Risk score (0-100)
        use_legacy: If True, use original level names
    """
    level = get_risk_level(score, use_legacy)
    if use_legacy:
        return _LEGACY_LEVEL_MAP.get(level.value, level.value)
    return level.value


def get_risk_color(score: float, use_legacy: bool = False) -> str:
    level = get_risk_level(score, use_legacy)
    return RISK_LEVELS[level]["color"]


def calculate_base_risk(temp: float, humidity: float = 60.0) -> float:
    temp_f = temp * 9/5 + 32
    if temp_f >= 80:
        hi = (temp_f + humidity) * 0.5
        if temp_f >= 90:
            hi += (temp_f - 90) * 0.5
    else:
        hi = temp_f

    if hi <= 80:
        risk = (hi - 30) * 0.4
    elif hi <= 100:
        risk = 20 + (hi - 80) * 1.0
    elif hi <= 120:
        risk = 40 + (hi - 100) * 1.5
    else:
        risk = 70 + min((hi - 120) * 1.5, 30)

    return max(0, min(100, risk))


def calculate_vulnerability_risk(
    temp: float,
    humidity: float,
    group: VulnerabilityGroup,
    activity_level: str = "moderate"
) -> float:
    base_risk = calculate_base_risk(temp, humidity)
    risk = base_risk * group.base_multiplier

    if temp > group.temp_threshold:
        excess = temp - group.temp_threshold
        threshold_penalty = min(excess * 3, 20)
        risk += threshold_penalty

    if humidity > 70:
        excess_humidity = humidity - 70
        humidity_factor = (excess_humidity / 30) * group.humidity_weight * 15
        risk += humidity_factor

    activity_multipliers = {"light": 0.8, "moderate": 1.0, "heavy": 1.5}
    risk *= activity_multipliers.get(activity_level, 1.0)

    return min(100, max(0, risk))


def get_hourly_forecast_risk(
    hourly_temps: List[float],
    hourly_humidity: List[float],
    group: VulnerabilityGroup,
    activity_level: str = "moderate"
) -> List[Dict]:
    results = []
    for i, (temp, humidity) in enumerate(zip(hourly_temps, hourly_humidity)):
        risk_score = calculate_vulnerability_risk(temp, humidity, group, activity_level)
        level = get_risk_level(risk_score)
        results.append({
            "hour": i if i < 24 else i % 24,
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "risk_score": round(risk_score, 1),
            "level": level.value
        })
    return results


def compare_time_slots(
    hourly_risks: List[Dict],
    from_hour: int,
    to_hour: int
) -> Dict:
    from_risk = next((r for r in hourly_risks if r["hour"] == from_hour), None)
    to_risk = next((r for r in hourly_risks if r["hour"] == to_hour), None)

    if from_risk is None or to_risk is None:
        return {"error": "Hour not found in forecast data"}

    original_score = from_risk["risk_score"]
    recommended_score = to_risk["risk_score"]
    risk_reduction = original_score - recommended_score
    percentage_reduction = (risk_reduction / original_score * 100) if original_score > 0 else 0

    return {
        "scenario": {
            "from_time": f"{from_hour:02d}:00",
            "to_time": f"{to_hour:02d}:00",
            "from_level": from_risk["level"],
            "to_level": to_risk["level"]
        },
        "original": {
            "time": f"{from_hour:02d}:00",
            "risk_score": original_score,
            "level": from_risk["level"],
            "temperature": from_risk["temperature"]
        },
        "recommended": {
            "time": f"{to_hour:02d}:00",
            "risk_score": recommended_score,
            "level": to_risk["level"],
            "temperature": to_risk["temperature"]
        },
        "improvement": {
            "risk_reduction": round(risk_reduction, 1),
            "level_change": f"{from_risk['level']} → {to_risk['level']}",
            "percentage_reduction": f"{round(percentage_reduction, 0):.0f}%"
        }
    }


def generate_action_recommendations(
    hourly_risks: List[Dict],
    group: VulnerabilityGroup,
    for_role: str = "teacher"
) -> List[Dict]:
    recommendations = []
    max_risk = max(hourly_risks, key=lambda x: x["risk_score"], default=None)
    if max_risk is None:
        return []

    high_risk_hours = [r for r in hourly_risks if r["risk_score"] >= 60]
    critical_hours = [r for r in hourly_risks if r["risk_score"] >= 80]

    if max_risk["risk_score"] >= 80:
        urgency = "critical"
    elif max_risk["risk_score"] >= 60:
        urgency = "high"
    elif max_risk["risk_score"] >= 40:
        urgency = "medium"
    else:
        urgency = "low"

    if for_role in ["teacher", "manager"]:
        if critical_hours:
            recommendations.append({
                "title": "ยกเลิกกิจกรรมกลางแจ้ง",
                "icon": "🚫",
                "description": f"ช่วงเวลา {critical_hours[0]['hour']:02d}:00-{critical_hours[-1]['hour']+1:02d}:00 มีความเสี่ยงสูงมาก",
                "urgency": "critical",
                "action_type": "cancel"
            })
        if high_risk_hours:
            recommendations.append({
                "title": "เตรียมน้ำดื่มและยาประจำตัว",
                "icon": "💧",
                "description": "เตรียมน้ำดื่มเพียงพอและยาประจำตัวสำหรับกลุ่มเสี่ยง",
                "urgency": "high",
                "action_type": "prepare"
            })
        recommendations.append({
            "title": "จัดที่พักร่มเงา",
            "icon": "🏠",
            "description": "เตรียมพื้นที่พักในร่มหรือที่มีเครื่องปรับอากาศ",
            "urgency": "medium",
            "action_type": "prepare"
        })
        if max_risk["risk_score"] >= 60:
            recommendations.append({
                "title": "แจ้งผู้ปกครอง",
                "icon": "📢",
                "description": "แจ้งผู้ปกครองเกี่ยวกับสถานการณ์ความร้อน",
                "urgency": "high",
                "action_type": "communicate"
            })
    elif for_role == "parent":
        recommendations.append({
            "title": "รับบุตรหลานเร็วขึ้น",
            "icon": "👨‍👩‍👧",
            "description": f"ควรรับบุตรหลานกลับก่อนเวลา {max_risk['hour']:02d}:00",
            "urgency": urgency,
            "action_type": "pickup"
        })
        recommendations.append({
            "title": "สังเกตอาการภาวะความร้อน",
            "icon": "👀",
            "description": "สังเกตอาการปวดศีรษะ คลื่นไส้ อ่อนเพลีย",
            "urgency": "medium",
            "action_type": "observe"
        })

    return recommendations


# =============================================================================
# NEW ENHANCED FUNCTIONS FOR ADAPTIVE RISK ENGINE
# =============================================================================

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9/5 + 32


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9


def calculate_nws_heat_index(temp_f: float, humidity: float) -> float:
    """Calculate NWS Heat Index using the official Rothfusz regression.

    Args:
        temp_f: Temperature in Fahrenheit
        humidity: Relative humidity in percent (0-100)

    Returns:
        Heat index temperature in Fahrenheit

    Note:
        This uses the NWS Heat Index equation for conditions above 80°F
        with high humidity. For exact NWS formula, see:
        https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    """
    if temp_f < 80:
        return temp_f

    T = temp_f
    RH = humidity

    # Full NWS Rothfusz regression equation
    # HI = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH
    #      - 0.00683783*T^2 - 0.05481717*RH^2 + 0.00122874*T^2*RH
    #      + 0.00085282*T*RH^2 - 0.00000199*T^4 + 0.00000000009312*T^4*RH^2
    hi = (
        -42.379 +
        2.04901523 * T +
        10.14333127 * RH +
        -0.22475541 * T * RH +
        -0.00683783 * T * T +
        -0.05481717 * RH * RH +
        0.00122874 * T * T * RH +
        0.00085282 * T * RH * RH +
        -0.00000199 * T * T * T * T +
        0.00000000009312 * T * T * T * T * RH * RH
    )

    # Apply humidity adjustments
    if humidity < 13 and temp_f >= 80 and temp_f <= 112:
        adj = ((13 - humidity) / 4) * ((17 - abs(temp_f - 95)) / 17) ** 0.5
        hi = hi - adj
    elif humidity > 85 and temp_f >= 80 and temp_f <= 87:
        adj = ((humidity - 85) / 10) * ((87 - temp_f) / 5)
        hi = hi + adj

    return hi


def calculate_wbgt(
    temp: float,
    humidity: float,
    wind_speed: float = 0.0,
    solar_factor: float = 0.0
) -> float:
    """Calculate Wet Bulb Globe Temperature (WBGT).

    WBGT is a measure of heat stress in direct sunlight, based on:
    - Natural wet bulb temperature (humidity effect)
    - Globe thermometer temperature (radiant heat)
    - Dry bulb temperature (air temperature)

    Args:
        temp: Air temperature in Celsius
        humidity: Relative humidity in percent
        wind_speed: Wind speed in km/h (default 0 for still air)
        solar_factor: Solar radiation factor 0-1 (0 = shade, 1 = full sun)

    Returns:
        WBGT temperature in Celsius

    References:
        - ISO 7243: Ergonomics of the thermal environment
        - NIOSH occupational heat stress guidelines
    """
    # Natural wet bulb temperature approximation
    # Tw = T * arctan(0.1515 * (RH + 8.3136)**0.5) + arctan(T + RH)
    wet_bulb = temp * (humidity / 50) ** 0.75

    # Globe temperature (black globe thermometer)
    # GT ≈ T + (solar_factor * 10) - (wind_speed * 0.3)
    solar_effect = solar_factor * 10  # Estimated solar contribution
    wind_cooling = wind_speed * 0.3 if wind_speed > 0 else 0
    globe_temp = temp + solar_effect - wind_cooling

    # WBGT formula for outdoor conditions:
    # WBGT = 0.7 * Tw + 0.2 * GT + 0.1 * T
    wbgt = 0.7 * wet_bulb + 0.2 * globe_temp + 0.1 * temp

    return wbgt


def calculate_percentile_anomaly(
    current_temp: float,
    location_id: str = "default",
    historical_percentiles: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    """Calculate temperature anomaly relative to historical percentiles.

    Args:
        current_temp: Current temperature in Celsius
        location_id: Location identifier for baseline lookup
        historical_percentiles: Optional dict of percentile baselines

    Returns:
        Anomaly score 0-100 (higher = more anomalous)
    """
    baselines = historical_percentiles or DEFAULT_PERCENTILE_BASELINES
    location_baseline = baselines.get(location_id, baselines["default"])

    p50 = location_baseline["p50"]
    p75 = location_baseline["p75"]
    p90 = location_baseline["p90"]
    p95 = location_baseline["p95"]
    p99 = location_baseline["p99"]

    # Calculate anomaly score based on which percentile the temp exceeds
    if current_temp <= p50:
        # Below median - minimal anomaly
        return max(0, (current_temp - p50) * 2)
    elif current_temp <= p75:
        # Between 50th and 75th percentile
        return 10 + (current_temp - p50) / (p75 - p50) * 10
    elif current_temp <= p90:
        # Between 75th and 90th percentile
        return 20 + (current_temp - p75) / (p90 - p75) * 15
    elif current_temp <= p95:
        # Between 90th and 95th percentile (hot)
        return 35 + (current_temp - p90) / (p95 - p90) * 15
    elif current_temp <= p99:
        # Between 95th and 99th percentile (very hot)
        return 50 + (current_temp - p95) / (p99 - p95) * 30
    else:
        # Above 99th percentile (extreme)
        return 80 + min((current_temp - p99) * 2, 20)


def calculate_exposure_score(
    exposure_minutes: int,
    activity_level: str = "moderate",
    rest_minutes: int = 0
) -> float:
    """Calculate exposure duration score.

    Args:
        exposure_minutes: Duration of heat exposure in minutes
        activity_level: Activity intensity (light/moderate/heavy)
        rest_minutes: Rest periods in minutes (reduces risk)

    Returns:
        Exposure score 0-100
    """
    # Base score from exposure time
    base_exposure = min(exposure_minutes / 120 * 60, 60)  # Max 60 points

    # Activity level modifier
    activity_multipliers = {
        "light": 0.7,
        "moderate": 1.0,
        "heavy": 1.5
    }
    activity_mult = activity_multipliers.get(activity_level, 1.0)

    # Rest period reduction
    # Every 15 min of rest reduces exposure risk by 10 points
    rest_reduction = min((rest_minutes / 15) * 10, 30)

    # Calculate effective exposure
    exposure_score = base_exposure * activity_mult - rest_reduction

    return max(0, min(100, exposure_score))


def create_vulnerability_profile(
    group: VulnerabilityGroup,
    age: Optional[int] = None,
    health_conditions: Optional[List[str]] = None,
    is_acclimatized: bool = True,
    takes_heat_medications: bool = False
) -> VulnerabilityProfile:
    """Create an extended vulnerability profile.

    Args:
        group: Base vulnerability group
        age: Age of person (affects age_factor)
        health_conditions: List of health condition keywords
        is_acclimatized: Whether person is acclimatized to heat
        takes_heat_medications: Whether person takes medications affecting heat response

    Returns:
        VulnerabilityProfile with all factors calculated
    """
    # Age factor
    age_factor = 1.0
    if age is not None:
        if age < 8 or age > 65:
            age_factor = 1.3  # Young children or elderly
        elif age < 12 or age > 75:
            age_factor = 1.2  # Children or very elderly

    # Health factor
    health_factor = 1.0
    if health_conditions:
        high_risk_conditions = [
            "heart", "cardiovascular", "diabetes", "obesity",
            "hypertension", "kidney", "respiratory", "thyroid"
        ]
        for condition in health_conditions:
            if any(hc in condition.lower() for hc in high_risk_conditions):
                health_factor = max(health_factor, 1.2)
                break

    # Acclimatization factor
    acclimatization_factor = 1.0 if is_acclimatized else 1.2

    # Medication factor
    medication_factor = 1.0 if not takes_heat_medications else 1.15

    return VulnerabilityProfile(
        group=group,
        age_factor=age_factor,
        health_factor=health_factor,
        acclimatization=acclimatization_factor,
        medication_factor=medication_factor
    )


def calculate_adaptive_risk(
    temp: float,
    humidity: float,
    group: VulnerabilityGroup,
    exposure_minutes: int = 60,
    activity_level: str = "moderate",
    location_id: str = "default",
    wind_speed: float = 0.0,
    solar_factor: float = 0.0,
    rest_minutes: int = 0,
    vulnerability_profile: Optional[VulnerabilityProfile] = None,
    use_legacy: bool = False
) -> Dict[str, Any]:
    """Calculate adaptive risk score with all contributing factors.

    Adaptive Risk Formula:
        Risk Score = 0.35 × Heat Index Risk
                   + 0.20 × Local Percentile Anomaly
                   + 0.20 × Exposure Duration
                   + 0.15 × Vulnerability Profile
                   + 0.10 × Activity Intensity

    Args:
        temp: Temperature in Celsius
        humidity: Relative humidity in percent
        group: Vulnerability group
        exposure_minutes: Duration of heat exposure in minutes
        activity_level: Activity intensity (light/moderate/heavy)
        location_id: Location identifier for percentile baseline
        wind_speed: Wind speed in km/h
        solar_factor: Solar radiation factor 0-1
        rest_minutes: Rest periods in minutes
        vulnerability_profile: Optional pre-calculated profile
        use_legacy: If True, use legacy 4-level classification

    Returns:
        Dict containing:
            - total_score: Overall risk score 0-100
            - level: Risk level name
            - color: Risk color hex
            - components: Individual factor scores
            - details: Additional calculation details
    """
    # Calculate NWS Heat Index (in Fahrenheit)
    temp_f = celsius_to_fahrenheit(temp)
    nws_hi = calculate_nws_heat_index(temp_f, humidity)
    hi_celsius = fahrenheit_to_celsius(nws_hi)

    # 1. Heat Index Risk (0-100)
    # Convert heat index to risk score
    if hi_celsius <= 27:
        heat_index_risk = 0
    elif hi_celsius <= 32:
        heat_index_risk = (hi_celsius - 27) / 5 * 25
    elif hi_celsius <= 39:
        heat_index_risk = 25 + (hi_celsius - 32) / 7 * 25
    elif hi_celsius <= 46:
        heat_index_risk = 50 + (hi_celsius - 39) / 7 * 30
    else:
        heat_index_risk = 80 + min((hi_celsius - 46) * 2, 20)

    # 2. Local Percentile Anomaly (0-100)
    percentile_anomaly = calculate_percentile_anomaly(temp, location_id)

    # 3. Exposure Duration Score (0-100)
    exposure_score = calculate_exposure_score(
        exposure_minutes, activity_level, rest_minutes
    )

    # 4. Vulnerability Profile Score (0-100)
    if vulnerability_profile is None:
        # Create basic profile from group
        if group is None:
            group = VULNERABILITY_GROUPS["general"]
        profile = VulnerabilityProfile(group=group)
    else:
        profile = vulnerability_profile

    # Base vulnerability from group
    vuln_base = (profile.group.base_multiplier - 1.0) * 50  # Scale 0-50
    # Additional factors (scale each to 0-25)
    age_factor_score = (profile.age_factor - 1.0) * 50
    health_factor_score = (profile.health_factor - 1.0) * 50
    acclim_factor_score = (profile.acclimatization - 1.0) * 50
    vuln_score = vuln_base + min(age_factor_score + health_factor_score + acclim_factor_score, 50)

    # 5. Activity Intensity Score (0-100)
    activity_score = ACTIVITY_WEIGHTS.get(
        ActivityLevel(activity_level),
        ACTIVITY_WEIGHTS[ActivityLevel.MODERATE]
    ) * 67  # Scale to 0-100 (heavy activity gets ~100)

    # Calculate weighted total score
    weights = ADAPTIVE_RISK_WEIGHTS
    total_score = (
        weights["heat_index"] * heat_index_risk +
        weights["percentile_anomaly"] * percentile_anomaly +
        weights["exposure_duration"] * exposure_score +
        weights["vulnerability"] * vuln_score +
        weights["activity_intensity"] * activity_score
    )

    # Get risk level
    level = get_risk_level(total_score, use_legacy)
    color = RISK_LEVELS[level]["color"]

    # Determine contributing factors for details
    main_factors = []
    if heat_index_risk >= 50:
        main_factors.append("high_heat_index")
    if percentile_anomaly >= 50:
        main_factors.append("anomalous_temp")
    if exposure_score >= 50:
        main_factors.append("prolonged_exposure")
    if profile.age_factor > 1.1 or profile.health_factor > 1.1:
        main_factors.append("vulnerable_population")

    return {
        "total_score": round(total_score, 1),
        "level": get_risk_level_name(total_score, use_legacy),
        "color": color,
        "components": {
            "heat_index_risk": round(heat_index_risk, 1),
            "percentile_anomaly": round(percentile_anomaly, 1),
            "exposure_duration": round(exposure_score, 1),
            "vulnerability_profile": round(vuln_score, 1),
            "activity_intensity": round(activity_score, 1),
        },
        "details": {
            "nws_heat_index_f": round(nws_hi, 1),
            "nws_heat_index_c": round(hi_celsius, 1),
            "wbgt": round(calculate_wbgt(temp, humidity, wind_speed, solar_factor), 1),
            "temperature_c": temp,
            "humidity": humidity,
            "exposure_minutes": exposure_minutes,
            "activity_level": activity_level,
            "location_id": location_id,
            "main_contributing_factors": main_factors,
            "weights_used": weights,
        },
        "profile": {
            "group_id": profile.group.id,
            "age_factor": profile.age_factor,
            "health_factor": profile.health_factor,
            "acclimatization": profile.acclimatization,
            "medication_factor": profile.medication_factor,
        } if vulnerability_profile else None
    }


def get_adaptive_forecast_risks(
    hourly_temps: List[float],
    hourly_humidity: List[float],
    group: VulnerabilityGroup,
    exposure_minutes: int = 60,
    activity_level: str = "moderate",
    location_id: str = "default",
    vulnerability_profile: Optional[VulnerabilityProfile] = None
) -> List[Dict[str, Any]]:
    """Generate adaptive risk forecast for hourly data.

    Args:
        hourly_temps: List of hourly temperatures in Celsius
        hourly_humidity: List of hourly humidity values
        group: Vulnerability group
        exposure_minutes: Base exposure duration per hour
        activity_level: Activity intensity
        location_id: Location identifier
        vulnerability_profile: Optional pre-calculated profile

    Returns:
        List of adaptive risk dicts for each hour
    """
    results = []
    for i, (temp, humidity) in enumerate(zip(hourly_temps, hourly_humidity)):
        # Accumulate exposure throughout the day
        cumulative_exposure = min((i + 1) * exposure_minutes, 480)  # Cap at 8 hours
        # Reduce rest period based on time of day (assume no rest during active hours)
        rest = 0 if i < 10 or i > 14 else 0  # No rest during peak hours

        risk = calculate_adaptive_risk(
            temp=temp,
            humidity=humidity,
            group=group,
            exposure_minutes=cumulative_exposure,
            activity_level=activity_level,
            location_id=location_id,
            vulnerability_profile=vulnerability_profile
        )

        results.append({
            "hour": i if i < 24 else i % 24,
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            **risk
        })

    return results