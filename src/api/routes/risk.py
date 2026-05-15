"""Heat Risk API Routes

Provides endpoints for hourly risk forecasting, what-if analysis,
and action card generation.

Optional TMD integration for real-time weather data.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import yaml
from pathlib import Path

from src.core.risk_engine import (
    VULNERABILITY_GROUPS,
    RiskLevel,
    get_risk_level,
    get_risk_color,
    get_hourly_forecast_risk,
    compare_time_slots,
    generate_action_recommendations
)

# TMD integration (lazy loaded)
_tmd_service = None


async def get_tmd_service():
    """Get TMD service instance lazily."""
    global _tmd_service
    if _tmd_service is None:
        try:
            from src.services.tmd_service import get_tmd_service as _get_tmd
            _tmd_service = await _get_tmd()
        except ImportError:
            return None
    return _tmd_service

router = APIRouter()


def load_locations():
    locations_path = Path(__file__).parent.parent.parent / "data" / "locations.yaml"
    if locations_path.exists():
        with open(locations_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('locations', [])
    return []


def generate_sample_forecast(date_str: str = None) -> List[Dict]:
    base_temps = [
        28, 28, 27, 26, 26, 27,
        29, 31, 33, 35, 37, 38,
        39, 40, 39, 38, 36, 35,
        34, 33, 32, 31, 30, 29
    ]
    base_humidity = [
        75, 78, 80, 82, 85, 84,
        80, 70, 60, 55, 50, 48,
        45, 42, 40, 45, 50, 55,
        60, 65, 70, 72, 74, 75
    ]
    return [
        {"hour": h, "temperature": temp, "humidity": hum}
        for h, (temp, hum) in enumerate(zip(base_temps, base_humidity))
    ]


async def get_forecast_data(lat: float, lon: float, date_str: str = None, use_tmd: bool = True) -> List[Dict]:
    """Get forecast data, preferring TMD API with fallback to sample data.

    Args:
        lat: Latitude
        lon: Longitude
        date_str: Date string (unused, for API compatibility)
        use_tmd: Whether to try TMD API first

    Returns:
        List of hourly forecast dictionaries with temperature and humidity
    """
    if use_tmd:
        try:
            from src.services.tmd_service import (
                get_hourly_weather_for_location,
                get_forecast_for_location,
                _generate_hourly_from_daily,
                FALLBACK_HOURLY_FORECAST
            )

            # Try to get hourly weather from TMD
            hourly = await get_hourly_weather_for_location(lat, lon, use_fallback=True)
            if hourly and len(hourly) >= 24:
                # Normalize to just temperature and humidity
                return [
                    {"hour": h, "temperature": item.get("temperature", 30), "humidity": item.get("humidity", 60)}
                    for h, item in enumerate(hourly[:24])
                ]

            # Try daily forecast and expand to hourly
            daily = await get_forecast_for_location(lat, lon, days=1, use_fallback=True)
            if daily:
                return _generate_hourly_from_daily(daily[0], lat, lon)

        except Exception as e:
            # Log but continue to fallback
            import logging
            logging.getLogger(__name__).warning(f"TMD unavailable, using sample data: {e}")

    # Return sample forecast as fallback
    return generate_sample_forecast(date_str)


class LocationResponse(BaseModel):
    id: str
    name: str
    name_en: str
    lat: float
    lon: float
    type: str


class HourlyRiskResponse(BaseModel):
    location: Dict
    date: str
    hourly_risks: List[Dict]
    group: Dict


class WhatIfResponse(BaseModel):
    scenario: Dict
    original: Dict
    recommended: Dict
    improvement: Dict


@router.get("/locations", response_model=List[LocationResponse])
async def get_locations():
    locations = load_locations()
    if not locations:
        return [
            {"id": "school_a", "name": "โรงเรียน A", "name_en": "School A",
             "lat": 13.7563, "lon": 100.5018, "type": "school"},
        ]
    return locations


@router.get("/groups")
async def get_vulnerability_groups():
    return [
        {
            "id": group.id,
            "name": group.name,
            "name_th": group.name_th,
            "icon": group.icon,
            "multiplier": group.base_multiplier,
            "temp_threshold": group.temp_threshold
        }
        for group in VULNERABILITY_GROUPS.values()
    ]


@router.get("/hourly", response_model=HourlyRiskResponse)
async def get_hourly_risk(
    location_id: str = Query(...),
    date_str: Optional[str] = Query(None),
    group_id: str = Query("elementary"),
    use_tmd: bool = Query(True, description="Whether to use TMD API for real data")
):
    locations = load_locations()
    location = next((loc for loc in locations if loc['id'] == location_id), None)

    if not location:
        if location_id == "school_a":
            location = {"id": "school_a", "name": "โรงเรียน A", "lat": 13.7563, "lon": 100.5018}
        else:
            raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found")

    if group_id not in VULNERABILITY_GROUPS:
        raise HTTPException(status_code=400, detail=f"Invalid group '{group_id}'")
    group = VULNERABILITY_GROUPS[group_id]

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Get forecast data (from TMD or fallback)
    forecast = await get_forecast_data(
        lat=location.get('lat', 13.7563),
        lon=location.get('lon', 100.5018),
        date_str=date_str,
        use_tmd=use_tmd
    )
    temps = [f["temperature"] for f in forecast]
    humidities = [f["humidity"] for f in forecast]

    hourly_risks = get_hourly_forecast_risk(temps, humidities, group)

    return HourlyRiskResponse(
        location={"id": location["id"], "name": location["name"], "lat": location["lat"], "lon": location["lon"]},
        date=date_str,
        hourly_risks=hourly_risks,
        group={
            "id": group.id,
            "name": group.name,
            "name_th": group.name_th,
            "icon": group.icon,
            "multiplier": group.base_multiplier
        }
    )


@router.get("/whatif", response_model=WhatIfResponse)
async def whatif_analysis(
    from_hour: int = Query(..., ge=0, le=23),
    to_hour: int = Query(..., ge=0, le=23),
    group_id: str = Query("elementary"),
    location_id: str = Query("school_a"),
    date_str: Optional[str] = Query(None),
    use_tmd: bool = Query(True, description="Whether to use TMD API for real data")
):
    if from_hour == to_hour:
        raise HTTPException(status_code=400, detail="from_hour and to_hour must be different")

    if group_id not in VULNERABILITY_GROUPS:
        raise HTTPException(status_code=400, detail=f"Invalid group '{group_id}'")

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Get location data
    locations = load_locations()
    location = next((loc for loc in locations if loc['id'] == location_id), None)
    if not location:
        if location_id == "school_a":
            location = {"id": "school_a", "name": "โรงเรียน A", "lat": 13.7563, "lon": 100.5018}
        else:
            location = {"id": location_id, "name": location_id, "lat": 13.7563, "lon": 100.5018}

    # Get forecast data
    forecast = await get_forecast_data(
        lat=location.get('lat', 13.7563),
        lon=location.get('lon', 100.5018),
        date_str=date_str,
        use_tmd=use_tmd
    )
    temps = [f["temperature"] for f in forecast]
    humidities = [f["humidity"] for f in forecast]
    group = VULNERABILITY_GROUPS[group_id]

    hourly_risks = get_hourly_forecast_risk(temps, humidities, group)
    result = compare_time_slots(hourly_risks, from_hour, to_hour)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return WhatIfResponse(**result)


@router.get("/action-card")
async def get_action_card(
    location_id: str = Query("school_a"),
    group_id: str = Query("elementary"),
    for_role: str = Query("teacher"),
    date_str: Optional[str] = Query(None),
    use_tmd: bool = Query(True, description="Whether to use TMD API for real data")
):
    if group_id not in VULNERABILITY_GROUPS:
        raise HTTPException(status_code=400, detail=f"Invalid group '{group_id}'")
    if for_role not in ["teacher", "manager", "parent"]:
        raise HTTPException(status_code=400, detail="for_role must be 'teacher', 'manager', or 'parent'")

    group = VULNERABILITY_GROUPS[group_id]
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Get location data
    locations = load_locations()
    location = next((loc for loc in locations if loc['id'] == location_id), None)
    if not location:
        if location_id == "school_a":
            location = {"id": "school_a", "name": "โรงเรียน A", "lat": 13.7563, "lon": 100.5018}
        else:
            location = {"id": location_id, "name": location_id, "lat": 13.7563, "lon": 100.5018}

    # Get forecast data
    forecast = await get_forecast_data(
        lat=location.get('lat', 13.7563),
        lon=location.get('lon', 100.5018),
        date_str=date_str,
        use_tmd=use_tmd
    )
    temps = [f["temperature"] for f in forecast]
    humidities = [f["humidity"] for f in forecast]

    hourly_risks = get_hourly_forecast_risk(temps, humidities, group)
    recommendations = generate_action_recommendations(hourly_risks, group, for_role)

    max_risk = max(hourly_risks, key=lambda x: x["risk_score"], default={"risk_score": 0, "hour": 0})
    critical_hours = [r for r in hourly_risks if r["risk_score"] >= 80]

    time_range = f"{critical_hours[0]['hour']:02d}:00 - {critical_hours[-1]['hour']+1:02d}:00" if critical_hours else f"{max_risk['hour']:02d}:00 - {max_risk['hour']+1:02d}:00"

    urgency = "low"
    if max_risk["risk_score"] >= 80:
        urgency = "critical"
    elif max_risk["risk_score"] >= 60:
        urgency = "high"
    elif max_risk["risk_score"] >= 40:
        urgency = "medium"

    return {
        "title": f"คำแนะนำสำหรับ{group.name_th}",
        "icon": group.icon,
        "urgency": urgency,
        "time_range": time_range,
        "temperature_range": f"{min(temps):.0f}°C - {max(temps):.0f}°C",
        "recommendations": recommendations
    }


@router.get("/risk-summary")
async def get_risk_summary(
    location_id: str = Query("school_a"),
    date_str: Optional[str] = Query(None),
    use_tmd: bool = Query(True, description="Whether to use TMD API for real data")
):
    # Get location data
    locations = load_locations()
    location = next((loc for loc in locations if loc['id'] == location_id), None)
    if not location:
        if location_id == "school_a":
            location = {"id": "school_a", "name": "โรงเรียน A", "lat": 13.7563, "lon": 100.5018}
        else:
            location = {"id": location_id, "name": location_id, "lat": 13.7563, "lon": 100.5018}

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Get forecast data
    forecast = await get_forecast_data(
        lat=location.get('lat', 13.7563),
        lon=location.get('lon', 100.5018),
        date_str=date_str,
        use_tmd=use_tmd
    )
    temps = [f["temperature"] for f in forecast]
    humidities = [f["humidity"] for f in forecast]

    summary = {
        "date": date_str,
        "location_id": location_id,
        "data_source": "tmd" if use_tmd else "sample",
        "groups": []
    }

    for group in VULNERABILITY_GROUPS.values():
        hourly_risks = get_hourly_forecast_risk(temps, humidities, group)
        max_risk = max(hourly_risks, key=lambda x: x["risk_score"], default={"risk_score": 0, "hour": 0})

        summary["groups"].append({
            "id": group.id,
            "name": group.name,
            "icon": group.icon,
            "max_risk": max_risk["risk_score"],
            "max_risk_hour": max_risk["hour"],
            "risk_level": get_risk_level(max_risk["risk_score"]).value,
            "risk_color": get_risk_color(max_risk["risk_score"])
        })

    return summary


@router.get("/weather-status")
async def get_weather_status():
    """Get TMD API status and availability."""
    try:
        from src.services.tmd_service import check_tmd_availability
        status = await check_tmd_availability()
        return {
            "tmd_available": status["available"],
            "latency_ms": status["latency_ms"],
            "stations_count": status["stations_count"],
            "using_real_data": status["available"]
        }
    except Exception as e:
        return {
            "tmd_available": False,
            "error": str(e),
            "using_real_data": False
        }