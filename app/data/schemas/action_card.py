"""Action card schemas"""
from __future__ import annotations
from pydantic import BaseModel, Field
from app.core.risk_scoring import ActivityIntensity
from app.core.vulnerability import ProfileID
from app.core.action_card import Audience


class ActionCardRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60)
    humidity_rh: float = Field(..., ge=0, le=100)
    profile_id: ProfileID
    activity_intensity: ActivityIntensity
    duration_minutes: int = Field(..., ge=0, le=720)
    shade_available: bool = False
    water_access: bool = True
    time_of_day_hour: int = Field(13, ge=0, le=23)
    acclimatized: bool = False
    audience: Audience = Audience.TEACHER
    forecast_horizon_h: int = Field(0, ge=0, le=72)


class ActionCardResponse(BaseModel):
    risk_class: str
    audience: str
    headline_th: str
    immediate_actions: list[str]
    monitoring_points: list[str]
    when_to_escalate: str
    valid_until_hint: str
    confidence_note: str
    dominant_factors: list[str]
