"""Risk score request/response schemas"""
from __future__ import annotations
from pydantic import BaseModel, Field
from app.core.risk_scoring import ActivityIntensity, RiskClass
from app.core.vulnerability import ProfileID


class RiskScoreRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60)
    humidity_rh: float = Field(..., ge=0, le=100)
    profile_id: ProfileID
    activity_intensity: ActivityIntensity
    duration_minutes: int = Field(..., ge=0, le=720)
    shade_available: bool = False
    water_access: bool = True
    time_of_day_hour: int = Field(13, ge=0, le=23)
    acclimatized: bool = False
    low_confidence: bool = False


class RiskScoreResponse(BaseModel):
    score: float
    risk_class: RiskClass
    dominant_factors: list[str]
    confidence: float
    heat_index_c: float
    profile_id: str
    conservative_applied: bool
