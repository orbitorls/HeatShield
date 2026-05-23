"""What-if simulation schemas"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from app.core.risk_scoring import ActivityIntensity
from app.core.vulnerability import ProfileID


class InterventionIn(BaseModel):
    intervention_type: str
    shift_hours: float = 0.0
    break_every_minutes: int = 20
    add_shade: bool = False
    add_water: bool = False
    reduce_duration_by: int = 0
    new_intensity: Optional[str] = None


class WhatIfRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60)
    humidity_rh: float = Field(..., ge=0, le=100)
    profile_id: ProfileID
    activity_intensity: ActivityIntensity
    duration_minutes: int = Field(..., ge=1, le=720)
    shade_available: bool = False
    water_access: bool = True
    time_of_day_hour: int = Field(13, ge=0, le=23)
    acclimatized: bool = False
    interventions: list[InterventionIn] = Field(..., min_length=1)


class ScenarioOut(BaseModel):
    intervention_type: str
    original_score: float
    original_class: str
    new_score: float
    new_class: str
    score_reduction: float
    effective: bool
    summary_th: str


class WhatIfResponse(BaseModel):
    original_risk_score: float
    original_risk_class: str
    scenarios: list[ScenarioOut]
    best_intervention: Optional[str] = None
    best_score_reduction: Optional[float] = None
