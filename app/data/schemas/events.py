"""Heatwave event detection schemas"""
from __future__ import annotations
from pydantic import BaseModel, Field


class DailyObsIn(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    max_value: float
    min_value: float


class HeatwaveDetectRequest(BaseModel):
    station_id: str
    observations: list[DailyObsIn] = Field(..., min_length=2)
    historical_values: list[float] = Field(
        ..., min_length=30, description="ค่าสูงสุดรายวันย้อนหลังสำหรับสร้าง baseline"
    )
    percentile: float = Field(90.0, ge=50, le=99)
    min_consecutive_days: int = Field(2, ge=1, le=7)
    require_warm_nights: bool = False
    metric: str = "heat_index"


class HeatwaveDetectResponse(BaseModel):
    is_heatwave: bool
    event_type: str
    percentile_threshold: float
    consecutive_days: int
    consecutive_nights: int
    trigger_reason: list[str]
    local_baseline: dict
