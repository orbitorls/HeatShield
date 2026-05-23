"""Forecast request/response schemas"""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
from app.data.schemas.stations import StationObservation


class ForecastPoint(BaseModel):
    station_id: str
    ts_utc: datetime
    horizon_h: int
    heat_index_c: float
    temp_c: float
    rh: float
    pi_lower: float | None = None
    pi_upper: float | None = None
    model_version: str


class ForecastRequest(BaseModel):
    station_id: str
    horizons: list[int] = Field(default=[6, 24, 48], description="Forecast horizons in hours")
    recent_obs: list[StationObservation] | None = Field(
        None,
        description="Recent observations — if None, loads from local parquet store"
    )


class ForecastResponse(BaseModel):
    station_id: str
    generated_at: datetime
    forecasts: list[ForecastPoint]
    model_version: str
    low_confidence: bool = False
    confidence_reason: str | None = None
