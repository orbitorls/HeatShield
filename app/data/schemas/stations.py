"""Station observation schemas"""
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class StationObservation(BaseModel):
    station_id: str
    ts_utc: datetime
    temp_c: float = Field(..., ge=-10, le=55)
    rh: float = Field(..., ge=0, le=100)
    wind_ms: float | None = None
    precip_mm: float | None = None
    source: Literal["tmd", "synthetic", "cache", "era5", "nasa_power", "modis"] = "tmd"
    solar_wm2: float | None = None
    cloud_cover: float | None = None
    blh_m: float | None = None
    pressure_hpa: float | None = None
    lst_c: float | None = None
