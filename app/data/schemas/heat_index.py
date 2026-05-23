"""Heat Index request/response schemas"""
from __future__ import annotations
from pydantic import BaseModel, Field


class HeatIndexRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60, description="อุณหภูมิ dry-bulb (°C)")
    humidity_rh: float = Field(..., ge=0, le=100, description="ความชื้นสัมพัทธ์ (%)")


class HeatIndexResponse(BaseModel):
    heat_index_c: float
    category: str
    temperature_c: float
    humidity_rh: float
