"""
Heat Index calculation using the Rothfusz regression equation (NOAA/NWS).
Valid for heat index >= 27°C and relative humidity >= 40%.
Below that range, uses Steadman's simpler formula.
"""
from dataclasses import dataclass
from enum import Enum


class HeatIndexCategory(str, Enum):
    CAUTION = "Caution"              # 27–32°C  — ความเหนื่อยล้าจากความร้อน
    EXTREME_CAUTION = "ExtremeC"     # 32–40°C  — ตะคริว/ความเหนื่อยล้า
    DANGER = "Danger"                # 40–54°C  — ตะคริว/เพลียแดดเป็นไปได้มาก
    EXTREME_DANGER = "ExtremeDanger" # >54°C    — ฮีทสโตรกใกล้จะเกิด


@dataclass(frozen=True)
class HeatIndexResult:
    heat_index: float          # °C
    category: HeatIndexCategory
    temp_c: float
    humidity_rh: float
    used_adjustment: bool      # True ถ้าใช้ low/high RH adjustment


# Rothfusz coefficients (inputs in °F, output in °F — แปลง C↔F ภายใน)
_C1 = -42.379
_C2 =  2.04901523
_C3 = 10.14333127
_C4 = -0.22475541
_C5 = -0.00683783
_C6 = -0.05481717
_C7 =  0.00122874
_C8 =  0.00085282
_C9 = -0.00000199


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def _f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def compute(temp_c: float, humidity_rh: float) -> HeatIndexResult:
    """
    คำนวณ heat index จากอุณหภูมิ (°C) และความชื้นสัมพัทธ์ (%).

    Args:
        temp_c: อุณหภูมิ dry-bulb เป็น °C
        humidity_rh: ความชื้นสัมพัทธ์ 0–100

    Returns:
        HeatIndexResult พร้อม heat_index (°C) และ category
    """
    if not (0 <= humidity_rh <= 100):
        raise ValueError(f"humidity_rh must be 0–100, got {humidity_rh}")

    t_f = _c_to_f(temp_c)
    rh = humidity_rh

    # สูตรง่าย (Steadman) สำหรับ temp < 27°C หรือ RH < 40
    if temp_c < 27 or rh < 40:
        simple_hi_f = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
        hi_c = _f_to_c(simple_hi_f)
        return HeatIndexResult(
            heat_index=round(hi_c, 1),
            category=_categorize(hi_c),
            temp_c=temp_c,
            humidity_rh=humidity_rh,
            used_adjustment=False,
        )

    # Rothfusz regression
    hi_f = (
        _C1
        + _C2 * t_f
        + _C3 * rh
        + _C4 * t_f * rh
        + _C5 * t_f**2
        + _C6 * rh**2
        + _C7 * t_f**2 * rh
        + _C8 * t_f * rh**2
        + _C9 * t_f**2 * rh**2
    )

    # Low-humidity adjustment (RH < 13% และ temp 27–112°F → ลด HI)
    adjustment_used = False
    if rh < 13 and 80 <= t_f <= 112:
        adj = ((13 - rh) / 4) * ((17 - abs(t_f - 95)) / 17) ** 0.5
        hi_f -= adj
        adjustment_used = True

    # High-humidity adjustment (RH > 85% และ temp 80–87°F → เพิ่ม HI)
    elif rh > 85 and 80 <= t_f <= 87:
        adj = ((rh - 85) / 10) * ((87 - t_f) / 5)
        hi_f += adj
        adjustment_used = True

    hi_c = _f_to_c(hi_f)
    return HeatIndexResult(
        heat_index=round(hi_c, 1),
        category=_categorize(hi_c),
        temp_c=temp_c,
        humidity_rh=humidity_rh,
        used_adjustment=adjustment_used,
    )


def _categorize(hi_c: float) -> HeatIndexCategory:
    if hi_c > 54:
        return HeatIndexCategory.EXTREME_DANGER
    if hi_c >= 40:
        return HeatIndexCategory.DANGER
    if hi_c >= 32:
        return HeatIndexCategory.EXTREME_CAUTION
    return HeatIndexCategory.CAUTION
