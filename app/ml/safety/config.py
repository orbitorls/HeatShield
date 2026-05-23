"""Per-station, per-horizon safety configuration.

Generated from analysis of 57 model registry evaluations.
Stations not listed get default config (no blend, no boost).
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_SUPPORTED_HORIZON = 24


@dataclass
class StationHorizonConfig:
    blend_weight: float = 1.0       # model weight in model+persistence blend (0.0-1.0)
    danger_boost: float = 0.0       # boost mean toward upper PI when HI >= 38C (0=off)
    enabled: bool = True            # whether to serve this horizon


STATION_SAFETY_CONFIG: dict[str, dict[int, StationHorizonConfig]] = {
    # Bangkok — urban heat island, persistence/climatology competitive at h6/h24
    "BKK_01": {
        6:  StationHorizonConfig(blend_weight=0.6,  danger_boost=0.0),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        24: StationHorizonConfig(blend_weight=0.6,  danger_boost=0.0),
        48: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.0, enabled=False),
        72: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.0, enabled=False),
    },
    # Rayong — worst danger recall in the fleet (22.6% at h6, 6.5% at h72)
    "RYG_01": {
        6:  StationHorizonConfig(blend_weight=1.0,  danger_boost=0.4),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.3),
        24: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.4),
        48: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.4, enabled=False),
        72: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.5, enabled=False),
    },
    # Nong Khai — danger recall 39.7% at h6, moderate across horizons
    "HYI_01": {
        6:  StationHorizonConfig(blend_weight=1.0,  danger_boost=0.3),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.2),
        24: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.2),
        48: StationHorizonConfig(blend_weight=0.5,  danger_boost=0.2, enabled=False),
        72: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.3, enabled=False),
    },
    # Lampang — danger recall 41.2% at h6, 29.4% at h24
    "LPT_01": {
        6:  StationHorizonConfig(blend_weight=1.0,  danger_boost=0.3),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.2),
        24: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.4),
    },
    # Khon Kaen — danger recall drops at longer horizons
    "KKN_01": {
        6:  StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        24: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        48: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.3, enabled=False),
        72: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.3, enabled=False),
    },
    # Chiang Mai — negative skill at long horizons
    "CNX_01": {
        6:  StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        24: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        48: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.0, enabled=False),
        72: StationHorizonConfig(blend_weight=0.3,  danger_boost=0.0, enabled=False),
    },
    # Suphan Buri — positive bias +0.8, negative skill at h6/h24
    "SPB_01": {
        6:  StationHorizonConfig(blend_weight=0.5,  danger_boost=0.0),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        24: StationHorizonConfig(blend_weight=0.5,  danger_boost=0.0),
    },
    # Nakhon Sawan — positive bias +1.0, negative skill at h6/h24
    "NSW_01": {
        6:  StationHorizonConfig(blend_weight=0.6,  danger_boost=0.0),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        24: StationHorizonConfig(blend_weight=0.6,  danger_boost=0.0),
    },
    # Phuket — no extreme heat events in training data, island climate
    "HKT_01": {
        6:  StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        24: StationHorizonConfig(blend_weight=0.6,  danger_boost=0.0),
    },
    # Nakhon Ratchasima — danger recall 45.8% at h12
    "NMA_01": {
        6:  StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
        12: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.3),
        24: StationHorizonConfig(blend_weight=1.0,  danger_boost=0.0),
    },
}


def get_safety_config(station_id: str, horizon_h: int) -> StationHorizonConfig:
    """Return the safety config for a (station, horizon) pair.

    Falls back to defaults (no blend, no boost, enabled) for unseen combos.
    """
    station_cfg = STATION_SAFETY_CONFIG.get(station_id, {})
    return station_cfg.get(horizon_h, StationHorizonConfig())
