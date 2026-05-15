"""
Heat Index Forecast Module

This module provides LightGBM quantile regression models for heat index forecasting
across multiple Thai meteorological stations.

Models support 6h, 12h, and 24h forecast horizons with 108 engineered features
including temporal, geographic, meteorological, lag, and rolling statistics features.
"""

# Supported stations
STATIONS = ["BKK_01", "CNX_01", "KKN_01", "HYI_01", "RYG_01"]

# Station names mapping
STATION_NAMES = {
    "BKK_01": "Bangkok",
    "CNX_01": "Chiang Mai",
    "KKN_01": "Khon Kaen",
    "HYI_01": "Nong Khai",
    "RYG_01": "Rayong",
}

# Supported horizons (in hours)
HORIZONS = [6, 12, 24]

# Risk categories
RISK_CATEGORIES = ["Caution", "Extreme Caution", "Danger", "Extreme Danger"]

# Risk thresholds (°C)
RISK_THRESHOLDS = {
    "Caution": (27, 32),
    "Extreme Caution": (32, 40),
    "Danger": (40, 54),
    "Extreme Danger": (54, float('inf')),
}

# Model registry path
MODEL_REGISTRY_PATH = "app/models/forecast_v3"

# Backend name
BACKEND_NAME = "lightgbm_quantile"

# Target kind
TARGET_KIND = "th"
