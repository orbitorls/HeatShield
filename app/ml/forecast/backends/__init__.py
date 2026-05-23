"""Pluggable forecast backends for HeatShield AI v3."""

from app.ml.forecast.backends.xgb_safety_backend import XGBoostSafetyForecaster
from app.ml.forecast.backends.lgbm import LGBMForecaster, LGBMDirectHIForecaster
from app.ml.forecast.backends.xgb_backend import XGBForecaster
from app.ml.forecast.backends.catboost_backend import CatBoostForecaster

__all__ = [
    "XGBoostSafetyForecaster",
    "LGBMForecaster",
    "LGBMDirectHIForecaster",
    "XGBForecaster",
    "CatBoostForecaster",
]
