"""Feature engineering for HeatShield AI forecast models."""
from app.ml.forecast.features.builders import (
    _DEFAULT_LAGS_H,
    _DEFAULT_ROLLING_H,
    _load_feature_config,
    _get_lags_for_horizon,
    _get_rolling_for_horizon,
    _subset_features_for_horizon,
    build_X_once,
    build_y_for_horizon,
    build_features,
    get_feature_names,
    calculate_required_history_hours,
)
from app.ml.forecast.features.physics import (
    add_heat_index_col,
)
from app.ml.forecast.features.temporal import (
    enhanced_temporal_embeddings,
    time2vec_embedding,
    add_seasonal_features,
    add_hour_of_day_features,
)
__all__ = [
    "_DEFAULT_LAGS_H",
    "_DEFAULT_ROLLING_H",
    "_load_feature_config",
    "_get_lags_for_horizon",
    "_get_rolling_for_horizon",
    "_subset_features_for_horizon",
    "build_X_once",
    "build_y_for_horizon",
    "build_features",
    "get_feature_names",
    "add_heat_index_col",
    "calculate_required_history_hours",
    "enhanced_temporal_embeddings",
    "time2vec_embedding",
    "add_seasonal_features",
    "add_hour_of_day_features",
]
