"""Ensemble blending methods for heat index forecasting.

Provides simple blending strategies to combine model predictions with
persistence and climatology baselines for improved robustness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


def blend_with_persistence(
    model_pred: np.ndarray,
    persistence_pred: np.ndarray,
    weight_model: float = 0.5,
) -> np.ndarray:
    """Blend model prediction with persistence baseline.

    Simple weighted average: y_pred = w * model_pred + (1-w) * persistence_pred

    Args:
        model_pred: Model predictions (heat index in °C)
        persistence_pred: Persistence baseline predictions
        weight_model: Weight for model prediction (0.0-1.0)

    Returns:
        Blended predictions
    """
    weight_persistence = 1.0 - weight_model
    return weight_model * model_pred + weight_persistence * persistence_pred


def blend_with_climatology(
    model_pred: np.ndarray,
    climatology_pred: np.ndarray,
    weight_model: float = 0.7,
) -> np.ndarray:
    """Blend model prediction with climatology baseline.

    Args:
        model_pred: Model predictions (heat index in °C)
        climatology_pred: Climatology baseline predictions
        weight_model: Weight for model prediction (0.0-1.0)

    Returns:
        Blended predictions
    """
    weight_clim = 1.0 - weight_model
    return weight_model * model_pred + weight_clim * climatology_pred


def triple_blend(
    model_pred: np.ndarray,
    persistence_pred: np.ndarray,
    climatology_pred: np.ndarray,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> np.ndarray:
    """Blend model with both persistence and climatology.

    Args:
        model_pred: Model predictions
        persistence_pred: Persistence baseline
        climatology_pred: Climatology baseline
        weights: (weight_model, weight_persistence, weight_climatology)

    Returns:
        Blended predictions
    """
    w_model, w_pers, w_clim = weights
    total = w_model + w_pers + w_clim
    if abs(total - 1.0) > 1e-6:
        # Normalize if weights don't sum to 1
        w_model /= total
        w_pers /= total
        w_clim /= total

    return w_model * model_pred + w_pers * persistence_pred + w_clim * climatology_pred


def adaptive_blend(
    model_pred: np.ndarray,
    persistence_pred: np.ndarray,
    recent_errors: np.ndarray,
    window: int = 24,
) -> np.ndarray:
    """Adaptively blend based on recent model performance.

    If model has been performing poorly recently (high errors), give more
    weight to persistence. This is useful for models that struggle against
    persistence on certain time periods.

    Args:
        model_pred: Model predictions
        persistence_pred: Persistence baseline predictions
        recent_errors: Recent model absolute errors (MAE over last `window` points)
        window: Number of recent points to consider

    Returns:
        Adaptively blended predictions
    """
    if len(recent_errors) == 0:
        # No error history, default to equal weight
        return 0.5 * model_pred + 0.5 * persistence_pred

    # Calculate normalized error metric (0 = perfect, 1 = terrible)
    # Use rolling mean of recent errors
    recent_mae = np.mean(recent_errors[-window:])
    # Normalize by typical error range (empirical: 0.5-3.0°C)
    normalized_error = np.clip((recent_mae - 0.5) / 2.5, 0.0, 1.0)

    # Higher error -> lower model weight
    weight_model = 1.0 - 0.5 * normalized_error  # Range: 0.5-1.0
    weight_persistence = 1.0 - weight_model

    return weight_model * model_pred + weight_persistence * persistence_pred


def quantile_ensemble(
    predictions_list: list[np.ndarray],
    method: Literal["mean", "median", "trimmed_mean"] = "median",
    trim: float = 0.1,
) -> np.ndarray:
    """Ensemble multiple prediction sources using statistical methods.

    Args:
        predictions_list: List of prediction arrays from different sources
        method: Ensemble method ("mean", "median", "trimmed_mean")
        trim: Fraction to trim from each end for trimmed_mean (0.0-0.5)

    Returns:
        Ensembled predictions
    """
    if not predictions_list:
        raise ValueError("predictions_list cannot be empty")

    stacked = np.stack(predictions_list, axis=0)

    if method == "mean":
        return np.mean(stacked, axis=0)
    elif method == "median":
        return np.median(stacked, axis=0)
    elif method == "trimmed_mean":
        # Trim from both ends
        k = int(len(predictions_list) * trim)
        if k == 0:
            return np.mean(stacked, axis=0)
        sorted_preds = np.sort(stacked, axis=0)
        trimmed = sorted_preds[k:-k] if k > 0 else sorted_preds
        return np.mean(trimmed, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_persistence_baseline(
    recent_hi: np.ndarray,
    horizon_h: int,
) -> np.ndarray:
    """Compute persistence baseline predictions.

    For horizon h, use heat index from h hours ago as prediction.

    Args:
        recent_hi: Recent heat index observations
        horizon_h: Forecast horizon in hours

    Returns:
        Persistence predictions (same length as recent_hi)
    """
    if horizon_h >= len(recent_hi):
        # Not enough history, return NaN
        return np.full_like(recent_hi, np.nan)

    # Shift by horizon hours
    persistence = np.roll(recent_hi, horizon_h)
    # First horizon_h values are NaN (no history)
    persistence[:horizon_h] = np.nan

    return persistence


def compute_climatology_baseline(
    recent_hi: np.ndarray,
    hour_of_day: np.ndarray,
) -> np.ndarray:
    """Compute climatology baseline predictions.

    Predict mean heat index for each hour of day.

    Args:
        recent_hi: Recent heat index observations
        hour_of_day: Hour of day for each observation (0-23)

    Returns:
        Climatology predictions
    """
    # Compute hourly means
    hourly_means = {}
    for h in range(24):
        mask = hour_of_day == h
        if mask.sum() > 0:
            hourly_means[h] = recent_hi[mask].mean()
        else:
            hourly_means[h] = np.nan

    # Map each observation to its hour's mean
    climatology = np.array([hourly_means.get(h, np.nan) for h in hour_of_day])

    return climatology
