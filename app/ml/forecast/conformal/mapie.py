"""MAPIE adaptive conformal inference (ACI) adapter for HeatShield AI forecasters.

Wraps any fitted BaseForecaster and calibrates its prediction intervals using
MapieTimeSeriesRegressor in ACI mode (Angelopoulos et al., 2023).

Primary use-case: CEI_01 h24 whose raw PI coverage is 79.9% vs the ≥80% gate.
ACI calibrates on a held-out calibration set and adapts coverage continuously.

Requirements:
    pip install 'mapie>=0.9,<2'
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from mapie.time_series_regression import MapieTimeSeriesRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False


class _MeanPredictor:
    """sklearn-compatible adapter that exposes a fitted forecaster's mean predictions."""

    def __init__(self, forecaster: Any) -> None:
        self._forecaster = forecaster

    def fit(self, X: Any, y: Any) -> "_MeanPredictor":
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        bundle = self._forecaster.predict_with_pi(X)
        return bundle.hi_mean


class MapieCalibrator:
    """Calibrate prediction intervals using MAPIE ACI for time-series forecasters.

    Usage::

        cal = MapieCalibrator(alpha=0.1)
        cal.fit(forecaster, X_calib, y_calib)
        mean, lower, upper = cal.predict_with_pi(X_test)
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not MAPIE_AVAILABLE:
            raise ImportError(
                "mapie>=0.9 is required for adaptive conformal calibration. "
                "Install with: pip install 'mapie>=0.9,<2'"
            )
        self.alpha = alpha
        self._mapie: "MapieTimeSeriesRegressor | None" = None

    def fit(
        self,
        forecaster: Any,
        X_calib: pd.DataFrame | np.ndarray,
        y_calib: np.ndarray,
    ) -> "MapieCalibrator":
        """Fit MAPIE on calibration data.

        Args:
            forecaster: Already-fitted BaseForecaster instance.
            X_calib: Calibration feature matrix.
            y_calib: True heat-index values for the calibration split.

        Returns:
            self (for chaining).
        """
        wrapper = _MeanPredictor(forecaster)
        self._mapie = MapieTimeSeriesRegressor(
            estimator=wrapper,
            method="aci",
            cv="prefit",
        )
        self._mapie.fit(X_calib, y_calib)
        logger.info(
            "MapieCalibrator fitted on %d calibration samples (alpha=%.2f)",
            len(y_calib),
            self.alpha,
        )
        return self

    def predict_with_pi(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower, upper) arrays with calibrated prediction intervals.

        Args:
            X: Feature matrix.
            alpha: Miscoverage rate (1 - coverage). Defaults to self.alpha.

        Returns:
            Tuple of (mean_pred, lower_bound, upper_bound).
        """
        if self._mapie is None:
            raise RuntimeError("Call fit() before predict_with_pi().")
        alpha = alpha if alpha is not None else self.alpha
        y_pred, y_pis = self._mapie.predict(X, alpha=alpha)
        # y_pis shape: (n_samples, 2, n_alpha)
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]
        return y_pred, lower, upper
