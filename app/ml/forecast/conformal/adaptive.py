"""Adaptive conformal prediction for time series forecasting.

Implements adaptive conformal prediction methods that adjust interval width
based on difficulty and handle non-stationary time series.
Based on recent research: Gibbs et al. (2022), CPTC (2024).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

try:
    from conformal_forecast import AdaptiveConformal
    CONFORMAL_FORECAST_AVAILABLE = True
except ImportError:
    CONFORMAL_FORECAST_AVAILABLE = False


class AdaptiveConformalPredictor:
    """Adaptive conformal prediction with rolling window quantiles."""

    def __init__(self, alpha: float = 0.1, window_size: int = 100):
        """Initialize adaptive conformal predictor.

        Args:
            alpha: Target miscoverage rate (1 - coverage)
            window_size: Rolling window size for quantile computation
        """
        self.alpha = alpha
        self.window_size = window_size
        self.quantiles: List[float] = []

    def fit(self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> None:
        """Compute adaptive conformal quantiles from calibration data.

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Compute residuals
        residuals = np.abs(y_true - y_pred)

        # Compute adaptive quantiles with rolling window
        self.quantiles = []
        for i in range(len(residuals)):
            start = max(0, i - self.window_size)
            window_residuals = residuals[start : i + 1]

            # Quantile with finite-sample correction
            if len(window_residuals) > 0:
                correction = 1 + 1 / len(window_residuals)
                q = np.quantile(
                    window_residuals, (1 - self.alpha) * correction, method="higher"
                )
            else:
                q = 0.0
            self.quantiles.append(q)

    def predict(
        self, y_pred: np.ndarray | pd.Series, index: int
    ) -> Tuple[float, float]:
        """Return prediction interval for a single prediction.

        Args:
            y_pred: Predicted value
            index: Index for quantile lookup

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if index >= len(self.quantiles):
            # Use last quantile if index out of range
            q = self.quantiles[-1] if self.quantiles else 0.0
        else:
            q = self.quantiles[index]

        return (y_pred - q, y_pred + q)

    def predict_batch(
        self, y_pred: np.ndarray | pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return prediction intervals for batch of predictions.

        Args:
            y_pred: Predicted values

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        n = len(y_pred)
        lower = np.zeros(n)
        upper = np.zeros(n)

        for i in range(n):
            lower[i], upper[i] = self.predict(y_pred[i], i)

        return lower, upper


class CPTCPredictor:
    """Conformal Prediction with Change Points (CPTC).

    Handles regime shifts by computing quantiles within each detected regime.
    Based on 2024 research on conformal prediction with changepoints.
    """

    def __init__(self, alpha: float = 0.1):
        """Initialize CPTC predictor.

        Args:
            alpha: Target miscoverage rate (1 - coverage)
        """
        self.alpha = alpha
        self.changepoints: List[int] = []
        self.quantiles: List[float] = []

    def detect_changepoints(
        self, residuals: np.ndarray, min_size: int = 50, penalty: float = 10.0
    ) -> List[int]:
        """Detect changepoints in residuals using simple variance method.

        Args:
            residuals: Residual values
            min_size: Minimum segment size
            penalty: Penalty for changepoint detection

        Returns:
            List of changepoint indices
        """
        # Simple variance-based changepoint detection
        # (Full PELT implementation would be in changepoint.py)
        changepoints = []
        rolling_var = pd.Series(residuals).rolling(window=min_size).var()
        rolling_mean = pd.Series(residuals).rolling(window=min_size).mean()

        # Detect points where variance shifts significantly
        for i in range(min_size, len(residuals) - min_size):
            var_before = rolling_var.iloc[i - min_size]
            var_after = rolling_var.iloc[i]
            mean_before = rolling_mean.iloc[i - min_size]
            mean_after = rolling_mean.iloc[i]

            if var_before > 0 and var_after > 0:
                var_ratio = var_after / var_before
                mean_diff = abs(mean_after - mean_before)

                if var_ratio > 2.0 or var_ratio < 0.5 or mean_diff > penalty:
                    changepoints.append(i)

        return changepoints

    def fit(self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> None:
        """Fit CPTC with changepoint-aware quantiles.

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Compute residuals
        residuals = np.abs(y_true - y_pred)

        # Detect changepoints
        self.changepoints = self.detect_changepoints(residuals)

        # Compute quantiles within each regime
        self.quantiles = []
        cp_idx = 0
        regime_start = 0

        for i in range(len(residuals)):
            if cp_idx < len(self.changepoints) and i >= self.changepoints[cp_idx]:
                regime_start = i
                cp_idx += 1

            # Get residuals in current regime
            regime_residuals = residuals[regime_start : i + 1]

            if len(regime_residuals) > 0:
                correction = 1 + 1 / len(regime_residuals)
                q = np.quantile(
                    regime_residuals, (1 - self.alpha) * correction, method="higher"
                )
            else:
                q = 0.0
            self.quantiles.append(q)

    def predict(
        self, y_pred: np.ndarray | pd.Series, index: int
    ) -> Tuple[float, float]:
        """Return prediction interval for a single prediction.

        Args:
            y_pred: Predicted value
            index: Index for quantile lookup

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if index >= len(self.quantiles):
            q = self.quantiles[-1] if self.quantiles else 0.0
        else:
            q = self.quantiles[index]

        return (y_pred - q, y_pred + q)

    def predict_batch(
        self, y_pred: np.ndarray | pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return prediction intervals for batch of predictions.

        Args:
            y_pred: Predicted values

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        n = len(y_pred)
        lower = np.zeros(n)
        upper = np.zeros(n)

        for i in range(n):
            lower[i], upper[i] = self.predict(y_pred[i], i)

        return lower, upper


class ConformalPIDController:
    """Conformal prediction with PID control for adaptive coverage.

    Uses proportional-integral-derivative control to track desired coverage.
    Based on Zhou et al. (2022).
    """

    def __init__(
        self, alpha: float = 0.1, kp: float = 0.5, ki: float = 0.1, kd: float = 0.01
    ):
        """Initialize conformal PID controller.

        Args:
            alpha: Target miscoverage rate (1 - coverage)
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.alpha = alpha
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_error = 0.0
        self.last_error = 0.0
        self.quantile = 0.0

    def fit(
        self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> None:
        """Fit PID controller to achieve target coverage.

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Start with simple quantile
        residuals = np.abs(y_true - y_pred)
        self.quantile = np.quantile(residuals, 1 - self.alpha, method="higher")

        # Online PID control (simplified - would need sequential data for full implementation)
        # This is a placeholder for the full PID control implementation
        # which requires streaming data and online updates

    def predict(self, y_pred: float) -> Tuple[float, float]:
        """Return prediction interval using current quantile.

        Args:
            y_pred: Predicted value

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        return (y_pred - self.quantile, y_pred + self.quantile)

    def update(
        self, y_true: float, y_pred: float, covered: bool
    ) -> None:
        """Update PID controller state based on coverage feedback.

        Args:
            y_true: True value
            y_pred: Predicted value
            covered: Whether true value was in prediction interval
        """
        # Compute error (1 if covered, 0 if not)
        error = 1.0 if covered else 0.0

        # PID control update
        self.integral_error += error
        derivative = error - self.last_error

        # Adjust quantile
        adjustment = (
            self.kp * error + self.ki * self.integral_error + self.kd * derivative
        )
        self.quantile *= (1 + adjustment)

        # Clip quantile to reasonable range
        self.quantile = max(0.1, min(10.0, self.quantile))

        self.last_error = error
