"""Base class and utilities for ML forecast backends.

Provides common patterns shared across LightGBM, CatBoost, and XGBoost backends:
- Device detection and configuration
- Sample weight calculation for tail-class emphasis
- Ensemble seed management
- Common training phases (pilot, refit, calibrate)
"""
from __future__ import annotations

import logging
import os
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

_SEEDS = [42, 123, 7]  # Ensemble seeds for stability


def make_sample_weight(y: np.ndarray) -> np.ndarray:
    """Tail-class sample weights: 5x for HI>38°C, 9x for HI>42°C.
    
    Args:
        y: Target values (heat index in Celsius)
        
    Returns:
        Sample weights array with same shape as y
    """
    return 1.0 + 4.0 * (y > 38).astype(float) + 8.0 * (y > 42).astype(float)


def detect_device(
    backend: Literal["lightgbm", "catboost", "xgboost"],
    force_cpu: bool = False,
) -> Literal["cpu", "gpu", "cuda"]:
    """Detect best device for the given backend.
    
    Args:
        backend: Which ML backend to check
        force_cpu: If True, force CPU regardless of GPU availability
        
    Returns:
        Device string: "cpu", "gpu", or "cuda"
    """
    if force_cpu or os.getenv("HEATSHIELD_FORCE_CPU") == "1":
        return "cpu"
    
    # Check for explicit device override
    if backend == "lightgbm":
        lgbm_device = os.getenv("LGBM_DEVICE", "").lower()
        if lgbm_device in ("cpu", "gpu", "cuda"):
            return lgbm_device
    
    # Auto-detect GPU
    if backend == "lightgbm":
        return _detect_lgbm_device()
    elif backend == "catboost":
        return _detect_catboost_device()
    elif backend == "xgboost":
        return _detect_xgboost_device()
    else:
        return "cpu"


def _detect_lgbm_device() -> Literal["cpu", "gpu", "cuda"]:
    """Detect LightGBM device preference."""
    try:
        import lightgbm as lgb
        # Try CUDA first
        try:
            lgb.train(
                {"objective": "regression", "device_type": "cuda", "verbose": -1, "num_leaves": 4},
                lgb.Dataset(np.zeros((4, 2)), np.zeros(4)),
                num_boost_round=1,
            )
            logger.info("LightGBM CUDA detected — using CUDA")
            return "cuda"
        except Exception:
            pass
        # Try OpenCL GPU
        try:
            lgb.train(
                {"objective": "regression", "device_type": "gpu", "verbose": -1, "num_leaves": 4},
                lgb.Dataset(np.zeros((4, 2)), np.zeros(4)),
                num_boost_round=1,
            )
            logger.info("LightGBM GPU detected — using GPU")
            return "gpu"
        except Exception:
            pass
    except ImportError:
        pass
    logger.info("LightGBM: no GPU detected — using CPU")
    return "cpu"


def _detect_catboost_device() -> Literal["cpu", "gpu"]:
    """Detect CatBoost device preference."""
    try:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(
            iterations=1,
            task_type="GPU",
            verbose=0,
            allow_writing_files=False,
        )
        m.fit(np.zeros((4, 2)), np.zeros(4), verbose=False)
        logger.info("CatBoost GPU detected — using GPU")
        return "GPU"
    except Exception:
        logger.info("CatBoost: no GPU detected — using CPU")
        return "CPU"


def _detect_xgboost_device() -> Literal["cpu", "cuda"]:
    """Detect XGBoost device preference."""
    try:
        import xgboost as xgb
        info = xgb.build_info()
        if info.get("USE_CUDA"):
            logger.info("XGBoost CUDA detected — using CUDA")
            return "cuda"
    except Exception:
        pass
    logger.info("XGBoost: no CUDA detected — using CPU")
    return "cpu"


class BaseBackendForecaster:
    """Base class for ML forecast backends with common patterns."""
    
    backend_name: str = "base"
    target_kind: str = "hi"
    
    def __init__(
        self,
        n_trials: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self.feature_list: list[str] = []
        self._station_id: str = ""
        self._horizon_h: int = 24
        self._metadata: dict = {}
    
    def _validate_inputs(self, X, y) -> None:
        """Validate training inputs."""
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        
        valid = y.notna() if hasattr(y, 'notna') else ~np.isnan(y)
        if valid.sum() == 0:
            raise ValueError("No valid (non-NaN) samples in y")
    
    def _log_training_start(self, device: str, n_rows: int) -> None:
        """Log training start with device info."""
        logger.info(
            "%s fit: station=%s h=%d device=%s rows=%d",
            self.backend_name,
            self._station_id,
            self._horizon_h,
            device,
            n_rows,
        )
