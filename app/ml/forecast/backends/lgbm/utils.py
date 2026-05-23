"""LightGBM utility functions — device detection, training helpers, HI computation"""
from __future__ import annotations

import hashlib
import json
import logging
import gc
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit

from app.ml.forecast.base import BaseForecaster, PredictionBundle
from app.ml.forecast.conformal import EnbPICalibrator, MondianCQRCalibrator
from app.ml.forecast.danger_gate import DangerGate
from app.ml.forecast.hpo.optuna_utils import feature_signature, forecast_model_root, write_heartbeat
from app.ml.forecast.splitting import apply_feature_medians, fit_feature_medians, split_xy, split_xy_4way

logger = logging.getLogger(__name__)

_QUANTILES = [0.05, 0.50, 0.95, 0.97]
_TARGETS = ["temp_c", "rh"]
_SEEDS = [42, 123, 7]          # ensemble seeds for median (q50) — needs 3 for stability
_SEEDS_TAIL = [42]             # single seed for tail quantiles (q05/q95/q97) — sufficient
_CV_SPLITS = 2
_CV_GAP_HOURS = 72
_LGBM_DEVICE_SUPPORT: dict[str, bool] = {}
_HW_DEVICE: str | None = None
_DEVICE_LOGGED: bool = False

# Detect best LightGBM device.
# Hardware probe runs once at import; env overrides are re-read on every call.
# Priority: LGBM_DEVICE env > HEATSHIELD_FORCE_CPU env > CUDA > benchmark(GPU vs CPU)

def _benchmark_device(device: str, n_rows: int = 30_000, n_cols: int = 50, rounds: int = 100) -> float:
    """Return seconds to train one booster on synthetic data of representative size."""
    import time
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X_bench = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    y_bench = rng.standard_normal(n_rows).astype(np.float32)
    t0 = time.perf_counter()
    lgb.train(
        {"objective": "regression", "device_type": device, "verbose": -1,
         "num_leaves": 64, "n_jobs": -1 if device == "cpu" else 1},
        lgb.Dataset(X_bench, y_bench),
        num_boost_round=rounds,
    )
    return time.perf_counter() - t0


def _probe_lgbm_hardware() -> str:
    """Probe available devices; benchmark GPU vs CPU and choose the faster one."""
    import lightgbm as lgb

    # CUDA is always faster than CPU for any meaningful workload — skip benchmark.
    try:
        lgb.train({"objective": "regression", "device_type": "cuda", "verbose": -1,
                   "num_leaves": 4},
                  lgb.Dataset(np.zeros((4, 2)), np.zeros(4)),
                  num_boost_round=1)
        logger.info("LightGBM CUDA detected — using CUDA")
        _LGBM_DEVICE_SUPPORT["cuda"] = True
        _LGBM_DEVICE_SUPPORT["gpu"] = True
        return "cuda"
    except Exception:
        _LGBM_DEVICE_SUPPORT["cuda"] = False

    # OpenCL GPU (T4/V100): benchmark against CPU since T4 OpenCL is often
    # slower than multi-threaded CPU for tabular data <100k rows.
    gpu_available = False
    try:
        lgb.train({"objective": "regression", "device_type": "gpu", "verbose": -1,
                   "num_leaves": 4},
                  lgb.Dataset(np.zeros((4, 2)), np.zeros(4)),
                  num_boost_round=1)
        _LGBM_DEVICE_SUPPORT["gpu"] = True
        gpu_available = True
    except Exception:
        _LGBM_DEVICE_SUPPORT["gpu"] = False

    if not gpu_available:
        logger.info("LightGBM: no GPU detected — using CPU")
        return "cpu"

    try:
        cpu_t = _benchmark_device("cpu")
        gpu_t = _benchmark_device("gpu")
        # Require GPU to be at least 15% faster to prefer it (CPU parallelises refit better).
        if gpu_t < cpu_t * 0.85:
            logger.info("LightGBM benchmark: GPU %.2fs < CPU %.2fs — using GPU", gpu_t, cpu_t)
            return "gpu"
        else:
            logger.info("LightGBM benchmark: CPU %.2fs <= GPU %.2fs — using CPU", cpu_t, gpu_t)
            return "cpu"
    except Exception:
        logger.warning("LightGBM benchmark failed; falling back to CPU")
        return "cpu"


def _supports_lgbm_device(device_type: str) -> bool:
    """Return whether a specific LightGBM device type is usable."""
    key = device_type.lower().strip()
    if key in _LGBM_DEVICE_SUPPORT:
        return _LGBM_DEVICE_SUPPORT[key]
    try:
        import lightgbm as lgb
        lgb.train(
            {"objective": "regression", "device_type": key, "verbose": -1, "num_leaves": 4},
            lgb.Dataset(np.zeros((8, 2)), np.zeros(8)),
            num_boost_round=1,
        )
        _LGBM_DEVICE_SUPPORT[key] = True
    except Exception:
        _LGBM_DEVICE_SUPPORT[key] = False
    return _LGBM_DEVICE_SUPPORT[key]


def _detect_device() -> str:
    """Return best LightGBM device, re-reading env vars on every call.

    Hardware capability is probed once at import (_HW_DEVICE); env overrides
    (LGBM_DEVICE, HEATSHIELD_FORCE_CPU) are re-read each call so test fixtures
    and shell changes made after import are respected.
    """
    global _HW_DEVICE, _DEVICE_LOGGED
    import os
    env = os.environ.get("LGBM_DEVICE", "").lower()
    if env:
        if _supports_lgbm_device(env):
            if not _DEVICE_LOGGED:
                logger.info("LightGBM device forced via LGBM_DEVICE=%s", env)
                _DEVICE_LOGGED = True
            return env
        logger.warning("LGBM_DEVICE=%s is not supported by this LightGBM build. Falling back to auto-detect.", env)
    if os.environ.get("HEATSHIELD_FORCE_CPU") == "1":
        if not _DEVICE_LOGGED:
            logger.info("LightGBM forced to CPU (HEATSHIELD_FORCE_CPU=1)")
            _DEVICE_LOGGED = True
        return "cpu"
    if _HW_DEVICE is None:
        _HW_DEVICE = _probe_lgbm_hardware()
    return _HW_DEVICE


_DEVICE_TYPE = "auto"  # kept for external references; actual device is resolved lazily

# Cross-slot warm-start: stores best Optuna params from the most recent completed slot
# per target_kind so the next slot can enqueue them as a starting point for TPE.
_last_best_params: dict[str, dict] = {}

# Default LightGBM params (overridden by Optuna)
_DEFAULT_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "verbose": -1,
    "max_bin": 63,  # GPU-friendly: smaller bins = faster training
}

# Change 6: persistent Optuna study path
def _optuna_storage_path() -> Path:
    return forecast_model_root() / "optuna_studies.db"


def _lgbm_thread_count(device: str, parallel_jobs: int = 1) -> int | None:
    """Avoid CPU oversubscription when multiple boosters train in parallel."""
    if device != "cpu":
        return None
    if os.environ.get("LGBM_N_JOBS"):
        return max(1, int(os.environ["LGBM_N_JOBS"]))
    cores = os.cpu_count() or 1
    return max(1, cores // max(1, parallel_jobs))


def _lgbm_device_params(device: str, parallel_jobs: int = 1) -> dict:
    params = {"device_type": device}
    if device in ("cuda", "gpu"):
        params["gpu_use_dp"] = False  # Single precision for faster GPU training
    n_jobs = _lgbm_thread_count(device, parallel_jobs=parallel_jobs)
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    return params


def _sanitize_lgbm_params_for_device(params: dict, device: str) -> dict:
    """Return LightGBM params adjusted for known device constraints."""
    sanitized = dict(params)
    if device.lower().strip() == "gpu" and int(sanitized.get("max_bin", 0) or 0) > 255:
        sanitized["max_bin"] = 255
    return sanitized


def _remaining_trials(study, requested_trials: int) -> int:
    """Run Optuna up to requested complete trials, not requested extra trials."""
    complete = sum(t.state.name == "COMPLETE" for t in study.trials)
    return max(0, requested_trials - complete)


def _compute_hi_array(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Fully vectorized heat-index computation replicating app.core.heat_index.compute().

    Implements:
    - Steadman simple formula for temp_c < 27°C or rh < 40%
    - Rothfusz regression (via °F) for the main regime
    - Low-humidity adjustment (rh < 13% and 80 <= t_f <= 112)
    - High-humidity adjustment (rh > 85% and 80 <= t_f <= 87)
    All in NumPy, no Python loop.
    """
    # Change 2: vectorized, no scalar loop, no import from app.core.heat_index
    T = np.asarray(temp_c, dtype=np.float64)
    R = np.clip(np.asarray(rh, dtype=np.float64), 0.0, 100.0)

    # °C → °F
    T_f = T * 9.0 / 5.0 + 32.0

    # Steadman simple formula (used when temp_c < 27 or rh < 40)
    simple_hi_f = 0.5 * (T_f + 61.0 + (T_f - 68.0) * 1.2 + R * 0.094)
    simple_hi_c = (simple_hi_f - 32.0) * 5.0 / 9.0

    # Rothfusz regression (operates in °F)
    rothfusz_hi_f = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * R
        + -0.22475541 * T_f * R
        + -0.00683783 * T_f**2
        + -0.05481717 * R**2
        + 0.00122874 * T_f**2 * R
        + 0.00085282 * T_f * R**2
        + -0.00000199 * T_f**2 * R**2
    )

    # Low-humidity adjustment: rh < 13 and 80 <= t_f <= 112
    low_rh_mask = (R < 13.0) & (T_f >= 80.0) & (T_f <= 112.0)
    low_adj = ((13.0 - R) / 4.0) * np.sqrt(np.maximum(0.0, (17.0 - np.abs(T_f - 95.0)) / 17.0))
    rothfusz_hi_f_adjusted = np.where(low_rh_mask, rothfusz_hi_f - low_adj, rothfusz_hi_f)

    # High-humidity adjustment: rh > 85 and 80 <= t_f <= 87
    high_rh_mask = (R > 85.0) & (T_f >= 80.0) & (T_f <= 87.0) & ~low_rh_mask
    high_adj = ((R - 85.0) / 10.0) * ((87.0 - T_f) / 5.0)
    rothfusz_hi_f_adjusted = np.where(high_rh_mask, rothfusz_hi_f_adjusted + high_adj, rothfusz_hi_f_adjusted)

    rothfusz_hi_c = (rothfusz_hi_f_adjusted - 32.0) * 5.0 / 9.0

    # Select regime: Steadman when temp_c < 27 or rh < 40, else Rothfusz
    use_simple = (T < 27.0) | (R < 40.0)
    return np.where(use_simple, simple_hi_c, rothfusz_hi_c)


def _compute_dense_weights(
    y_hi: np.ndarray,
    *,
    alpha: float = 0.8,
    min_weight: float = 0.25,
    max_weight: float = 6.0,
) -> np.ndarray:
    """Compute density-based sample weights with clipping.

    Lower-density (rarer) samples get higher weights. Weights are normalized
    to mean 1 and clipped to avoid gradient explosions.
    """
    y = np.asarray(y_hi, dtype=float)
    if len(y) < 20 or np.nanstd(y) < 1e-6:
        return np.ones(len(y), dtype=float)
    try:
        # Histogram-based density estimation (O(n) instead of O(n²) KDE)
        counts, edges = np.histogram(y, bins=100, density=True)
        indices = np.clip(np.searchsorted(edges, y) - 1, 0, len(counts) - 1)
        density = np.clip(counts[indices], 1e-9, None)
        weight = 1.0 / (density ** float(alpha))
        weight = weight / max(float(np.mean(weight)), 1e-9)
        return np.clip(weight, min_weight, max_weight)
    except Exception:
        return np.ones(len(y), dtype=float)


def _compute_combined_weights(
    y_hi: np.ndarray,
    *,
    dense_alpha: float = 0.8,
    danger_alpha: float = 3.0,
    danger_threshold: float = 40.0,
    min_weight: float = 0.25,
    max_weight: float = 6.0,
) -> np.ndarray:
    """Density-aware + danger-aware sample weights (Focal-L1 style).

    Merges KDE-based rarity weighting with focal up-weighting for
    high-heat samples (HI >= danger_threshold). Penalises missing
    rare danger events more than over-predicting them.
    """
    dense = _compute_dense_weights(
        y_hi, alpha=dense_alpha, min_weight=min_weight, max_weight=max_weight
    )
    focal = np.ones(len(y_hi), dtype=float)
    danger_mask = y_hi >= danger_threshold
    near_mask = (y_hi >= danger_threshold - 2.0) & (y_hi < danger_threshold)
    focal[danger_mask] = danger_alpha
    focal[near_mask] = max(1.0, danger_alpha * 0.5)
    focal = focal / max(float(np.mean(focal)), 1e-9)
    combined = dense * focal
    return np.clip(combined, min_weight, max_weight)


def _danger_alpha_for_horizon(horizon_h: int) -> float:
    """Return danger focal weight scaled by horizon.

    Short horizons (h6/h12) get moderate up-weighting because the model
    has enough recent context to detect danger accurately.
    Long horizons (h48/h72) get aggressive up-weighting because danger
    events become harder to predict and the cost of missing them is higher.
    """
    return {6: 3.0, 12: 3.0, 24: 4.0, 48: 5.0, 72: 6.0}.get(horizon_h, 3.0)


def _expanding_window_splits(
    n_rows: int,
    *,
    gap: int,
    n_splits: int = _CV_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-window folds with purge gap."""
    if n_rows <= gap + 48:
        return []
    # Ensure fold count remains valid for short series.
    split_count = max(2, min(n_splits, (n_rows // max(gap + 24, 48))))
    splitter = TimeSeriesSplit(n_splits=split_count, gap=gap)
    return list(splitter.split(np.arange(n_rows)))


def _train_single_booster(
    target: str,
    alpha: float,
    seed: int,
    X_train: pd.DataFrame,
    y_target: np.ndarray,
    X_val: pd.DataFrame,
    y_val_target: np.ndarray,
    weights: np.ndarray,
    best_params: dict,
    parallel_jobs: int,
):
    """Train a single LightGBM booster for one (target, quantile, seed) combination.

    This function is defined at module level so joblib can pickle it.
    """
    import lightgbm as lgb

    params = {
        **_DEFAULT_PARAMS,
        **best_params,
        **_lgbm_device_params(_detect_device(), parallel_jobs=parallel_jobs),
        "alpha": alpha,
        "seed": seed,
    }
    dtrain = lgb.Dataset(X_train, label=y_target, weight=weights)
    dval = lgb.Dataset(X_val, label=y_val_target, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )


def _make_lgb_dataset_pair(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """Construct reusable LightGBM bins for serial booster training."""
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_train, free_raw_data=False)
    dtrain.construct()
    dval = lgb.Dataset(X_val, reference=dtrain, free_raw_data=False)
    dval.construct()
    return dtrain, dval


def _train_cached_lgb_booster(
    dtrain,
    dval,
    y_train: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    params: dict,
):
    import lightgbm as lgb

    dtrain.set_label(y_train)
    dtrain.set_weight(weights)
    dval.set_label(y_val)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )


def _build_monotone_constraints(feature_list: list[str]) -> list[int]:
    """Return +1 for lag columns of warm-weather variables, 0 otherwise.
    Enforces: higher lag value → higher forecast (physics prior).
    """
    WARM_PREFIXES = (
        "temp_c_lag", "heat_index_c_lag", "dewpoint_c_lag",
        "vpd_kpa_lag", "wbgt_stull_c_lag", "rh_lag",
    )
    return [1 if any(col.startswith(p) for p in WARM_PREFIXES) else 0 for col in feature_list]


def _add_monotone_constraints_if_supported(params: dict, feature_list: list[str]) -> dict:
    """Return params with monotone constraints only for LightGBM objectives that support them."""
    constrained = dict(params)
    objective = str(constrained.get("objective", "")).strip().lower()
    if objective == "quantile":
        constrained.pop("monotone_constraints", None)
        return constrained

    constraints = _build_monotone_constraints(feature_list)
    if any(c != 0 for c in constraints):
        constrained["monotone_constraints"] = constraints
    else:
        constrained.pop("monotone_constraints", None)
    return constrained


def _refit_single_booster(
    target: str,
    alpha: float,
    seed: int,
    X_tv: pd.DataFrame,
    y_tv_target: np.ndarray,
    weights_tv: np.ndarray,
    best_params: dict,
    best_iter: int,
    parallel_jobs: int,
):
    """Refit a single booster on train+val_es at best_iter (no early-stop)."""
    import lightgbm as lgb

    params = {
        **_DEFAULT_PARAMS,
        **best_params,
        **_lgbm_device_params(_detect_device(), parallel_jobs=parallel_jobs),
        "alpha": alpha,
        "seed": seed,
    }
    params = _add_monotone_constraints_if_supported(params, list(X_tv.columns))
    dtrain = lgb.Dataset(X_tv, label=y_tv_target, weight=weights_tv)
    return lgb.train(params, dtrain, num_boost_round=best_iter)
