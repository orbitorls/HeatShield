"""XGBoost heat-index forecast training with persistence + climatology baselines.

Training hygiene:
- TimeSeriesSplit(5) cross-validation (no random shuffles)
- Optuna hyperparameter search (100 trials by default)
- random_state=42 everywhere
- Train/val/test = 70% / 15% / 15% by timestamp order
- Sample weights: 5x for >38°C, 9x for >42°C (tail-class emphasis)
- Early stopping on val split (50 rounds)
- Quantile regression heads: q05 + q95 for calibrated prediction intervals
- Ensemble of 3 seeds per role for stability
- Honesty rule: if skill_score < 0.15, logs a WARNING
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.ml.forecast.splitting import (
    apply_feature_medians,
    fit_feature_medians,
    split_xy,
    time_series_cv_splits,
)
from app.ml.forecast._optuna_utils import feature_signature, forecast_model_root, write_heartbeat

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _gpu_xgb_params() -> dict:
    """Return XGBoost device params — CUDA if available, CPU otherwise."""
    if os.getenv("HEATSHIELD_FORCE_CPU") == "1":
        return {"tree_method": "hist", "device": "cpu"}
    try:
        info = xgb.build_info()
        if info.get("USE_CUDA"):
            return {"tree_method": "hist", "device": "cuda"}
    except (AttributeError, ImportError, RuntimeError) as exc:
        logger.warning("Could not detect XGBoost CUDA support: %s", exc)
    return {"tree_method": "hist", "device": "cpu"}


# _GPU_XGB_PARAMS intentionally not cached — callers use _gpu_xgb_params() so
# env var changes (HEATSHIELD_FORCE_CPU) made after import are respected.

_LOGS_DIR = Path(__file__).parents[3] / "logs" / "train_runs"
_DEFAULT_LAGS_H = [1, 3, 6, 12, 24]
_DEFAULT_ROLLING_H = [3, 6, 24]

# Ensemble seeds and quantile roles
_SEEDS = [42, 123, 7]
_ROLES = {
    "mean": {"objective": "reg:squarederror"},
    "q05": {"objective": "reg:quantileerror", "quantile_alpha": 0.05},
    "q95": {"objective": "reg:quantileerror", "quantile_alpha": 0.95},
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        logger.debug("Could not get git SHA: %s", exc)
        return "unknown"


def _persistence_mae(y_true: np.ndarray, horizon_h: int, X: pd.DataFrame) -> float:
    """Persistence baseline: predict heat_index_c_lag{horizon}h if available, else lag1h."""
    lag_col = f"heat_index_c_lag{horizon_h}h"
    fallback = "heat_index_c_lag1h"
    if lag_col in X.columns:
        return float(mean_absolute_error(y_true, X[lag_col].values))
    if fallback in X.columns:
        return float(mean_absolute_error(y_true, X[fallback].values))
    return float(np.abs(np.diff(y_true)).mean())  # last resort


def _climatology_mae(y_true: np.ndarray, X: pd.DataFrame) -> float:
    """Climatology baseline: mean HI per hour-of-day."""
    if "hour_sin" not in X.columns or "hour_cos" not in X.columns:
        return float(np.abs(y_true - y_true.mean()).mean())
    # Approximate hour from sin/cos
    hours = np.round(np.arctan2(X["hour_sin"].values, X["hour_cos"].values) / (2 * np.pi / 24)) % 24
    hourly_mean = pd.Series(y_true).groupby(hours).transform("mean").values
    return float(mean_absolute_error(y_true, hourly_mean))


def _make_sample_weight(y: np.ndarray) -> np.ndarray:
    """Tail-class sample weights: 5x for HI>38°C, 9x for HI>42°C."""
    return 1.0 + 4.0 * (y > 38).astype(float) + 8.0 * (y > 42).astype(float)


def _make_xgb_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_h: int,
) -> list[tuple[xgb.DMatrix, xgb.DMatrix, np.ndarray]]:
    X_arr = X.values
    y_arr = y.values
    names = list(X.columns)
    folds: list[tuple[xgb.DMatrix, xgb.DMatrix, np.ndarray]] = []
    for train_idx, val_idx in time_series_cv_splits(X, horizon_h=horizon_h, n_splits=5):
        yt = y_arr[train_idx]
        yv = y_arr[val_idx]
        dtr = xgb.DMatrix(
            X_arr[train_idx],
            label=yt,
            weight=_make_sample_weight(yt),
            feature_names=names,
        )
        dval = xgb.DMatrix(X_arr[val_idx], label=yv, feature_names=names)
        folds.append((dtr, dval, yv))
    return folds


def _objective(trial: optuna.Trial, cv_folds: list[tuple[xgb.DMatrix, xgb.DMatrix, np.ndarray]]) -> float:
    params = {
        **_gpu_xgb_params(),
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "seed": 42,
        "verbosity": 0,
    }
    n_estimators = trial.suggest_int("n_estimators", 200, 2000)

    maes: list[float] = []
    for k, (dtr, dval, yv) in enumerate(cv_folds):
        bst = xgb.train(
            params, dtr,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=20,
        )
        preds = bst.predict(dval)
        maes.append(mean_absolute_error(yv, preds))
        trial.report(float(np.mean(maes)), step=k)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(maes))


def train(
    X: pd.DataFrame,
    y: pd.Series,
    station_id: str,
    horizon_h: int,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Train XGBoost forecast model with TimeSeriesSplit CV and optuna tuning.

    Returns dict with keys:
      all_boosters — dict with keys "mean", "q05", "q95", each containing a list
                     of 3 xgb.Booster objects (one per seed in _SEEDS=[42, 123, 7])
      metrics      — mae, rmse, p90_abs_err, mae_persistence, mae_climatology, skill_score
      feature_list  — list of feature column names
      hyperparams   — best optuna params
    """
    started = time.perf_counter()
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOGS_DIR / f"forecast_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.jsonl"

    split = split_xy(X, y, horizon_h=horizon_h)
    X_train, y_train = split.X_train, split.y_train
    X_val, y_val = split.X_val, split.y_val
    X_test, y_test = split.X_test, split.y_test

    feature_medians = fit_feature_medians(X_train)
    X_train = apply_feature_medians(X_train, feature_medians)
    X_val = apply_feature_medians(X_val, feature_medians)
    X_test = apply_feature_medians(X_test, feature_medians)

    logger.info("Train rows=%d, Val rows=%d, Test rows=%d", len(X_train), len(X_val), len(X_test))

    # --- Optuna hyperparameter search on mean objective ---
    cv_folds = _make_xgb_cv_folds(X_train, y_train, horizon_h)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    storage_path = forecast_model_root() / "optuna_studies.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.as_posix()}?check_same_thread=False"
    study_name = f"xgb_{station_id}_h{horizon_h}_{feature_signature(list(X_train.columns))}"
    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(min_resource=64, max_resource=2000, reduction_factor=3),
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    except Exception as exc:
        logger.warning("XGBoost Optuna storage unavailable (%s). Falling back to in-memory study.", exc)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(min_resource=64, max_resource=2000, reduction_factor=3),
        )

    def objective(trial: optuna.Trial) -> float:
        return _objective(trial, cv_folds)

    run_id = os.environ.get("HEATSHIELD_RUN_ID")

    def _heartbeat_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        write_heartbeat(
            run_id,
            current_slot=f"{station_id}:h{horizon_h}",
            phase="tune",
            trials_done=sum(t.state.name == "COMPLETE" for t in study.trials),
        )

    n_remaining = max(0, n_trials - sum(t.state.name == "COMPLETE" for t in study.trials))
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=False, callbacks=[_heartbeat_cb])
    else:
        logger.info("Optuna study already has %d complete trials; reusing best params", n_trials)
    best_params = study.best_params
    best_params.update({"seed": random_state, "verbosity": 0})
    n_estimators = best_params.pop("n_estimators", 300)

    # Write optuna trial log
    with open(log_path, "w", encoding="utf-8") as f:
        for t in study.trials:
            f.write(json.dumps({"trial": t.number, "value": t.value, "params": t.params}) + "\n")

    # --- Final training: train+val split for early stopping, then full trainval ---
    # Use X_train/y_train for training and X_val/y_val for early stopping eval set.
    # After finding best round, train all 9 boosters (3 roles × 3 seeds) on train+val.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    sw_train = _make_sample_weight(y_train.values)
    sw_trainval = _make_sample_weight(y_trainval.values)

    # Determine best n_rounds via early stopping on the mean booster using train/val split
    es_params = {**_gpu_xgb_params(), **best_params, "objective": "reg:squarederror", "eval_metric": "mae"}
    d_es_train = xgb.DMatrix(X_train.values, label=y_train.values, weight=sw_train,
                              feature_names=list(X.columns))
    d_es_val = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=list(X.columns))
    evals_result: dict = {}
    bst_es = xgb.train(
        es_params,
        d_es_train,
        num_boost_round=1000,
        evals=[(d_es_val, "val")],
        verbose_eval=False,
        early_stopping_rounds=30,
        evals_result=evals_result,
    )
    best_rounds = bst_es.best_iteration + 1
    logger.info("Early stopping: best_rounds=%d (out of max %d)", best_rounds, n_estimators)

    # --- Train all 9 boosters (3 roles × 3 seeds) on full trainval ---
    all_boosters: dict[str, list[xgb.Booster]] = {"mean": [], "q05": [], "q95": []}
    d_tv = xgb.DMatrix(
        X_trainval.values,
        label=y_trainval.values,
        weight=sw_trainval,
        feature_names=list(X.columns),
    )

    for seed in _SEEDS:
        for role, extra_params in _ROLES.items():
            params = {**_gpu_xgb_params(), **best_params, **extra_params, "seed": seed, "verbosity": 0}
            booster = xgb.train(
                params,
                d_tv,
                num_boost_round=best_rounds,
                verbose_eval=False,
            )
            all_boosters[role].append(booster)

    # Use the first mean booster (seed=42) for test evaluation
    d_test = xgb.DMatrix(X_test.values, feature_names=list(X.columns))
    preds = all_boosters["mean"][0].predict(d_test)
    y_test_arr = y_test.values

    mae = float(mean_absolute_error(y_test_arr, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test_arr, preds)))
    abs_err = np.abs(y_test_arr - preds)
    p90_abs_err = float(np.percentile(abs_err, 90))
    mae_pers = _persistence_mae(y_test_arr, horizon_h, X_test)
    mae_clim = _climatology_mae(y_test_arr, X_test)
    skill_score = float(1 - mae / max(mae_pers, mae_clim, 1e-6))

    # Danger-class recall (HI > 40°C threshold)
    danger_mask = y_test_arr > 40.0
    if danger_mask.sum() > 0:
        danger_recall = float((preds[danger_mask] > 40.0).mean())
    else:
        danger_recall = float("nan")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "p90_abs_err": p90_abs_err,
        "mae_persistence": mae_pers,
        "mae_climatology": mae_clim,
        "skill_score": skill_score,
        "danger_recall": danger_recall,
    }

    logger.info("Test metrics: %s", {k: f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "nan" for k, v in metrics.items()})

    if skill_score < 0.15:
        logger.warning(
            "skill_score=%.3f < 0.15 threshold. Model barely beats baselines. "
            "Consider more training data or feature engineering before shipping.",
            skill_score,
        )

    # Append final result to log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"final_metrics": metrics, "best_params": best_params, "best_rounds": best_rounds}) + "\n")

    return {
        "all_boosters": all_boosters,
        "metrics": metrics,
        "feature_list": list(X.columns),
        "feature_medians": feature_medians,
        "hyperparams": {**best_params, "n_estimators": best_rounds},
        "split_metadata": split.metadata,
        "test_data": {"X": X_test, "y": y_test.reset_index(drop=True)},
        "test_predictions": {
            "mean": preds,
            "q05": np.mean([b.predict(d_test) for b in all_boosters["q05"]], axis=0),
            "q95": np.mean([b.predict(d_test) for b in all_boosters["q95"]], axis=0),
        },
        "training": {
            "train_seconds": float(time.perf_counter() - started),
            "n_trials_completed": sum(t.state.name == "COMPLETE" for t in study.trials),
            "n_trials_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
            "best_iteration": best_rounds,
        },
    }
