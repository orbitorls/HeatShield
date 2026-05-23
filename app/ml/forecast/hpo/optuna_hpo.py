"""Optuna TPE hyperparameter optimisation for HeatShield AI LightGBM backends.

Provides a ready-made Optuna objective for LightGBM quantile forecasters and a
convenience wrapper that creates the study with HyperbandPruner for 2–3× faster
wall-clock convergence (Li et al., 2018).
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def lgbm_optuna_objective(
    trial: "optuna.Trial",
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
) -> float:
    """Optuna objective for a LightGBM MAE regressor.

    Returns validation MAE (minimise).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required for lgbm_optuna_objective")

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    try:
        from optuna.integration import LightGBMPruningCallback
        callbacks.append(LightGBMPruningCallback(trial, "valid_0-mae"))
    except ImportError:
        pass

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return float(np.mean(np.abs(y_val - y_pred)))


def optimize_lgbm(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    n_trials: int = 50,
    use_hyperband: bool = True,
) -> Dict[str, Any]:
    """Run Optuna TPE optimisation (+ optional HyperbandPruner) for LightGBM.

    Args:
        X_train, y_train: Training split.
        X_val, y_val: Validation split.
        n_trials: Number of trials.
        use_hyperband: Use HyperbandPruner for early stopping (default True).

    Returns:
        best_params dict ready to pass to lgb.train.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required — pip install optuna")

    pruner = (
        optuna.pruners.HyperbandPruner(min_resource=20, max_resource=200, reduction_factor=3)
        if use_hyperband
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        lambda trial: lgbm_optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params
