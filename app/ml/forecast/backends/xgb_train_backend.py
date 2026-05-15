"""XGBoost quantile forecaster for HeatShield AI v3 — FULL TRAINING backend.

Predicts heat_index_c directly via XGBoost quantile regression.
Uses Optuna hyperparameter search, ensemble of seeds, and conformal calibration.
Designed to be a drop-in replacement for LGBMForecaster with faster training.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.ml.forecast.base import BaseForecaster, PredictionBundle
from app.ml.forecast.conformal import MondianCQRCalibrator
from app.ml.forecast.splitting import apply_feature_medians, fit_feature_medians, split_xy_4way
from app.ml.forecast._optuna_utils import feature_signature, forecast_model_root, write_heartbeat

logger = logging.getLogger(__name__)

_QUANTILES = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
_SEEDS = [42, 123, 7]


def _gpu_xgb_params() -> dict:
    """Return XGBoost device params — CUDA if available, CPU otherwise."""
    if os.getenv("HEATSHIELD_FORCE_CPU") == "1":
        return {"tree_method": "hist", "device": "cpu"}
    try:
        import xgboost as xgb
        info = xgb.build_info()
        if info.get("USE_CUDA"):
            return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        pass
    return {"tree_method": "hist", "device": "cpu"}


def _make_sample_weight(y: np.ndarray) -> np.ndarray:
    """Tail-class sample weights: 5x for HI>38°C, 9x for HI>42°C."""
    return 1.0 + 4.0 * (y > 38).astype(float) + 8.0 * (y > 42).astype(float)


class XGBoostTrainForecaster:
    """XGBoost quantile forecaster with full training support."""

    backend_name: str = "xgboost"
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
        self._boosters: dict[str, list] = {q: [] for q in _QUANTILES}
        self._feature_medians: dict = {}
        self._calibrator: MondianCQRCalibrator | None = None
        self._metadata: dict = {}
        self._best_params: dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        *,
        station_id: str,
        horizon_h: int,
    ) -> None:
        """Train XGBoost quantile forecaster."""
        try:
            import xgboost as xgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("pip install xgboost optuna") from exc

        started = time.perf_counter()
        
        # Handle y input
        if isinstance(y, pd.DataFrame):
            if "heat_index_c" not in y.columns:
                raise ValueError("XGBoostTrainForecaster requires scalar y or y['heat_index_c']")
            y = y["heat_index_c"]

        self._station_id = station_id
        self._horizon_h = horizon_h
        self.feature_list = list(X.columns)

        valid = y.notna()
        X, y = X[valid].copy(), y[valid].copy()

        split = split_xy_4way(X, y, horizon_h=horizon_h)
        X_train, y_train = split.X_train, split.y_train
        X_val_es, y_val_es = split.X_val_es, split.y_val_es
        X_val_cal, y_val_cal = split.X_val_cal, split.y_val_cal
        X_test, y_test = split.X_test, split.y_test

        self._feature_medians = fit_feature_medians(X_train)
        X_train = apply_feature_medians(X_train, self._feature_medians)
        X_val_es = apply_feature_medians(X_val_es, self._feature_medians)
        X_val_cal = apply_feature_medians(X_val_cal, self._feature_medians)
        X_test = apply_feature_medians(X_test, self._feature_medians)

        device_params = _gpu_xgb_params()
        logger.info(
            "XGBoostTrainForecaster fit: station=%s h=%d device=%s rows=%d",
            station_id, horizon_h, device_params.get("device", "cpu"), len(X),
        )

        # Optuna tuning
        X_tune = pd.concat([X_train, X_val_es, X_val_cal], ignore_index=True)
        y_tune = pd.concat([y_train, y_val_es, y_val_cal], ignore_index=True)
        
        self._best_params = self._tune(X_tune, y_tune, horizon_h=horizon_h)
        
        # Phase 1: pilot booster with early stopping
        weights_train = _make_sample_weight(y_train.to_numpy(dtype=float))
        
        pilot_params = {
            **self._best_params,
            **device_params,
            "objective": "reg:squarederror",
            "seed": self.random_state,
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train.to_numpy(dtype=float), weight=weights_train)
        dval = xgb.DMatrix(X_val_es, label=y_val_es.to_numpy(dtype=float))
        
        pilot = xgb.train(
            pilot_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        best_iter = max(pilot.best_iteration, 1)
        logger.info("XGBoost pilot early-stop: best_iter=%d", best_iter)
        del pilot

        # Phase 2: refit all quantiles on train+val_es at best_iter
        X_tv = pd.concat([X_train, X_val_es], ignore_index=True)
        y_tv = pd.concat([y_train, y_val_es], ignore_index=True)
        weights_tv = _make_sample_weight(y_tv.to_numpy(dtype=float))
        
        for role, alpha in _QUANTILES.items():
            seed_boosters = []
            for seed in _SEEDS:
                params = {
                    **self._best_params,
                    **device_params,
                    "objective": "reg:quantileerror",
                    "quantile_alpha": alpha,
                    "seed": seed,
                }
                dtrain_tv = xgb.DMatrix(X_tv, label=y_tv.to_numpy(dtype=float), weight=weights_tv)
                bst = xgb.train(params, dtrain_tv, num_boost_round=best_iter, verbose_eval=False)
                seed_boosters.append(bst)
            self._boosters[role] = seed_boosters
            logger.debug("XGBoost refit: role=%s alpha=%.2f seeds=%d", role, alpha, len(_SEEDS))

        # Mondrian CQR on val_cal
        bundle_val_cal = self.predict_with_pi(X_val_cal)
        station_ids_cal = np.full(len(X_val_cal), station_id)
        angles_cal = np.arctan2(X_val_cal["local_hour_sin"].values, X_val_cal["local_hour_cos"].values)
        local_hours_cal = np.round(angles_cal * 24 / (2 * np.pi)).astype(int) % 24
        danger_tier_cal = np.zeros(len(X_val_cal), dtype=int)

        self._calibrator = MondianCQRCalibrator()
        self._calibrator.fit(
            y_val_cal.to_numpy(dtype=float),
            bundle_val_cal.hi_lower,
            bundle_val_cal.hi_upper,
            station_ids=station_ids_cal,
            local_hours=local_hours_cal,
            danger_tiers=danger_tier_cal,
            alpha=0.10,
        )

        # Evaluate on test set
        if len(X_test) > 0:
            bundle_test = self._predict_bundle_calibrated(X_test)
            y_true = y_test.to_numpy(dtype=float)
            mae = float(np.mean(np.abs(bundle_test.hi_mean - y_true)))
            coverage = float(((y_true >= bundle_test.hi_lower) & (y_true <= bundle_test.hi_upper)).mean())
            self._metadata.update({
                "mae": round(mae, 3),
                "pi_coverage_90": round(coverage, 3),
                "n_test": len(X_test),
                "n_train": len(X_train),
                "split_metadata": split.metadata,
            })
            logger.info("XGBoost test: MAE=%.3f coverage_90=%.3f", mae, coverage)

        # Explicit GC + optional GPU cache clear after heavy training
        gc.collect()
        if device_params.get("device") == "cuda":
            import sys
            if sys.platform == "linux":
                try:
                    import ctypes
                    lib = ctypes.CDLL("libcuda.so.1")
                    lib.cuCtxSynchronize()
                except (OSError, AttributeError):
                    pass

        logger.info(
            "XGBoostTrainForecaster fit done: station=%s h=%d elapsed=%.1fs",
            station_id, horizon_h, time.perf_counter() - started,
        )

    def _tune(self, X_tune: pd.DataFrame, y_tune: pd.Series, *, horizon_h: int) -> dict:
        """Optuna hyperparameter tuning for XGBoost."""
        import optuna
        
        device_params = _gpu_xgb_params()
        
        def objective(trial):
            import xgboost as xgb
            params = {
                **device_params,
                "objective": "reg:squarederror",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                "seed": self.random_state,
            }
            
            # TimeSeriesSplit CV
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=2)
            maes = []
            
            for train_idx, val_idx in tscv.split(X_tune):
                X_tr, X_val = X_tune.iloc[train_idx], X_tune.iloc[val_idx]
                y_tr, y_val = y_tune.iloc[train_idx], y_tune.iloc[val_idx]
                
                dtrain = xgb.DMatrix(X_tr, label=y_tr.to_numpy(dtype=float))
                dval = xgb.DMatrix(X_val, label=y_val.to_numpy(dtype=float))
                
                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=_trial_rounds,
                    evals=[(dval, "val")],
                    early_stopping_rounds=20,
                    verbose_eval=False,
                )
                
                preds = bst.predict(dval)
                mae = np.mean(np.abs(preds - y_val.to_numpy(dtype=float)))
                maes.append(mae)
            
            return np.mean(maes)

        # Use a unique study name per (station, horizon, feature signature)
        sig = feature_signature(X_tune.columns.tolist(), extra=f"h{horizon_h}")
        study_name = f"xgb_{self._station_id}_h{horizon_h}_{sig}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state, n_startup_trials=5),
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
        )
        
        _trial_rounds = int(os.environ.get("XGB_TUNE_ROUNDS", "25"))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        best = study.best_params
        logger.info(
            "XGBoost Optuna best: MAE=%.4f params=%s",
            study.best_value, json.dumps(best, indent=2),
        )
        return best

    def predict_with_pi(self, X: pd.DataFrame, alpha: float = 0.10) -> PredictionBundle:
        """Predict with prediction intervals (uncalibrated)."""
        import xgboost as xgb
        if not self._boosters:
            raise RuntimeError("Forecaster not fitted — call fit() first")
        
        X = apply_feature_medians(X, self._feature_medians)
        
        mean_preds = np.stack([b.predict(xgb.DMatrix(X)) for b in self._boosters["q50"]], axis=0).mean(axis=0)
        lower_preds = np.stack([b.predict(xgb.DMatrix(X)) for b in self._boosters["q05"]], axis=0).mean(axis=0)
        upper_preds = np.stack([b.predict(xgb.DMatrix(X)) for b in self._boosters["q95"]], axis=0).mean(axis=0)
        
        return PredictionBundle(
            hi_mean=mean_preds,
            hi_lower=lower_preds,
            hi_upper=upper_preds,
        )

    def _predict_bundle_calibrated(self, X: pd.DataFrame) -> PredictionBundle:
        """Predict with Mondrian CQR calibration applied."""
        bundle = self.predict_with_pi(X)
        if self._calibrator is None:
            return bundle
        
        station_ids = np.full(len(X), self._station_id)
        angles = np.arctan2(X["local_hour_sin"].values, X["local_hour_cos"].values)
        local_hours = np.round(angles * 24 / (2 * np.pi)).astype(int) % 24
        danger_tiers = np.zeros(len(X), dtype=int)
        
        lo, hi = self._calibrator.adjust(
            bundle.hi_lower,
            bundle.hi_upper,
            station_ids=station_ids,
            local_hours=local_hours,
            danger_tiers=danger_tiers,
        )
        return PredictionBundle(hi_mean=bundle.hi_mean, hi_lower=lo, hi_upper=hi)

    def save(self, dir_path: Path) -> None:
        """Save booster artifacts + metadata."""
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for role, boosters in self._boosters.items():
            for i, bst in enumerate(boosters):
                bst.save_model(str(dir_path / f"{role}_s{_SEEDS[i]}.json"))
        
        meta = {
            "backend_name": self.backend_name,
            "station_id": self._station_id,
            "horizon_h": self._horizon_h,
            "feature_list": self.feature_list,
            "feature_medians": self._feature_medians,
            "best_params": self._best_params,
            "metadata": self._metadata,
        }
        (dir_path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Path) -> "XGBoostTrainForecaster":
        """Load saved forecaster."""
        import xgboost as xgb
        
        obj = cls()
        meta = json.loads((dir_path / "metadata.json").read_text(encoding="utf-8"))
        
        obj._station_id = meta["station_id"]
        obj._horizon_h = meta["horizon_h"]
        obj.feature_list = meta["feature_list"]
        obj._feature_medians = meta.get("feature_medians", {})
        obj._best_params = meta.get("best_params", {})
        obj._metadata = meta.get("metadata", {})
        
        for role in _QUANTILES:
            seed_boosters = []
            for seed in _SEEDS:
                model_path = dir_path / f"{role}_s{seed}.json"
                if model_path.exists():
                    bst = xgb.Booster()
                    bst.load_model(str(model_path))
                    seed_boosters.append(bst)
            obj._boosters[role] = seed_boosters
        
        return obj
