"""XGBoost CUDA Safety Forecaster for HeatShield AI v3.

Combines XGBoost quantile regression with:
- GPU (CUDA) training when available
- DangerGate three-tier classifier
- Tail-aware sample weights (focal weighting for HI >= 40°C)
- Mondrian Conformal Quantile Regression (CQR) calibration
- Threshold sweep on validation calibration set

This backend is designed as the GPU-first champion in the hybrid
champion-challenger training orchestration.
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
from app.ml.forecast.danger_gate import DangerGate
from app.ml.forecast.splitting import (
    apply_feature_medians,
    fit_feature_medians,
    split_xy_4way,
)

logger = logging.getLogger(__name__)

_QUANTILES = {"q05": 0.05, "q50": 0.50, "q95": 0.95, "q97": 0.97}
_SEEDS = [42, 123, 7]


def _detect_xgb_device() -> dict:
    """Return XGBoost device params — CUDA if available, else CPU."""
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


def _make_focal_weights(
    y_hi: np.ndarray,
    *,
    danger_threshold: float = 38.0,
    base_weight: float = 1.0,
    danger_weight: float = 8.0,
    near_weight: float = 4.0,
) -> np.ndarray:
    """Compute focal sample weights with strong up-weighting for danger events.

    Uses 38C as the focal threshold so the model learns to escalate predictions
    earlier — this improves 42C recall because the booster sees more weight
    on the 38-42C transition zone.
    """
    weights = np.full(len(y_hi), base_weight, dtype=float)
    danger_mask = y_hi >= danger_threshold
    near_mask = (y_hi >= danger_threshold - 3.0) & (y_hi < danger_threshold)
    weights[danger_mask] = danger_weight
    weights[near_mask] = near_weight
    return weights


def _danger_weight_for_horizon(horizon_h: int) -> float:
    """Return danger focal weight scaled by horizon difficulty."""
    # Strong up-weighting for danger events; short horizons also need high weight
    # because danger events are rare and critical for public health safety
    mapping = {6: 12.0, 12: 14.0, 24: 16.0, 48: 20.0, 72: 24.0}
    return mapping.get(horizon_h, 16.0)


class XGBoostSafetyForecaster(BaseForecaster):
    """XGBoost quantile forecaster with danger gate and conformal calibration.

    Satisfies BaseForecaster Protocol for v3 registry dispatch.
    """

    backend_name: str = "xgboost_safety"
    target_kind: str = "hi"

    def __init__(
        self,
        n_trials: int = 20,
        random_state: int = 42,
        gate_backend: str = "lightgbm",
    ) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self._gate_backend = gate_backend
        self.feature_list: list[str] = []
        self._station_id: str = ""
        self._horizon_h: int = 24
        self._boosters: dict[str, list] = {q: [] for q in _QUANTILES}
        self._feature_medians: dict = {}
        self._calibrator: MondianCQRCalibrator | None = None
        self._gate: DangerGate | None = None
        self._metadata: dict = {}
        self._best_params: dict = {}
        self._device_used: str = "cpu"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_oversample_train(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        support_threshold: int = 200,
        noise_std: float = 0.3,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Duplicate near-danger rows (HI 39-42C) with small Gaussian noise
        when natural danger support is below threshold. ONLY applied to train split."""
        y_hi = y_train.to_numpy(dtype=float)
        n_danger = int((y_hi >= 42.0).sum())
        if n_danger >= support_threshold:
            return X_train, y_train
        near_mask = (y_hi >= 39.0) & (y_hi < 42.0)
        n_near = int(near_mask.sum())
        if n_near == 0:
            return X_train, y_train
        copies_needed = max(1, (support_threshold - n_danger) // n_near)
        rng = np.random.default_rng(42)
        X_dup = pd.concat([X_train[near_mask]] * copies_needed, ignore_index=True)
        y_dup = pd.concat([y_train[near_mask]] * copies_needed, ignore_index=True)
        for col in X_dup.select_dtypes(include=[np.number]).columns:
            X_dup[col] = X_dup[col] + rng.normal(0, noise_std, len(X_dup))
        y_dup = y_dup + rng.normal(0, noise_std * 0.3, len(y_dup))
        X_out = pd.concat([X_train.reset_index(drop=True), X_dup], ignore_index=True)
        y_out = pd.concat([y_train.reset_index(drop=True), y_dup], ignore_index=True)
        return X_out, y_out

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        *,
        station_id: str,
        horizon_h: int,
    ) -> None:
        """Train XGBoost quantile forecaster with safety gate and calibration."""
        try:
            import xgboost as xgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("pip install xgboost optuna") from exc

        started = time.perf_counter()

        if isinstance(y, pd.DataFrame):
            if "heat_index_c" not in y.columns:
                raise ValueError("XGBoostSafetyForecaster requires scalar y or y['heat_index_c']")
            y = y["heat_index_c"]

        self._station_id = station_id
        self._horizon_h = horizon_h
        self.feature_list = list(X.columns)

        valid = y.notna()
        X, y = X[valid].copy(), y[valid].copy()

        split = split_xy_4way(X, y, horizon_h=horizon_h)
        X_train, y_train = split.X_train, split.y_train
        X_train, y_train = self._maybe_oversample_train(X_train, y_train)
        X_val_es, y_val_es = split.X_val_es, split.y_val_es
        X_val_cal, y_val_cal = split.X_val_cal, split.y_val_cal
        X_test, y_test = split.X_test, split.y_test

        self._feature_medians = fit_feature_medians(X_train)
        X_train = apply_feature_medians(X_train, self._feature_medians)
        X_val_es = apply_feature_medians(X_val_es, self._feature_medians)
        X_val_cal = apply_feature_medians(X_val_cal, self._feature_medians)
        X_test = apply_feature_medians(X_test, self._feature_medians)

        device_params = _detect_xgb_device()
        self._device_used = device_params.get("device", "cpu")
        logger.info(
            "XGBoostSafetyForecaster fit: station=%s h=%d device=%s rows=%d",
            station_id, horizon_h, self._device_used, len(X),
        )

        # Optuna tuning
        X_tune = pd.concat([X_train, X_val_es, X_val_cal], ignore_index=True)
        y_tune = pd.concat([y_train, y_val_es, y_val_cal], ignore_index=True)

        self._best_params = self._tune(X_tune, y_tune, horizon_h=horizon_h)

        # Phase 1: pilot booster with early stopping
        weights_train = _make_focal_weights(
            y_train.to_numpy(dtype=float),
            danger_weight=_danger_weight_for_horizon(horizon_h),
        )

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
        logger.info("XGBoostSafety pilot early-stop: best_iter=%d", best_iter)
        del pilot

        # Phase 2: refit all quantiles on train+val_es at best_iter
        X_tv = pd.concat([X_train, X_val_es], ignore_index=True)
        y_tv = pd.concat([y_train, y_val_es], ignore_index=True)
        weights_tv = _make_focal_weights(
            y_tv.to_numpy(dtype=float),
            danger_weight=_danger_weight_for_horizon(horizon_h),
        )

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
            logger.debug("XGBoostSafety refit: role=%s alpha=%.2f seeds=%d", role, alpha, len(_SEEDS))

        # Danger gate: fit on train+val_es, sweep thresholds on val_cal
        y_hi_tv = y_tv.to_numpy(dtype=float)
        y_hi_val_cal = y_val_cal.to_numpy(dtype=float)
        self._gate = DangerGate(gate_backend=self._gate_backend)
        self._gate.fit(
            X_tv,
            pd.Series(y_hi_tv, index=X_tv.index),
            val_X=X_val_cal,
            val_y_hi=pd.Series(y_hi_val_cal, index=X_val_cal.index),
        )
        self._metadata["gate_warning_threshold"] = float(self._gate.warning_threshold)
        self._metadata["gate_danger_threshold"] = float(self._gate.danger_threshold)

        # Mondrian CQR calibration on val_cal
        bundle_val_cal = self._predict_bundle_raw(X_val_cal)
        station_ids_cal = np.full(len(X_val_cal), station_id)
        angles_cal = np.arctan2(X_val_cal["local_hour_sin"].values, X_val_cal["local_hour_cos"].values)
        local_hours_cal = np.round(angles_cal * 24 / (2 * np.pi)).astype(int) % 24
        danger_tier_cal = self._gate.predict_tier(X_val_cal) if self._gate else np.zeros(len(X_val_cal), dtype=int)

        self._calibrator = MondianCQRCalibrator()
        self._calibrator.fit(
            y_hi_val_cal,
            bundle_val_cal["hi_lower"],
            bundle_val_cal["hi_upper"],
            station_ids=station_ids_cal,
            local_hours=local_hours_cal,
            danger_tiers=danger_tier_cal,
            alpha=0.10,
        )

        # Evaluate on test set
        if len(X_test) > 0:
            bundle_test = self._predict_bundle_calibrated(X_test)
            y_true = y_test.to_numpy(dtype=float)
            mae = float(np.mean(np.abs(bundle_test["hi_mean"] - y_true)))
            danger_mask = y_true >= 42.0
            if danger_mask.sum() > 0:
                danger_recall_42 = float((bundle_test["hi_mean"][danger_mask] >= 42.0).mean())
            else:
                danger_recall_42 = float("nan")
            danger_mask_40 = y_true >= 40.0
            if danger_mask_40.sum() > 0:
                danger_recall_40 = float((bundle_test["hi_mean"][danger_mask_40] >= 40.0).mean())
            else:
                danger_recall_40 = float("nan")
            coverage = float(((y_true >= bundle_test["hi_lower"]) & (y_true <= bundle_test["hi_upper"])).mean())
            self._metadata.update({
                "mae": round(mae, 3),
                "danger_recall": round(danger_recall_42, 3) if not np.isnan(danger_recall_42) else None,
                "danger_recall_40": round(danger_recall_40, 3) if not np.isnan(danger_recall_40) else None,
                "pi_coverage_90": round(coverage, 3),
                "n_test": len(X_test),
                "n_train": len(X_train),
                "split_metadata": split.metadata,
                "device": self._device_used,
            })
            logger.info("XGBoostSafety test: MAE=%.3f DangerR42=%.3f DangerR40=%.3f PI=%.3f",
                        mae, danger_recall_42, danger_recall_40, coverage)

        self._metadata["train_seconds"] = float(time.perf_counter() - started)
        logger.info(
            "XGBoostSafetyForecaster fit done: station=%s h=%d elapsed=%.1fs device=%s",
            station_id, horizon_h, self._metadata["train_seconds"], self._device_used,
        )

    def _tune(
        self,
        X_tune: pd.DataFrame,
        y_tune: pd.Series,
        *,
        horizon_h: int,
    ) -> dict:
        """Optuna hyperparameter tuning for XGBoost quantile regression."""
        import optuna
        import xgboost as xgb

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            }

            device_params = _detect_xgb_device()
            full_params = {
                **params,
                **device_params,
                "objective": "reg:squarederror",
                "seed": self.random_state,
            }

            weights = _make_focal_weights(
                y_tune.to_numpy(dtype=float),
                danger_weight=_danger_weight_for_horizon(horizon_h),
            )
            dtrain = xgb.DMatrix(X_tune, label=y_tune.to_numpy(dtype=float), weight=weights)

            # Simple 5-fold time-series CV
            n = len(X_tune)
            fold_size = n // 5
            maes = []
            for fold in range(5):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < 4 else n
                train_idx = list(range(0, val_start)) + list(range(val_end, n))
                val_idx = list(range(val_start, val_end))
                if len(train_idx) < 100 or len(val_idx) < 10:
                    continue
                dtr = xgb.DMatrix(X_tune.iloc[train_idx], label=y_tune.iloc[train_idx].to_numpy(dtype=float))
                dv = xgb.DMatrix(X_tune.iloc[val_idx], label=y_tune.iloc[val_idx].to_numpy(dtype=float))
                bst = xgb.train(full_params, dtr, num_boost_round=500, evals=[(dv, "val")],
                                early_stopping_rounds=30, verbose_eval=False)
                preds = bst.predict(dv)
                maes.append(float(np.mean(np.abs(preds - y_tune.iloc[val_idx].to_numpy(dtype=float)))))

            return float(np.mean(maes)) if maes else float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(
            "XGBoostSafety Optuna best: MAE=%.4f params=%s",
            study.best_value, study.best_params,
        )
        return dict(study.best_params)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_quantile(self, X: pd.DataFrame, role: str) -> np.ndarray:
        """Predict a single quantile by averaging ensemble seeds."""
        import xgboost as xgb
        X_aligned = self._align(X)
        dmat = xgb.DMatrix(X_aligned)
        preds = np.stack([b.predict(dmat) for b in self._boosters[role]], axis=0)
        return preds.mean(axis=0)

    def _predict_bundle_raw(self, X: pd.DataFrame) -> dict:
        """Raw (uncalibrated) prediction dict."""
        hi_mean = self._predict_quantile(X, "q50")
        hi_lower = self._predict_quantile(X, "q05")
        hi_upper = self._predict_quantile(X, "q95")
        hi_q97 = self._predict_quantile(X, "q97")

        # Apply bias correction
        bias = self._metadata.get("bias_correction", 0.0)
        if bias != 0.0:
            hi_mean = hi_mean - bias
            hi_lower = hi_lower - bias
            hi_upper = hi_upper - bias
            hi_q97 = hi_q97 - bias

        danger_proba = self._gate.predict_proba(self._align(X)) if self._gate else None
        return {
            "hi_mean": hi_mean,
            "hi_lower": hi_lower,
            "hi_upper": hi_upper,
            "hi_q97": hi_q97,
            "danger_proba": danger_proba,
        }

    def _predict_bundle_calibrated(self, X: pd.DataFrame) -> dict:
        """Calibrated prediction dict with Mondrian CQR."""
        X_aligned = self._align(X)
        raw = self._predict_bundle_raw(X_aligned)
        hi_mean = raw["hi_mean"]
        hi_lower = raw["hi_lower"]
        hi_upper = raw["hi_upper"]
        hi_q97 = raw["hi_q97"]
        danger_proba = raw["danger_proba"]

        danger_tier = self._gate.predict_tier(X_aligned) if self._gate else np.zeros(len(X_aligned), dtype=int)

        # Mondrian CQR calibration
        if self._calibrator is not None:
            station_ids = np.full(len(X_aligned), self._station_id)
            angles = np.arctan2(X_aligned["local_hour_sin"].values, X_aligned["local_hour_cos"].values)
            local_hours = np.round(angles * 24 / (2 * np.pi)).astype(int) % 24
            hi_lower, hi_upper = self._calibrator.adjust(
                hi_lower, hi_upper,
                station_ids=station_ids,
                local_hours=local_hours,
                danger_tiers=danger_tier,
            )

        # Tier-aware gate override
        warning_mask = danger_tier == 1
        danger_mask = danger_tier == 2
        if warning_mask.any():
            hi_mean[warning_mask] = np.maximum(
                hi_mean[warning_mask],
                0.8 * hi_mean[warning_mask] + 0.2 * hi_q97[warning_mask],
            )
        if danger_mask.any():
            hi_mean[danger_mask] = np.maximum(hi_mean[danger_mask], hi_q97[danger_mask])
            hi_upper[danger_mask] = np.maximum(hi_upper[danger_mask], hi_q97[danger_mask])

        return {
            "hi_mean": hi_mean,
            "hi_lower": hi_lower,
            "hi_upper": hi_upper,
            "hi_q97": hi_q97,
            "danger_proba": danger_proba,
        }

    def predict_with_pi(self, X: pd.DataFrame, alpha: float = 0.10) -> PredictionBundle:
        """Return calibrated PredictionBundle for rows in X."""
        bundle = self._predict_bundle_calibrated(X)
        return PredictionBundle(
            hi_mean=bundle["hi_mean"],
            hi_lower=bundle["hi_lower"],
            hi_upper=bundle["hi_upper"],
            danger_proba=bundle["danger_proba"],
        )

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align X columns to training feature list, fill missing with medians."""
        out = X.reindex(columns=self.feature_list)
        for col, med in self._feature_medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)
        return out.fillna(0.0)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        """Persist all boosters + calibrator + gate + metadata."""
        import xgboost as xgb
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save boosters per quantile/seed
        for role, boosters in self._boosters.items():
            for i, bst in enumerate(boosters):
                fname = f"{role}_s{_SEEDS[i]}.json"
                bst.save_model(str(dir_path / fname))

        # Save metadata
        meta = {
            "backend_name": self.backend_name,
            "target_kind": self.target_kind,
            "feature_list": self.feature_list,
            "feature_medians": self._feature_medians,
            "station_id": self._station_id,
            "horizon_h": self._horizon_h,
            "best_params": self._best_params,
            **self._metadata,
        }
        (dir_path / "bundle.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        # Save calibrator
        if self._calibrator is not None:
            self._calibrator.save(dir_path / "calibrator.json")

        # Save gate
        if self._gate is not None:
            self._gate.save(dir_path / "gate")

        logger.info("XGBoostSafetyForecaster saved to %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "XGBoostSafetyForecaster":
        """Load persisted forecaster."""
        import xgboost as xgb
        dir_path = Path(dir_path)

        bundle = json.loads((dir_path / "bundle.json").read_text(encoding="utf-8"))
        obj = cls(
            n_trials=bundle.get("n_trials", 20),
            random_state=bundle.get("random_state", 42),
            gate_backend=bundle.get("gate_backend", "lightgbm"),
        )
        obj.backend_name = bundle.get("backend_name", cls.backend_name)
        obj.target_kind = bundle.get("target_kind", cls.target_kind)
        obj.feature_list = bundle["feature_list"]
        obj._feature_medians = bundle["feature_medians"]
        obj._station_id = bundle["station_id"]
        obj._horizon_h = bundle["horizon_h"]
        obj._best_params = bundle.get("best_params", {})
        obj._metadata = {k: v for k, v in bundle.items()
                         if k not in ("backend_name", "target_kind", "feature_list",
                                      "feature_medians", "station_id", "horizon_h",
                                      "best_params", "n_trials", "random_state", "gate_backend")}

        for role in _QUANTILES:
            seed_boosters = []
            for seed in _SEEDS:
                fname = f"{role}_s{seed}.json"
                bst = xgb.Booster()
                bst.load_model(str(dir_path / fname))
                seed_boosters.append(bst)
            obj._boosters[role] = seed_boosters

        cal_path = dir_path / "calibrator.json"
        if cal_path.exists():
            obj._calibrator = MondianCQRCalibrator.load(cal_path)

        gate_path = dir_path / "gate"
        if gate_path.exists():
            obj._gate = DangerGate.load(gate_path)

        return obj
