"""LGBMDirectHIForecaster — direct HI quantile forecaster"""
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
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from app.ml.forecast.base import BaseForecaster, PredictionBundle
from app.ml.forecast.conformal import EnbPICalibrator, MondianCQRCalibrator
from app.ml.forecast.hpo.optuna_utils import feature_signature, forecast_model_root, write_heartbeat
from app.ml.forecast.splitting import apply_feature_medians, fit_feature_medians, split_xy_4way
from app.ml.forecast.backends.lgbm.utils import (
    _QUANTILES, _TARGETS, _SEEDS, _SEEDS_TAIL,
    _DEFAULT_PARAMS,
    _optuna_storage_path,
    _detect_device, _lgbm_device_params, _sanitize_lgbm_params_for_device,
    _remaining_trials,
    _add_monotone_constraints_if_supported,
)

logger = logging.getLogger(__name__)


class LGBMDirectHIForecaster:
    """Direct heat-index LightGBM quantile forecaster."""

    backend_name: str = "lightgbm_hi_quantile"
    target_kind: Literal["hi", "th"] = "hi"

    def __init__(self, n_trials: int = 40, random_state: int = 42, gate_backend: str = "lightgbm") -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self._gate_backend = gate_backend  # accepted for API symmetry with LGBMForecaster; not used in fit
        self.feature_list: list[str] = []
        self._station_id: str = ""
        self._horizon_h: int = 24
        # Change 3g: added q97 slot
        self._boosters: dict[str, list] = {"q05": [], "q50": [], "q95": [], "q97": []}
        self._best_params: dict = {}
        self._feature_medians: dict = {}
        self._metadata: dict = {}
        self._calibrator: EnbPICalibrator | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        *,
        station_id: str,
        horizon_h: int,
    ) -> None:
        try:
            import lightgbm as lgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("pip install lightgbm optuna") from exc

        started = time.perf_counter()
        if isinstance(y, pd.DataFrame):
            if "heat_index_c" not in y.columns:
                raise ValueError("Direct HI training requires a Series target or y['heat_index_c']")
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

        from app.ml.forecast.train import _make_sample_weight
        weights = _make_sample_weight(y_train.to_numpy(dtype=float))
        self._best_params = self._tune(X_train, y_train, weights, X_val_es, y_val_es)

        # Phase 1: pilot with early-stop on val_es → best_iter
        import lightgbm as lgb
        pilot_params = {
            **_DEFAULT_PARAMS,
            **self._best_params,
            **_lgbm_device_params(_detect_device()),
            "alpha": 0.50,
            "seed": self.random_state,
        }
        pilot_booster = lgb.train(
            pilot_params,
            lgb.Dataset(X_train, label=y_train.to_numpy(dtype=float), weight=weights),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X_val_es, label=y_val_es.to_numpy(dtype=float))],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        best_iter = max(pilot_booster.best_iteration, 1)
        logger.info("DirectHI pilot early-stop: best_iter=%d", best_iter)
        del pilot_booster
        gc.collect()

        # Phase 2: refit ALL boosters on train+val_es at best_iter (no early-stop).
        X_tv = pd.concat([X_train, X_val_es], ignore_index=True)
        y_tv = pd.concat([y_train, y_val_es], ignore_index=True)
        weights_tv = _make_sample_weight(y_tv.to_numpy(dtype=float))

        quantile_map = {"q05": 0.05, "q50": 0.50, "q95": 0.95, "q97": 0.97}
        for role, alpha in quantile_map.items():
            seeds = _SEEDS if alpha == 0.50 else _SEEDS_TAIL
            seed_boosters = []
            for seed in seeds:
                params = {
                    **_DEFAULT_PARAMS,
                    **self._best_params,
                    **_lgbm_device_params(_detect_device()),
                    "alpha": alpha,
                    "seed": seed,
                }
                params = _add_monotone_constraints_if_supported(params, list(X_tv.columns))
                booster = lgb.train(
                    params,
                    lgb.Dataset(X_tv, label=y_tv.to_numpy(dtype=float), weight=weights_tv),
                    num_boost_round=best_iter,
                )
                seed_boosters.append(booster)
            self._boosters[role] = seed_boosters

        # Mondrian CQR calibration on val_cal — boosters never saw this split.
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

        lo_cal, hi_cal = self._calibrate(X_val_cal, bundle_val_cal.hi_lower, bundle_val_cal.hi_upper)
        self._metadata["median_pi_width"] = float(np.median(hi_cal - lo_cal))

        if len(X_test) > 0:
            bundle = self._predict_bundle_calibrated(X_test)
            y_true = y_test.to_numpy(dtype=float)
            mae = float(np.mean(np.abs(bundle.hi_mean - y_true)))
            coverage = float(((y_true >= bundle.hi_lower) & (y_true <= bundle.hi_upper)).mean())
            self._metadata.update({
                "mae": round(mae, 3),
                "pi_coverage_90": round(coverage, 3),
                "n_test": len(X_test),
                "n_train": len(X_train),
                "split_metadata": split.metadata,
            })
        self._metadata["train_seconds"] = float(time.perf_counter() - started)

    def _tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        weights: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        import lightgbm as lgb
        import optuna

        def objective(trial: optuna.Trial) -> float:
            dev = _detect_device()
            _max_bin_choices = [127, 255, 511]
            _suggested_bin = trial.suggest_categorical("max_bin", _max_bin_choices)
            _effective_bin = min(_suggested_bin, 255) if dev == "gpu" else _suggested_bin
            params = {
                **_DEFAULT_PARAMS,
                **_lgbm_device_params(dev),
                "alpha": 0.50,
                "num_leaves": trial.suggest_int("num_leaves", 64, 256),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "max_bin": _effective_bin,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "seed": self.random_state,
            }

            _should_prune = [False]

            def _pruning_cb(env) -> None:
                if not env.evaluation_result_list:
                    return
                val_loss = env.evaluation_result_list[0][2]
                trial.report(float(val_loss), step=env.iteration)
                if trial.should_prune():
                    _should_prune[0] = True

            dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1), _pruning_cb],
            )
            if _should_prune[0]:
                raise optuna.TrialPruned()
            return float(np.mean(np.abs(booster.predict(X_val) - y_val.values)))

        # Change 6: persistent Optuna study with warm-start (LGBMDirectHIForecaster)
        # station_id is included so parallel --workers runs don't share TPE state
        _feat_sig = feature_signature(
            list(X_train.columns),
            extra="lags=[1,3,6,12,24]|roll=[3,6,24]",
        )
        _hi_space_keys = [
            "max_bin", "num_leaves", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_samples",
            "min_child_weight", "reg_alpha", "reg_lambda",
        ]
        _space_sig = hashlib.md5(",".join(_hi_space_keys).encode()).hexdigest()[:8]
        _dev_for_name = _detect_device()
        storage_path = _optuna_storage_path()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        study_name = f"lgbm_{self._station_id}_{self.target_kind}_h{self._horizon_h}_{_feat_sig}_{_dev_for_name}_{_space_sig}"
        try:
            _storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path.as_posix()}?check_same_thread=False")
        except Exception:
            _storage = None

        try:
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state, n_startup_trials=10),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=150, reduction_factor=3),
                storage=_storage,
                load_if_exists=True,
            )
        except Exception as exc:
            logger.warning("Optuna persistent storage unavailable (%s). Falling back to in-memory study.", exc)
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state, n_startup_trials=10),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=150, reduction_factor=3),
            )
        run_id = os.environ.get("HEATSHIELD_RUN_ID")
        slot = f"{self._station_id}:h{self._horizon_h}"

        def _heartbeat_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            write_heartbeat(
                run_id,
                current_slot=slot,
                phase="tune",
                trials_done=sum(t.state.name == "COMPLETE" for t in study.trials),
            )

        remaining_trials = _remaining_trials(study, self.n_trials)
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False, callbacks=[_heartbeat_cb])
        else:
            logger.info("Optuna study already has %d complete trials; reusing best params", self.n_trials)
        best_params = _sanitize_lgbm_params_for_device(study.best_params, _dev_for_name)
        self._metadata["n_trials_completed"] = sum(t.state.name == "COMPLETE" for t in study.trials)
        self._metadata["n_trials_pruned"] = sum(t.state.name == "PRUNED" for t in study.trials)
        return best_params

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.reindex(columns=self.feature_list)
        for col, med in self._feature_medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)
        return out.fillna(0.0)

    def _predict_role(self, X: pd.DataFrame, role: str) -> np.ndarray:
        # Change 3g: role can now be "q97" in addition to q05/q50/q95
        aligned = self._align(X)
        return np.stack([b.predict(aligned) for b in self._boosters[role]], axis=0).mean(axis=0)

    def predict_with_pi(self, X: pd.DataFrame, alpha: float = 0.10) -> PredictionBundle:
        return PredictionBundle(
            hi_mean=self._predict_role(X, "q50"),
            hi_lower=self._predict_role(X, "q05"),
            hi_upper=self._predict_role(X, "q95"),
        )

    def _calibrate(
        self, X: pd.DataFrame, lo: np.ndarray, hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Mondrian CQR calibrator to prediction intervals."""
        if self._calibrator is None:
            return lo, hi
        station_ids = np.full(len(X), self._station_id)
        angles = np.arctan2(X["local_hour_sin"].values, X["local_hour_cos"].values)
        local_hours = np.round(angles * 24 / (2 * np.pi)).astype(int) % 24
        return self._calibrator.adjust(
            lo, hi, station_ids=station_ids, local_hours=local_hours,
        )

    def _predict_bundle_calibrated(
        self, X: pd.DataFrame, alpha: float = 0.10
    ) -> PredictionBundle:
        """Predict with calibrated prediction intervals."""
        raw = self.predict_with_pi(X, alpha)
        lo, hi = self._calibrate(X, raw.hi_lower, raw.hi_upper)
        return PredictionBundle(
            hi_mean=raw.hi_mean, hi_lower=lo, hi_upper=hi,
        )

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        # Change 3g: iterates all roles in _boosters including q97
        for role, boosters in self._boosters.items():
            for booster, seed in zip(boosters, _SEEDS):
                booster.save_model(str(dir_path / f"hi_{role}_s{seed}.txt"))
        meta = {
            "backend_name": self.backend_name,
            "target_kind": self.target_kind,
            "feature_list": self.feature_list,
            "feature_medians": self._feature_medians,
            "best_params": self._best_params,
            "station_id": self._station_id,
            "horizon_h": self._horizon_h,
            **self._metadata,
        }
        (dir_path / "bundle.json").write_text(json.dumps(meta, indent=2, default=str))

        if self._calibrator:
            self._calibrator.save(dir_path / "calibrator.json")

    @classmethod
    def load(cls, dir_path: Path) -> "LGBMDirectHIForecaster":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        dir_path = Path(dir_path)
        bundle = json.loads((dir_path / "bundle.json").read_text())
        obj = cls()
        obj.feature_list = bundle["feature_list"]
        obj._feature_medians = bundle.get("feature_medians", {})
        obj._best_params = bundle.get("best_params", {})
        obj._station_id = bundle.get("station_id", "")
        obj._horizon_h = bundle.get("horizon_h", 24)
        obj.target_kind = bundle.get("target_kind", "hi")
        obj._metadata = {k: v for k, v in bundle.items()
                         if k not in ("feature_list", "feature_medians", "best_params",
                                      "backend_name", "target_kind", "station_id", "horizon_h")}
        # Change 3g: load iterates all keys in _boosters (includes q97) — no hardcoded list
        for role in obj._boosters:
            boosters = []
            for seed in _SEEDS:
                path = dir_path / f"hi_{role}_s{seed}.txt"
                if path.exists():
                    boosters.append(lgb.Booster(model_file=str(path)))
            obj._boosters[role] = boosters

        cal_path = dir_path / "calibrator.json"
        if cal_path.exists():
            cal_data = json.loads(cal_path.read_text())
            if "stratum_q" in cal_data:
                obj._calibrator = MondianCQRCalibrator.from_dict(cal_data)
            else:
                obj._calibrator = EnbPICalibrator.from_dict(cal_data)

        return obj
