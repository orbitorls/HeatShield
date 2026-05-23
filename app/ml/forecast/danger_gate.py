"""Danger-tier gate classifier for HeatShield AI forecast v3.

Three tiers are used across training/inference:
  0 = safe    (HI < 38C)
  1 = warning (38C <= HI < 42C)
  2 = danger  (HI >= 42C)

Two backends available:
  "lightgbm": multiclass LightGBM classifier
  "brf": BalancedRandomForestClassifier (imbalanced-learn)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _training_n_jobs() -> int:
    """Conservative default avoids multiprocessing failures in Colab/Windows sandboxes."""
    return max(1, int(os.environ.get("HEATSHIELD_TRAIN_N_JOBS", "1")))


def _lgbm_device() -> str:
    """Return best LightGBM device respecting runtime env overrides."""
    env = os.getenv("LGBM_DEVICE", "").lower()
    if env:
        return env
    if os.getenv("HEATSHIELD_FORCE_CPU") == "1":
        return "cpu"
    mod = sys.modules.get("app.ml.forecast.backends.lgbm_backend")
    if mod is None:
        import importlib
        mod = importlib.import_module("app.ml.forecast.backends.lgbm_backend")
    detect = getattr(mod, "_detect_device", None)
    if detect is not None:
        return detect()
    return getattr(mod, "_DEVICE_TYPE", "cpu")


class DangerGate:
    """Three-class danger gate with warning/danger thresholds."""

    def __init__(self, gate_backend: Literal["lightgbm", "brf"] = "lightgbm", recall_weight: float = 1.0) -> None:
        self.warning_floor: float = 38.0
        self.danger_floor: float = 42.0
        self.warning_threshold: float = 0.30
        self.danger_threshold: float = 0.35
        self.gate_backend: str = gate_backend
        self.recall_weight: float = recall_weight
        self._clf = None
        self._is_legacy_binary: bool = False

    @property
    def threshold(self) -> float:
        """Backward-compat alias used by older callers."""
        return self.danger_threshold

    @staticmethod
    def _to_tier(y_hi: pd.Series | np.ndarray, warning_floor: float, danger_floor: float) -> np.ndarray:
        y_arr = np.asarray(y_hi, dtype=float)
        return np.where(y_arr >= danger_floor, 2, np.where(y_arr >= warning_floor, 1, 0)).astype(int)

    @staticmethod
    def _proba_3class_from_binary(danger_proba: np.ndarray) -> np.ndarray:
        danger_proba = np.clip(np.asarray(danger_proba, dtype=float), 0.0, 1.0)
        warning_proba = 0.5 * danger_proba
        safe_proba = np.clip(1.0 - warning_proba - danger_proba, 0.0, 1.0)
        proba = np.column_stack([safe_proba, warning_proba, danger_proba])
        denom = np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        return proba / denom

    def fit(
        self,
        X: pd.DataFrame,
        y_hi: pd.Series,
        *,
        warning_floor: float = 38.0,
        danger_floor: float = 42.0,
        val_X: pd.DataFrame | None = None,
        val_y_hi: pd.Series | None = None,
        n_estimators: int = 500,
        random_state: int = 42,
        gate_backend: str | None = None,
    ) -> "DangerGate":
        if gate_backend is not None:
            self.gate_backend = gate_backend
        self.warning_floor = warning_floor
        self.danger_floor = danger_floor

        y_tier = self._to_tier(y_hi, warning_floor, danger_floor)
        if self.gate_backend == "brf":
            self._fit_brf(X, y_tier, n_estimators=n_estimators, random_state=random_state)
        else:
            try:
                self._fit_lgbm(
                    X,
                    y_tier,
                    val_X=val_X,
                    val_y_hi=val_y_hi,
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
            except (ImportError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "DangerGate LightGBM training failed (%s: %s). Falling back to BRF backend. "
                    "samples=%d danger_ratio=%.3f",
                    type(exc).__name__,
                    exc,
                    len(X),
                    (y_tier == 2).mean() if len(y_tier) > 0 else 0.0,
                )
                self.gate_backend = "brf"
                self._fit_brf(X, y_tier, n_estimators=n_estimators, random_state=random_state)

        if val_X is not None and val_y_hi is not None:
            val_tier = self._to_tier(val_y_hi, warning_floor, danger_floor)
            self._sweep_thresholds(val_X, val_tier)
        return self

    def _fit_lgbm(
        self,
        X: pd.DataFrame,
        y_tier: np.ndarray,
        *,
        val_X: pd.DataFrame | None,
        val_y_hi: pd.Series | None,
        n_estimators: int,
        random_state: int,
    ) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        class_counts = np.bincount(y_tier, minlength=3)
        class_weight = {
            cls: float(len(y_tier)) / max(int(count), 1)
            for cls, count in enumerate(class_counts)
        }
        sample_weight = np.array([class_weight[int(cls)] for cls in y_tier], dtype=float)

        # GPU triggers tree-learner assertion failures on small multiclass data — always CPU.
        base_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "seed": random_state,
            "verbose": -1,
            "device_type": "cpu",
            "n_jobs": _training_n_jobs(),
        }
        best_lgbm_params = {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

        params = {**base_params, **best_lgbm_params}
        dtrain = lgb.Dataset(X, label=y_tier, weight=sample_weight)
        valid_sets = None
        callbacks = [lgb.log_evaluation(-1)]
        if val_X is not None and val_y_hi is not None:
            y_val_tier_final = self._to_tier(val_y_hi, self.warning_floor, self.danger_floor)
            dval = lgb.Dataset(val_X, label=y_val_tier_final, reference=dtrain)
            valid_sets = [dval]
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

        self._clf = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self._is_legacy_binary = False

    def _fit_brf(
        self,
        X: pd.DataFrame,
        y_tier: np.ndarray,
        *,
        n_estimators: int,
        random_state: int,
    ) -> None:
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier
        except ImportError as exc:
            raise ImportError("pip install imbalanced-learn>=0.13") from exc

        self._clf = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=8,
            sampling_strategy="all",
            replacement=True,
            bootstrap=True,
            random_state=random_state,
            n_jobs=_training_n_jobs(),
        )
        self._clf.fit(X, y_tier)
        self._is_legacy_binary = False

    def _sweep_thresholds(self, X: pd.DataFrame, y_tier: np.ndarray) -> None:
        proba = self.predict_tier_proba(X)
        warn_true = (y_tier >= 1).astype(int)
        danger_true = (y_tier == 2).astype(int)

        self.warning_threshold = self._best_threshold(
            proba[:, 1] + proba[:, 2],
            warn_true,
            precision_min=0.50,
            recall_min=0.55,
            default=0.30,
            beta=1.0,
        )
        self.danger_threshold = self._best_threshold(
            proba[:, 2],
            danger_true,
            precision_min=0.25,
            recall_min=0.50,
            default=0.25,
            beta=2.5 * self.recall_weight,
        )

    @staticmethod
    def _best_threshold(
        score: np.ndarray,
        y_true: np.ndarray,
        *,
        precision_min: float,
        recall_min: float,
        default: float,
        beta: float = 1.0,
    ) -> float:
        """Find threshold maximizing F-beta score subject to min precision/recall constraints."""
        candidates: list[tuple[float, float]] = []  # (threshold, f_beta)
        for t in np.linspace(0.05, 0.90, 86):
            pred = (score >= t).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            if prec >= precision_min and rec >= recall_min:
                b2 = beta ** 2
                f_beta = (1 + b2) * prec * rec / max(b2 * prec + rec, 1e-9)
                candidates.append((float(t), f_beta))

        if candidates:
            return max(candidates, key=lambda x: x[1])[0]

        # Fallback: best F-beta ignoring constraints
        fallback: list[tuple[float, float]] = []
        for t in np.linspace(0.05, 0.90, 86):
            pred = (score >= t).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            if prec >= 0.40:
                b2 = beta ** 2
                f_beta = (1 + b2) * prec * rec / max(b2 * prec + rec, 1e-9)
                fallback.append((float(t), f_beta))
        return max(fallback, key=lambda x: x[1])[0] if fallback else default

    def predict_tier_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities ordered as [safe, warning, danger]."""
        if self._clf is None:
            raise RuntimeError("DangerGate not fitted — call .fit() first")

        if self.gate_backend == "brf":
            proba = self._clf.predict_proba(X)
            classes = list(getattr(self._clf, "classes_", [0, 1, 2]))
            out = np.zeros((len(X), 3), dtype=float)
            for idx, cls in enumerate(classes):
                if 0 <= int(cls) <= 2:
                    out[:, int(cls)] = proba[:, idx]
            row_sum = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
            return out / row_sum

        raw = self._clf.predict(X)
        raw_arr = np.asarray(raw)
        if raw_arr.ndim == 1:
            self._is_legacy_binary = True
            return self._proba_3class_from_binary(raw_arr)
        self._is_legacy_binary = False
        if raw_arr.shape[1] != 3:
            raise ValueError(f"Unexpected LightGBM gate output shape: {raw_arr.shape}")
        row_sum = np.clip(raw_arr.sum(axis=1, keepdims=True), 1e-12, None)
        return raw_arr / row_sum

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Backward-compatible danger probability P(tier=2)."""
        return self.predict_tier_proba(X)[:, 2]

    def predict_tier(self, X: pd.DataFrame) -> np.ndarray:
        """Predict discrete tier using tuned warning/danger thresholds."""
        proba = self.predict_tier_proba(X)
        warn_score = proba[:, 1] + proba[:, 2]
        danger_score = proba[:, 2]
        tier = np.zeros(len(X), dtype=int)
        tier[warn_score >= self.warning_threshold] = 1
        tier[danger_score >= self.danger_threshold] = 2
        return tier

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Backward-compatible binary danger prediction."""
        return (self.predict_tier(X) == 2).astype(int)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._clf is not None:
            if self.gate_backend == "brf":
                import joblib
                joblib.dump(self._clf, str(path / "gate_clf.joblib"))
            else:
                self._clf.save_model(str(path / "gate_clf.txt"))
        meta = {
            "warning_floor": self.warning_floor,
            "danger_floor": self.danger_floor,
            "warning_threshold": self.warning_threshold,
            "danger_threshold": self.danger_threshold,
            "gate_backend": self.gate_backend,
            "recall_weight": self.recall_weight,
            "version": "v3-tier",
        }
        (path / "gate_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DangerGate":
        path = Path(path)
        meta = json.loads((path / "gate_meta.json").read_text())
        gate_backend = meta.get("gate_backend", "lightgbm")
        obj = cls(gate_backend=gate_backend)
        obj.warning_floor = float(meta.get("warning_floor", 38.0))
        obj.danger_floor = float(meta.get("danger_floor", meta.get("danger_thresh", 42.0)))
        obj.warning_threshold = float(meta.get("warning_threshold", meta.get("threshold", 0.30)))
        obj.danger_threshold = float(meta.get("danger_threshold", meta.get("threshold", 0.35)))
        obj.recall_weight = float(meta.get("recall_weight", 1.0))

        if gate_backend == "brf":
            clf_path = path / "gate_clf.joblib"
            if clf_path.exists():
                import joblib
                obj._clf = joblib.load(str(clf_path))
        else:
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise ImportError("pip install lightgbm") from exc
            clf_path = path / "gate_clf.txt"
            if clf_path.exists():
                obj._clf = lgb.Booster(model_file=str(clf_path))
        return obj
