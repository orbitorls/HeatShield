"""Reusable forecast evaluation metrics and artifacts."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: no popup windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from app.reporting.report import generate_pdf_report
from app.data.stations import STATIONS, STATIONS_BY_REGION

_CATEGORY_LABELS = ["Caution", "Extreme Caution", "Danger", "Extreme Danger"]


def categorize_heat_index(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.cut(
        pd.Series(values),
        bins=[-np.inf, 33, 42, 52, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int).to_numpy()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _hours_from_features(X: pd.DataFrame, n: int) -> np.ndarray:
    if {"hour_sin", "hour_cos"} <= set(X.columns):
        return (np.round(np.arctan2(X["hour_sin"], X["hour_cos"]) / (2 * np.pi / 24)) % 24).astype(int).to_numpy()
    return np.arange(n) % 24


def _station_names(X: pd.DataFrame, labels: Mapping[int, str] | None) -> pd.Series:
    if "station_enc" not in X.columns:
        return pd.Series(["all"] * len(X))
    codes = X["station_enc"].astype(int)
    if labels:
        return codes.map(lambda c: labels.get(int(c), f"station_{int(c)}"))
    return codes.map(lambda c: f"station_{int(c)}")


def _get_region_from_station_id(station_id: str) -> str:
    """Get region from station_id using canonical STATIONS_BY_REGION mapping."""
    for region, station_ids in STATIONS_BY_REGION.items():
        if station_id in station_ids:
            return region.capitalize()
    return "Unknown"


def _get_thai_region(lat: float, lon: float) -> str:
    """Determine Thai region from coordinates (fallback when station_id unavailable)."""
    # Northern: lat > 16°N
    if lat > 16.0:
        return "Northern"
    # Northeastern: lat > 13°N and lon < 102°E
    if lat > 13.0 and lon < 102.0:
        return "Northeastern"
    # Central: 13°N <= lat <= 16°N and 100°E <= lon <= 102°E
    if 13.0 <= lat <= 16.0 and 100.0 <= lon <= 102.0:
        return "Central"
    # Eastern: lat < 13°N and lon > 101°E
    if lat < 13.0 and lon > 101.0:
        return "Eastern"
    # Southern: lat < 13°N and lon < 101°E
    if lat < 13.0 and lon < 101.0:
        return "Southern"
    return "Unknown"


def _get_season(month: int) -> str:
    """Determine Thai season from month (1-12)."""
    # Hot season: March-May (3-5)
    if 3 <= month <= 5:
        return "Hot"
    # Rainy season: June-October (6-10)
    if 6 <= month <= 10:
        return "Rainy"
    # Cool season: November-February (11-2)
    return "Cool"


def _get_hour_bucket(hour: int) -> str:
    """Determine hour bucket from hour (0-23)."""
    # Night: 22-5
    if hour >= 22 or hour < 5:
        return "Night"
    # Morning: 5-11
    if 5 <= hour < 11:
        return "Morning"
    # Afternoon: 11-16
    if 11 <= hour < 16:
        return "Afternoon"
    # Evening: 16-22
    return "Evening"


def _compute_grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    group_col: str,
    group_name: str,
    horizon_h: int,
) -> dict:
    """Compute metrics for each group."""
    if group_col not in X.columns:
        return {}
    
    groups = X[group_col].unique()
    grouped_metrics = {}
    
    for group in groups:
        mask = X[group_col] == group
        if mask.sum() < 10:  # Skip groups with too few samples
            continue
        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]
        group_X = X[mask]
        
        try:
            group_metric = compute_metrics(
                group_y_true,
                group_y_pred,
                group_X,
                horizon_h=horizon_h,
                runtime=None,
                split_metadata=None,
            )
            # Simplify the output for grouped metrics
            grouped_metrics[str(group)] = {
                "count": int(mask.sum()),
                "mae": group_metric["regression"]["mae"],
                "rmse": group_metric["regression"]["rmse"],
                "bias": group_metric["regression"]["bias"],
                "skill_score": group_metric["baselines"]["skill_score"],
                "danger_40_recall": group_metric["safety"]["danger_40"]["recall"],
                "danger_42_recall": group_metric["safety"]["danger_42"]["recall"],
            }
        except Exception:
            continue
    
    return grouped_metrics


def _baseline_metrics(y_true: np.ndarray, X: pd.DataFrame, horizon_h: int = 1) -> dict:
    if "heat_index_c_lag1h" in X.columns:
        persistence = X["heat_index_c_lag1h"].to_numpy(dtype=float)
    else:
        # When lag-1h is absent (e.g. h48/h72 lag-restricted feature sets), shift by
        # horizon_h rows so persistence[i] = y_true[i - horizon_h] = HI(t_i), the
        # true same-time value h steps earlier — a fair "copy current value forward" baseline.
        shift = min(horizon_h, len(y_true) - 1)
        persistence = np.roll(y_true, shift)
        persistence[:shift] = persistence[shift]
    persistence_mae = float(mean_absolute_error(y_true, persistence))

    if {"station_enc", "month", "hour_sin", "hour_cos"} <= set(X.columns):
        frame = pd.DataFrame({
            "y": y_true,
            "station": X["station_enc"].astype(int).to_numpy(),
            "month": X["month"].astype(int).to_numpy(),
            "hour": _hours_from_features(X, len(y_true)),
        })
        # Causal expanding mean: for each row, use only past values in same bucket
        # Sort by index to ensure temporal order
        frame_sorted = frame.reset_index(drop=True)
        # Compute expanding mean within each (station, month, hour) group
        climatology = (
            frame_sorted.groupby(["station", "month", "hour"])["y"]
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
            .reindex(frame_sorted.index)
            .to_numpy()
        )
    else:
        # Causal expanding mean for simple case (no grouping)
        climatology = np.array(pd.Series(y_true).expanding(min_periods=1).mean())
    climatology_mae = float(mean_absolute_error(y_true, climatology))

    return {
        "persistence_mae": persistence_mae,
        "climatology_mae": climatology_mae,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    *,
    q05: np.ndarray | None = None,
    q95: np.ndarray | None = None,
    horizon_h: int,
    runtime: dict | None = None,
    split_metadata: dict | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    abs_err = np.abs(residual)
    baselines = _baseline_metrics(y_true, X, horizon_h=horizon_h)
    best_baseline = min(baselines["persistence_mae"], baselines["climatology_mae"])

    true_cat = categorize_heat_index(y_true)
    pred_cat = categorize_heat_index(y_pred)
    cm = confusion_matrix(true_cat, pred_cat, labels=[0, 1, 2, 3])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums), where=True)
    report = classification_report(
        true_cat,
        pred_cat,
        labels=[0, 1, 2, 3],
        target_names=_CATEGORY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    def danger(threshold: float) -> dict:
        mask = y_true >= threshold
        support = int(mask.sum())
        if support == 0:
            return {"support": 0, "recall": float("nan"), "false_negative_rate": float("nan")}
        recall = float((y_pred[mask] >= threshold).mean())
        return {"support": support, "recall": recall, "false_negative_rate": 1.0 - recall}

    pi_metrics = {"available": False}
    if q05 is not None and q95 is not None:
        q05 = np.asarray(q05, dtype=float)
        q95 = np.asarray(q95, dtype=float)
        width = q95 - q05
        pi_metrics = {
            "available": True,
            "coverage_90": float(((y_true >= q05) & (y_true <= q95)).mean()),
            "mean_width": float(np.mean(width)),
            "p25_width": float(np.percentile(width, 25)),
            "p75_width": float(np.percentile(width, 75)),
            "width_ratio": float(np.percentile(width, 75) / max(np.percentile(width, 25), 1e-6)),
            "pinball_q05": float(np.mean(np.maximum(0.05 * (y_true - q05), (0.05 - 1) * (y_true - q05)))),
            "pinball_q95": float(np.mean(np.maximum(0.95 * (y_true - q95), (0.95 - 1) * (y_true - q95)))),
        }

    # Calculate skill score with proper handling for edge cases
    model_mae = mean_absolute_error(y_true, y_pred)
    # If baseline is extremely small or zero, skill score is not meaningful
    # Use a minimum threshold of 0.1°C to avoid division by near-zero
    effective_baseline = max(best_baseline, 0.1)
    skill_score = float(1.0 - model_mae / effective_baseline)
    # Clamp skill score to reasonable range [-1, 1] for sanity
    skill_score = max(-1.0, min(1.0, skill_score))

    # Compute breakdown metrics by region, season, and hour bucket
    breakdown_metrics = {}
    
    # Add season breakdown if month column exists
    if "month" in X.columns:
        X_with_season = X.copy()
        X_with_season["season"] = X["month"].apply(_get_season)
        breakdown_metrics["by_season"] = _compute_grouped_metrics(
            y_true, y_pred, X_with_season, "season", "season", horizon_h
        )
    
    # Add hour bucket breakdown if hour_sin/hour_cos columns exist
    if {"hour_sin", "hour_cos"} <= set(X.columns):
        X_with_hour_bucket = X.copy()
        hours = _hours_from_features(X, len(X))
        X_with_hour_bucket["hour_bucket"] = pd.Series(hours).apply(_get_hour_bucket)
        breakdown_metrics["by_hour_bucket"] = _compute_grouped_metrics(
            y_true, y_pred, X_with_hour_bucket, "hour_bucket", "hour_bucket", horizon_h
        )
    
    # Add region breakdown using canonical STATIONS_BY_REGION when station_enc available
    if "station_enc" in X.columns:
        X_with_region = X.copy()
        # Map station_enc to station_id, then to region using canonical mapping
        station_ids = _station_names(X, labels={i: sid for i, sid in enumerate(sorted(STATIONS.keys()))})
        X_with_region["region"] = station_ids.apply(_get_region_from_station_id)
        breakdown_metrics["by_region"] = _compute_grouped_metrics(
            y_true, y_pred, X_with_region, "region", "region", horizon_h
        )
    # Fallback to coordinate-based region determination if station_enc unavailable
    elif {"lat", "lon"} <= set(X.columns):
        X_with_region = X.copy()
        X_with_region["region"] = X.apply(lambda row: _get_thai_region(row["lat"], row["lon"]), axis=1)
        breakdown_metrics["by_region"] = _compute_grouped_metrics(
            y_true, y_pred, X_with_region, "region", "region", horizon_h
        )

    return {
        "horizon_h": horizon_h,
        "regression": {
            "mae": float(model_mae),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "bias": float(np.mean(residual)),
            "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
            "p50_abs_error": float(np.percentile(abs_err, 50)),
            "p90_abs_error": float(np.percentile(abs_err, 90)),
            "p95_abs_error": float(np.percentile(abs_err, 95)),
        },
        "baselines": {
            **baselines,
            "skill_score": skill_score,
        },
        "risk": {
            "labels": _CATEGORY_LABELS,
            "classification_report": report,
            "confusion_matrix": cm,
            "confusion_matrix_normalized": cm_norm,
        },
        "safety": {
            "danger_40": danger(40.0),
            "danger_42": danger(42.0),
        },
        "prediction_interval": pi_metrics,
        "runtime": {
            "feature_rows": int(len(X)),
            **(runtime or {}),
        },
        "split": split_metadata or {},
        "breakdown": breakdown_metrics,
    }


def _write_summary(metrics: dict, out_dir: Path) -> None:
    reg = metrics["regression"]
    base = metrics["baselines"]
    pi = metrics["prediction_interval"]
    lines = [
        "# Forecast Evaluation",
        "",
        f"- Horizon: h{metrics['horizon_h']}",
        f"- MAE: {reg['mae']:.3f}",
        f"- RMSE: {reg['rmse']:.3f}",
        f"- Skill score: {base['skill_score']:.3f}",
        f"- Danger recall >=42C: {metrics['safety']['danger_42']['recall']}",
        f"- Danger recall >=40C: {metrics['safety']['danger_40']['recall']}",
        f"- PI available: {pi['available']}",
    ]
    if pi.get("available"):
        lines.append(f"- PI coverage 90: {pi['coverage_90']:.3f}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    X: pd.DataFrame,
    *,
    out_dir: str | Path,
    horizon_h: int,
    station_labels: Mapping[int, str] | None = None,
    q05: np.ndarray | pd.Series | None = None,
    q95: np.ndarray | pd.Series | None = None,
    runtime: dict | None = None,
    split_metadata: dict | None = None,
    station: str = "",
    backend: str = "",
    hyperparams: dict | None = None,
    feature_importance: dict[str, float] | None = None,
    model_version: str = "v3",
    run_id: str = "",
    eval_dpi: int = 80,
    skip_pdf: bool = False,
) -> dict:
    """Compute metrics and write standard evaluation artifacts.
    
    Args:
        skip_pdf: If True, skip PDF report generation for faster evaluation.
    """
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    q05_arr = None if q05 is None else np.asarray(q05, dtype=float)
    q95_arr = None if q95 is None else np.asarray(q95, dtype=float)
    runtime_data = dict(runtime or {})

    # Derive station name from station_labels if not provided
    if not station and station_labels:
        station = list(station_labels.values())[0] if station_labels else ""

    metrics = compute_metrics(
        y_true_arr,
        y_pred_arr,
        X,
        q05=q05_arr,
        q95=q95_arr,
        horizon_h=horizon_h,
        runtime=runtime_data,
        split_metadata=split_metadata,
    )
    metrics["runtime"]["eval_seconds"] = metrics["runtime"].get(
        "eval_seconds", float(time.perf_counter() - started)
    )

    (out_path / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(metrics, out_path)
    if not skip_pdf:
        generate_pdf_report(
            metrics,
            out_path / "report.pdf",
            station=station,
            horizon_h=horizon_h,
            backend=backend,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            X=X,
            station_labels=station_labels,
            q05=q05_arr,
            q95=q95_arr,
            hyperparams=hyperparams,
            feature_importance=feature_importance,
            model_version=model_version,
            run_id=run_id,
            eval_dpi=eval_dpi,
        )
    return _json_safe(metrics)
