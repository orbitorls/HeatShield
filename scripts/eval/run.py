"""Comprehensive model evaluation — confusion matrix, residuals, per-station/hour analysis.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --start 2024-05-03 --end 2025-04-30
    python scripts/evaluate_model.py --out logs/eval
    python scripts/evaluate_model.py --version 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error,
)
import xgboost as xgb

from app.data.loaders import read_observations
from app.data.stations import STATIONS
from app.ml.forecast.features import build_features, _get_lags_for_horizon, _get_rolling_for_horizon
from app.ml.registry import load_latest
from app.ml.forecast.evaluation import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10

# Heat-alert categories matching app/core/heat_index.py
# Thai-adapted WHO thresholds: Caution (27-32°C), Extreme Caution (32-40°C), 
# Danger (40-54°C), Extreme Danger (>54°C)
_CATEGORIES = ["Caution", "Extreme\nCaution", "Danger", "Extreme\nDanger"]
_THRESHOLDS = [32, 40, 54]  # lower bound of each step above Caution
_CAT_COLORS = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]


def _categorize(hi_series: pd.Series) -> pd.Series:
    """Map heat-index values → 0=Caution, 1=ExtrCaution, 2=Danger, 3=ExtremeDanger."""
    cats = pd.cut(
        hi_series,
        bins=[-np.inf, 32, 40, 54, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)
    return cats


def _load_test_data(start: date, end: date, horizon_h: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load observations for all stations, build features, return test split."""
    frames: list[pd.DataFrame] = []
    for sid in STATIONS:
        df = read_observations(sid, start, end)
        if df.empty:
            logger.warning("No data for %s", sid)
            continue
        df["station_id"] = sid
        frames.append(df)

    if not frames:
        raise RuntimeError("No data loaded — run ingest first.")

    all_data = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows across %d stations", len(all_data), len(frames))

    # Use horizon-specific lags and rolling windows
    lags_h = _get_lags_for_horizon(horizon_h)
    rolling_h = _get_rolling_for_horizon(horizon_h)
    X, y = build_features(all_data, horizon_h=horizon_h,
                          lags_h=lags_h, rolling_h=rolling_h)
    logger.info("Feature matrix: %s", X.shape)

    # Same 70/15/15 split used during training
    n = len(X)
    val_end = int(n * 0.85)
    X_test = X.iloc[val_end:].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)

    return X_test, y_test, all_data


def _predict(booster: xgb.Booster, metadata: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Align columns with trained feature list and run inference."""
    feature_list = metadata["feature_list"]
    medians = metadata.get("feature_medians", {})
    for col in feature_list:
        if col not in X_test.columns:
            X_test[col] = medians.get(col, 0.0)
    X_aligned = X_test[feature_list]
    dmat = xgb.DMatrix(X_aligned.values, feature_names=feature_list)
    return booster.predict(dmat)


def _predict_ensemble(loaded: dict, metadata: dict, X_test: pd.DataFrame,
                      horizon_h: int = 24) -> dict:
    """Run inference with mean/q05/q95 boosters and return all predictions.

    Returns: {"mean": np.ndarray, "q05": np.ndarray, "q95": np.ndarray}
    Falls back to single-booster mean-only if v1 format.
    For multi-horizon models, uses horizon_h to select the correct boosters.
    """
    feature_list = metadata["feature_list"]
    medians = metadata.get("feature_medians", {})
    X = X_test.copy()
    for col in feature_list:
        if col not in X.columns:
            X[col] = medians.get(col, 0.0)
    X_aligned = X[feature_list]
    dmat = xgb.DMatrix(X_aligned.values, feature_names=feature_list)

    # Multi-horizon: extract boosters for the requested horizon
    if "horizons" in loaded:
        h_data = loaded["horizons"].get(horizon_h)
        if h_data is None:
            fallback_h = next(iter(loaded["horizons"]))
            logger.warning("Horizon h%d not in model; falling back to h%d", horizon_h, fallback_h)
            h_data = loaded["horizons"][fallback_h]
        role_source = h_data
    else:
        role_source = loaded

    result = {}
    for role in ["mean", "q05", "q95"]:
        boosters = role_source.get(role, [])
        if boosters:
            preds = np.mean([b.predict(dmat) for b in boosters], axis=0)
        elif role == "mean":
            preds = loaded["booster"].predict(dmat)  # v1 fallback
        else:
            continue
        result[role] = preds
    return result


def _load_model_path(path: str) -> dict:
    """Load a specific v1/v2 XGBoost artifact path instead of load_latest()."""
    from app.ml.registry import _load_role_boosters

    p = Path(path)
    if p.is_dir():
        metadata = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        horizon_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("h")]
        if horizon_dirs:
            horizons = {int(d.name[1:]): _load_role_boosters(d) for d in horizon_dirs}
            shim_h = 24 if 24 in horizons else min(horizons)
            return {"horizons": horizons, "metadata": metadata, "booster": horizons[shim_h]["mean"][0]}
        loaded = _load_role_boosters(p)
        return {**loaded, "metadata": metadata, "booster": loaded["mean"][0]}

    booster = xgb.Booster()
    booster.load_model(str(p))
    meta_path = p.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {"booster": booster, "metadata": metadata}


# ── Individual chart functions ──────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> Path:
    """Confusion matrix on risk categories."""
    true_cat = _categorize(pd.Series(y_true))
    pred_cat = _categorize(pd.Series(y_pred))

    cm = confusion_matrix(true_cat, pred_cat, labels=[0, 1, 2, 3])
    row_sums = cm.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    cm_pct = cm.astype(float) / safe_sums * 100
    cm_pct[row_sums.flatten() == 0] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=_CATEGORIES, yticklabels=_CATEGORIES,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Count"})
    ax.set_title("Confusion Matrix — Risk Category (Count)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted Category", fontweight="bold")
    ax.set_ylabel("Actual Category", fontweight="bold")

    # Row-normalised %
    ax = axes[1]
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=_CATEGORIES, yticklabels=_CATEGORIES,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Row %"})
    ax.set_title("Confusion Matrix — Recall per Category (%)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted Category", fontweight="bold")
    ax.set_ylabel("Actual Category", fontweight="bold")

    plt.suptitle("Heat-Alert Category Classification Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


def plot_classification_report_chart(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> Path:
    """Precision / Recall / F1 per risk category as grouped bar chart."""
    true_cat = _categorize(pd.Series(y_true))
    pred_cat = _categorize(pd.Series(y_pred))

    report = classification_report(
        true_cat, pred_cat,
        labels=[0, 1, 2, 3],
        target_names=["Caution", "Ext.Caution", "Danger", "Ext.Danger"],
        output_dict=True, zero_division=0,
    )

    cats = ["Caution", "Ext.Caution", "Danger", "Ext.Danger"]
    precision = [report[c]["precision"] for c in cats]
    recall    = [report[c]["recall"]    for c in cats]
    f1        = [report[c]["f1-score"]  for c in cats]
    support   = [report[c]["support"]   for c in cats]

    x = np.arange(len(cats))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.bar(x - width, precision, width, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x,         recall,    width, label="Recall",    color="#2ecc71", alpha=0.85)
    ax.bar(x + width, f1,        width, label="F1-Score",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per Risk Category", fontweight="bold")
    ax.legend()
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i,         r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    colors = [_CAT_COLORS[i] for i in range(4)]
    ax.bar(cats, support, color=colors, alpha=0.85, edgecolor="black")
    ax.set_title("Class Distribution in Test Set", fontweight="bold")
    ax.set_ylabel("Number of samples")
    for i, (cat, sup) in enumerate(zip(cats, support)):
        ax.text(i, sup + 20, f"{sup:,}", ha="center", fontweight="bold")

    plt.suptitle("Classification Metrics per Heat-Alert Category", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "classification_report.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> Path:
    """Scatter: predicted vs actual heat index with category zones."""
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter
    ax = axes[0]
    for i, (lo, hi, col, cat) in enumerate(zip(
        [-np.inf, 33, 42, 52],
        [33, 42, 52, np.inf],
        _CAT_COLORS, _CATEGORIES
    )):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            ax.scatter(y_true[mask], y_pred[mask], alpha=0.15, s=4,
                       color=col, label=f"{cat.replace(chr(10), ' ')} (n={mask.sum():,})")
    lo_val = min(y_true.min(), y_pred.min()) - 1
    hi_val = max(y_true.max(), y_pred.max()) + 1
    ax.plot([lo_val, hi_val], [lo_val, hi_val], "k--", linewidth=1.5, label="Perfect prediction")
    for thresh in _THRESHOLDS:
        ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(thresh, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Actual Heat Index (°C)", fontweight="bold")
    ax.set_ylabel("Predicted Heat Index (°C)", fontweight="bold")
    ax.set_title("Predicted vs Actual Heat Index", fontweight="bold")
    ax.legend(markerscale=4, fontsize=8)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(0.03, 0.97, f"MAE={mae:.2f}°C  RMSE={rmse:.2f}°C  r={corr:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=80, color="#3498db", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.axvline(residuals.mean(), color="red", linewidth=1.5, label=f"Mean={residuals.mean():.2f}°C")
    ax.axvline(np.percentile(residuals, 10), color="#e67e22", linewidth=1, linestyle=":",
               label=f"P10={np.percentile(residuals,10):.2f}°C")
    ax.axvline(np.percentile(residuals, 90), color="#e67e22", linewidth=1, linestyle=":",
               label=f"P90={np.percentile(residuals,90):.2f}°C")
    ax.set_xlabel("Residual (Predicted − Actual, °C)", fontweight="bold")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.legend()

    plt.suptitle("Prediction Quality Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "pred_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


def plot_error_by_hour(y_true: np.ndarray, y_pred: np.ndarray,
                       X_test: pd.DataFrame, out_dir: Path) -> Path:
    """MAE and bias (mean error) broken down by hour of day."""
    residuals = y_pred - y_true
    abs_err = np.abs(residuals)

    hour_sin = X_test["hour_sin"].values
    hour_cos = X_test["hour_cos"].values
    hours = (np.round(np.arctan2(hour_sin, hour_cos) / (2 * np.pi / 24)) % 24).astype(int)

    hourly_mae  = [abs_err[hours == h].mean() if (hours == h).sum() > 0 else 0 for h in range(24)]
    hourly_bias = [residuals[hours == h].mean() if (hours == h).sum() > 0 else 0 for h in range(24)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    bars = ax.bar(range(24), hourly_mae, color=plt.cm.RdYlGn_r(np.array(hourly_mae) / max(hourly_mae)),
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("MAE (°C)", fontweight="bold")
    ax.set_title("Mean Absolute Error by Hour of Day (UTC)", fontweight="bold")
    ax.axhline(np.mean(hourly_mae), color="navy", linestyle="--", linewidth=1.5,
               label=f"Overall MAE = {np.mean(abs_err):.2f}°C")
    ax.legend()

    ax = axes[1]
    colors = ["#e74c3c" if b > 0 else "#3498db" for b in hourly_bias]
    ax.bar(range(24), hourly_bias, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Hour of Day (UTC+0)", fontweight="bold")
    ax.set_ylabel("Bias (°C)", fontweight="bold")
    ax.set_title("Prediction Bias by Hour (red=overpredict, blue=underpredict)", fontweight="bold")
    ax.set_xticks(range(24))

    plt.suptitle("Temporal Error Analysis — Hour of Day", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "error_by_hour.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


def plot_error_by_station(y_true: np.ndarray, y_pred: np.ndarray,
                          X_test: pd.DataFrame, out_dir: Path) -> Path:
    """MAE and skill score broken down per station."""
    abs_err = np.abs(y_pred - y_true)
    station_codes = X_test["station_enc"].values

    # Map encoded → station name
    station_ids = list(STATIONS.keys())
    enc_to_name = {
        i: sid
        for i, sid in enumerate(sorted(station_ids))
    }

    station_mae: dict[str, float] = {}
    station_rmse: dict[str, float] = {}
    station_n: dict[str, int] = {}

    for enc in np.unique(station_codes):
        mask = station_codes == enc
        name = enc_to_name.get(int(enc), f"Station {enc}")
        station_mae[name]  = float(abs_err[mask].mean())
        station_rmse[name] = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        station_n[name]    = int(mask.sum())

    names = list(station_mae.keys())
    maes  = [station_mae[n] for n in names]
    rmses = [station_rmse[n] for n in names]
    ns    = [station_n[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    x = np.arange(len(names))
    width = 0.4
    ax.bar(x - width/2, maes,  width, label="MAE",  color="#3498db", alpha=0.85, edgecolor="black")
    ax.bar(x + width/2, rmses, width, label="RMSE", color="#e74c3c", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Error (°C)", fontweight="bold")
    ax.set_title("MAE and RMSE per Station", fontweight="bold")
    ax.legend()
    for i, (m, r) in enumerate(zip(maes, rmses)):
        ax.text(i - width/2, m + 0.05, f"{m:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + width/2, r + 0.05, f"{r:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax = axes[1]
    ax.bar(range(len(names)), ns, color=_CAT_COLORS[:len(names)], edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Test samples", fontweight="bold")
    ax.set_title("Test Sample Count per Station", fontweight="bold")
    for i, n in enumerate(ns):
        ax.text(i, n + 10, f"{n:,}", ha="center", fontsize=9)

    plt.suptitle("Per-Station Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "error_by_station.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


def plot_error_heatmap(y_true: np.ndarray, y_pred: np.ndarray,
                       X_test: pd.DataFrame, out_dir: Path) -> Path:
    """MAE heatmap: station (rows) × hour-of-day (cols)."""
    abs_err = np.abs(y_pred - y_true)
    station_codes = X_test["station_enc"].values

    hour_sin = X_test["hour_sin"].values
    hour_cos = X_test["hour_cos"].values
    hours = (np.round(np.arctan2(hour_sin, hour_cos) / (2 * np.pi / 24)) % 24).astype(int)

    station_ids = list(STATIONS.keys())
    enc_to_name = {i: sid for i, sid in enumerate(sorted(station_ids))}

    unique_encs = sorted(np.unique(station_codes).astype(int))
    station_names = [enc_to_name.get(e, f"Stn{e}") for e in unique_encs]

    matrix = np.zeros((len(unique_encs), 24))
    for i, enc in enumerate(unique_encs):
        for h in range(24):
            mask = (station_codes == enc) & (hours == h)
            matrix[i, h] = abs_err[mask].mean() if mask.sum() > 0 else np.nan

    fig, ax = plt.subplots(figsize=(16, max(4, len(unique_encs) * 1.2)))
    im = sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=range(24), yticklabels=station_names,
        linewidths=0.3, ax=ax, cbar_kws={"label": "MAE (°C)"},
    )
    ax.set_xlabel("Hour of Day (UTC)", fontweight="bold")
    ax.set_ylabel("Station", fontweight="bold")
    ax.set_title("MAE Heatmap: Station × Hour of Day", fontweight="bold", fontsize=13)

    plt.tight_layout()
    path = out_dir / "error_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


# ── v2-specific chart functions ─────────────────────────────────────────────

def plot_pi_calibration(y_true: np.ndarray, q05_preds: np.ndarray,
                        q95_preds: np.ndarray, out_dir: Path) -> Path:
    """Plot PI calibration: actual coverage vs expected for q05 and q95 points.

    Also annotates the 90% PI coverage and acceptance zone [0.87, 0.93].
    Saves to out_dir/v2/pi_calibration.png.
    """
    # 90% PI coverage: fraction of y_true inside [q05_pred, q95_pred]
    coverage_90 = float(((y_true >= q05_preds) & (y_true <= q95_preds)).mean())

    # Empirical coverage at the two quantile levels we have
    # For level α=0.05: fraction of y_true <= q05_pred (should be ~0.05)
    # For level α=0.95: fraction of y_true <= q95_pred (should be ~0.95)
    actual_q05_coverage = float((y_true <= q05_preds).mean())
    actual_q95_coverage = float((y_true <= q95_preds).mean())

    expected_levels = [0.05, 0.95]
    actual_coverages = [actual_q05_coverage, actual_q95_coverage]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Perfect calibration diagonal
    diag = np.linspace(0, 1, 100)
    ax.plot(diag, diag, "k--", linewidth=1.5, label="Perfect calibration", zorder=1)

    # Acceptance zone band [0.87, 0.93] — shown as a horizontal band around 90% PI
    # We show it as a shaded region between y=0.87 and y=0.93 for x in [0.87, 0.93]
    ax.axhspan(0.87, 0.93, alpha=0.15, color="#2ecc71", label="Acceptance zone [0.87, 0.93]")

    # Plot the two calibration points
    ax.scatter(expected_levels, actual_coverages, color="#e74c3c", s=120, zorder=5,
               label="Observed calibration (q05, q95)")
    for exp, act in zip(expected_levels, actual_coverages):
        ax.annotate(f"({exp:.2f}, {act:.3f})", xy=(exp, act),
                    xytext=(exp + 0.04, act - 0.03), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray"))

    # Annotate 90% PI coverage
    ax.text(0.05, 0.92, f"90% PI coverage: {coverage_90:.1%}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected quantile level", fontweight="bold")
    ax.set_ylabel("Actual coverage (fraction of y_true ≤ predicted quantile)", fontweight="bold")
    ax.set_title("PI Calibration Plot — Forecast v2", fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    path = out_dir / "pi_calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


def plot_pi_width(y_true: np.ndarray, q05_preds: np.ndarray,
                  q95_preds: np.ndarray, out_dir: Path) -> Path:
    """Histogram of PI widths with P25/P75 lines and ratio annotation.

    Saves to out_dir/v2/pi_width.png.
    """
    pi_width = q95_preds - q05_preds
    p25, p75 = float(np.percentile(pi_width, 25)), float(np.percentile(pi_width, 75))
    ratio = p75 / max(p25, 1e-6)
    ratio_color = "#2ecc71" if ratio >= 1.5 else "#e74c3c"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(pi_width, bins=60, color="#3498db", alpha=0.8, edgecolor="none")
    ax.axvline(p25, color="#e67e22", linewidth=2.0, linestyle="--", label=f"P25 = {p25:.1f}°C")
    ax.axvline(p75, color="#8e44ad", linewidth=2.0, linestyle="--", label=f"P75 = {p75:.1f}°C")

    ax.set_xlabel("PI Width = q95 − q05 (°C)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of 90% Prediction Interval Widths — Forecast v2",
                 fontweight="bold", fontsize=13)

    # Annotate ratio
    ax.text(0.97, 0.95,
            f"P25={p25:.1f}°C  P75={p75:.1f}°C  Ratio={ratio:.2f} (target ≥1.5)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="bold",
            color=ratio_color,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.legend(fontsize=10)
    plt.tight_layout()
    path = out_dir / "pi_width.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


def plot_v1_v2_comparison(metadata: dict, y_true: np.ndarray, mean_preds: np.ndarray,
                          q05_preds: np.ndarray | None, q95_preds: np.ndarray | None,
                          out_dir: Path) -> list[Path]:
    """Side-by-side bar chart comparing v1 and v2 metrics.

    Loads v1 metrics from app/models/forecast_v1.json if available.
    Writes comparison.md and comparison.png to out_dir.
    """
    # ── Compute v2 metrics ──────────────────────────────────────────────────
    mae_v2  = float(mean_absolute_error(y_true, mean_preds))
    rmse_v2 = float(np.sqrt(mean_squared_error(y_true, mean_preds)))

    # Skill score: 1 - MAE / MAE_persistence
    mae_persistence = float(np.abs(np.diff(y_true)).mean()) if len(y_true) > 1 else 1e-6
    skill_v2 = 1.0 - mae_v2 / max(mae_persistence, 1e-6)

    # Danger recall (HI > 42°C)
    danger_mask = y_true > 42.0
    if danger_mask.sum() > 0:
        danger_recall_v2 = float((mean_preds[danger_mask] > 42.0).mean())
    else:
        danger_recall_v2 = float("nan")

    # 90% PI coverage
    if q05_preds is not None and q95_preds is not None:
        pi_coverage_v2 = float(((y_true >= q05_preds) & (y_true <= q95_preds)).mean())
    else:
        pi_coverage_v2 = float("nan")

    # ── Load v1 metrics ─────────────────────────────────────────────────────
    v1_json_path = Path(__file__).parents[2] / "app" / "models" / "forecast_v1.json"
    v1_metrics: dict = {}
    if v1_json_path.exists():
        try:
            with open(v1_json_path, "r") as f:
                v1_metrics = json.load(f)
            logger.info("Loaded v1 metrics from %s", v1_json_path)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", v1_json_path, exc)

    mae_v1          = v1_metrics.get("mae", 2.87)
    rmse_v1         = v1_metrics.get("rmse", 3.62)
    skill_v1        = v1_metrics.get("skill_score", 0.545)
    danger_recall_v1= v1_metrics.get("danger_recall", 0.0)

    # ── Bar chart ────────────────────────────────────────────────────────────
    metrics_names   = ["MAE (°C)", "RMSE (°C)", "Skill Score", "Danger Recall"]
    v1_vals         = [mae_v1, rmse_v1, skill_v1, danger_recall_v1]
    v2_vals         = [mae_v2, rmse_v2, skill_v2, danger_recall_v2]
    # Acceptance thresholds (per metric)
    thresholds      = [2.0, None, 0.55, 0.5]
    # Whether lower is better (MAE/RMSE) or higher is better (Skill, Recall)
    lower_is_better = [True, True, False, False]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))
    bars_v1 = ax.bar(x - width / 2, v1_vals, width, label="v1 (baseline)",
                     color="#95a5a6", alpha=0.85, edgecolor="black")
    bars_v2 = ax.bar(x + width / 2, v2_vals, width, label="v2 (current)",
                     color="#3498db", alpha=0.85, edgecolor="black")

    # Acceptance threshold lines
    threshold_styles = {"linestyle": "--", "linewidth": 1.8, "alpha": 0.85}
    threshold_colors = ["#e74c3c", None, "#2ecc71", "#f39c12"]
    for i, (thresh, tcol) in enumerate(zip(thresholds, threshold_colors)):
        if thresh is not None:
            ax.hlines(thresh, i - width, i + width, colors=tcol,
                      label=f"Target {metrics_names[i]}: {thresh}",
                      **threshold_styles)

    # Annotate bar values
    for bar in bars_v1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="gray")
    for bar in bars_v2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylabel("Metric value", fontweight="bold")
    ax.set_title("Model Comparison: v1 vs v2 — HeatShield AI Forecast",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    chart_path = out_dir / "comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", chart_path)

    # ── Markdown table ───────────────────────────────────────────────────────
    def _fmt(val: float, decimals: int = 2) -> str:
        return f"{val:.{decimals}f}" if not np.isnan(val) else "N/A"

    def _status(val: float, target: float | None, lower_better: bool) -> str:
        if target is None or np.isnan(val):
            return "—"
        ok = (val <= target) if lower_better else (val >= target)
        return "✓" if ok else "✗"

    rows = [
        ("MAE (°C)",       _fmt(mae_v1),           _fmt(mae_v2),           "≤ 2.00",
         _status(mae_v2, 2.0, True)),
        ("RMSE (°C)",      _fmt(rmse_v1),           _fmt(rmse_v2),          "—",
         "—"),
        ("Skill Score",    _fmt(skill_v1, 3),       _fmt(skill_v2, 3),      "≥ 0.55",
         _status(skill_v2, 0.55, False)),
        ("Danger Recall",  _fmt(danger_recall_v1, 3), _fmt(danger_recall_v2, 3), "≥ 0.50",
         _status(danger_recall_v2, 0.5, False)),
        ("PI Coverage 90%", "—",                    _fmt(pi_coverage_v2, 3), "∈ [0.87, 0.93]",
         "✓" if (not np.isnan(pi_coverage_v2) and 0.87 <= pi_coverage_v2 <= 0.93) else "✗"),
    ]

    md_lines = [
        "| Metric | v1 | v2 | Target | Status |",
        "|--------|----|----|--------|--------|",
    ]
    for name, v1_s, v2_s, target, status in rows:
        md_lines.append(f"| {name} | {v1_s} | {v2_s} | {target} | {status} |")

    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Saved: %s", md_path)

    return [chart_path, md_path]


def print_summary(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a full evaluation summary to stdout."""
    true_cat = _categorize(pd.Series(y_true))
    pred_cat = _categorize(pd.Series(y_pred))

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = (y_pred - y_true).mean()
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    p90  = float(np.percentile(np.abs(y_pred - y_true), 90))

    logger.info("")
    logger.info("=" * 60)
    logger.info("REGRESSION METRICS (test set)")
    logger.info("=" * 60)
    logger.info("  MAE   : %.4f °C", mae)
    logger.info("  RMSE  : %.4f °C", rmse)
    logger.info("  Bias  : %+.4f °C  (positive = overpredict)", bias)
    logger.info("  r     : %.4f", corr)
    logger.info("  P90   : %.4f °C", p90)

    logger.info("")
    logger.info("=" * 60)
    logger.info("CLASSIFICATION REPORT (risk category)")
    logger.info("=" * 60)
    cat_names = ["Caution", "ExtrCaution", "Danger", "ExtDanger"]
    print(classification_report(
        true_cat, pred_cat,
        labels=[0, 1, 2, 3],
        target_names=cat_names,
        zero_division=0,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HeatShield AI forecast model")
    parser.add_argument("--start", type=str, default=None,
                        help="Data start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="Data end date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--horizon", type=int, default=24,
                        help="Forecast horizon in hours (default: 24)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for charts (default: logs/eval)")
    parser.add_argument("--version", type=int, default=None, choices=[1, 2],
                        help="Force evaluation mode: 1=v1 only, 2=v2 with PI plots "
                             "(auto-detected if not specified)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Evaluate a specific saved model file/directory instead of load_latest().")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run id to include in the default output path.")
    args = parser.parse_args()

    today = date.today()
    end_date   = date.fromisoformat(args.end)   if args.end   else today - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=730)

    if args.out:
        out_dir = Path(args.out)
    elif args.run_id:
        out_dir = Path("logs") / "eval" / "runs" / args.run_id / "manual" / "all" / f"h{args.horizon}"
    else:
        out_dir = Path("logs") / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model...")
    loaded = _load_model_path(args.model_path) if args.model_path else load_latest("forecast")
    metadata = loaded["metadata"]
    logger.info("Model version %s | trained on %s", metadata.get("version"), metadata.get("training_date_utc"))

    logger.info("Loading test data (%s → %s)...", start_date, end_date)
    X_test, y_test, _ = _load_test_data(start_date, end_date, args.horizon)

    logger.info("Running ensemble inference on %d test rows...", len(X_test))
    preds = _predict_ensemble(loaded, metadata, X_test, horizon_h=args.horizon)
    y_pred  = preds["mean"]
    q05_pred = preds.get("q05")
    q95_pred = preds.get("q95")
    y_true  = y_test.values

    # Auto-detect v2 if quantile boosters present
    is_v2 = bool(q05_pred is not None and q95_pred is not None)
    if args.version == 2 and not is_v2:
        logger.warning("--version 2 requested but model has no quantile boosters; "
                       "running in v1 mode.")
    if args.version == 1:
        is_v2 = False  # force v1 mode

    logger.info("Evaluation mode: %s", "v2 (ensemble + PI)" if is_v2 else "v1 (single booster)")

    print_summary(y_true, y_pred)

    logger.info("Generating reusable evaluation artifacts → %s", out_dir)
    metrics = evaluate_predictions(
        y_true,
        y_pred,
        X_test,
        out_dir=out_dir,
        horizon_h=args.horizon,
        station_labels={i: sid for i, sid in enumerate(sorted(STATIONS.keys()))},
        q05=q05_pred if is_v2 else None,
        q95=q95_pred if is_v2 else None,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation complete — metrics.json and charts saved to %s", out_dir)
    logger.info("MAE=%.4f Skill=%.4f",
                metrics["regression"]["mae"], metrics["baselines"]["skill_score"])


if __name__ == "__main__":
    main()
