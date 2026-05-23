"""Visualization of training results — generate charts from model metadata."""
from pathlib import Path
from typing import Optional
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


def viz_model_metrics(metadata: dict, output_dir: Optional[Path] = None) -> Path:
    """Generate metrics comparison chart: model vs persistence vs climatology baselines.

    Args:
        metadata: Model metadata dict from registry containing:
            - metrics: {mae, rmse, p90_abs_err, mae_persistence, mae_climatology, skill_score}
            - training_date_utc: timestamp
        output_dir: Directory to save chart (default: logs/viz)

    Returns:
        Path to saved PNG image
    """
    if output_dir is None:
        output_dir = Path(__file__).parents[2] / "logs" / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = metadata.get("metrics", {})
    mae_model = metrics.get("mae", 0)
    mae_persistence = metrics.get("mae_persistence", 0)
    mae_climatology = metrics.get("mae_climatology", 0)
    rmse = metrics.get("rmse", 0)
    skill_score = metrics.get("skill_score", 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MAE Comparison
    ax = axes[0, 0]
    methods = ["Model", "Persistence\nBaseline", "Climatology\nBaseline"]
    maes = [mae_model, mae_persistence, mae_climatology]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    bars = ax.bar(methods, maes, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("MAE (°C)", fontsize=11, fontweight="bold")
    ax.set_title("Mean Absolute Error Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(maes) * 1.2)
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.2f}°C', ha='center', va='bottom', fontweight='bold')

    # 2. Skill Score
    ax = axes[0, 1]
    skill_display = max(0, min(1, skill_score))  # Clamp to [0, 1] for display
    color = "#2ecc71" if skill_score >= 0.15 else "#e74c3c"
    bar = ax.barh(["Skill Score"], [skill_display], color=color, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Skill Score", fontsize=11, fontweight="bold")
    ax.set_title("Model Skill (1 - MAE/max(persistence,climatology))", fontsize=12, fontweight="bold")
    ax.axvline(0.15, color="red", linestyle="--", linewidth=2, label="Ship Threshold (0.15)")
    ax.text(skill_display + 0.02, 0, f'{skill_score:.3f}', va='center', fontweight='bold', fontsize=11)
    ax.legend(loc="lower right")

    # 3. MAE vs RMSE
    ax = axes[1, 0]
    metrics_names = ["MAE", "RMSE"]
    metrics_vals = [mae_model, rmse]
    colors_metrics = ["#3498db", "#9b59b6"]
    bars = ax.bar(metrics_names, metrics_vals, color=colors_metrics, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Error (°C)", fontsize=11, fontweight="bold")
    ax.set_title("Model Error Metrics", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(metrics_vals) * 1.2)
    for bar, val in zip(bars, metrics_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}°C', ha='center', va='bottom', fontweight='bold')

    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = f"""
TRAINING SUMMARY

Model Metrics:
  • MAE: {mae_model:.3f}°C
  • RMSE: {rmse:.3f}°C
  • p90 Absolute Error: {metrics.get('p90_abs_err', 0):.3f}°C
  • Skill Score: {skill_score:.3f}

Baselines:
  • Persistence MAE: {mae_persistence:.3f}°C
  • Climatology MAE: {mae_climatology:.3f}°C

Shipping Decision:
  • Threshold: Skill Score ≥ 0.15
  • Status: {'✓ PASS' if skill_score >= 0.15 else '✗ FAIL'}

Training Date: {metadata.get('training_date_utc', 'N/A')}
Stations: {', '.join(metadata.get('stations', []))}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("XGBoost Heat Index Forecast — Training Results", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    output_path = output_dir / f"metrics_{metadata.get('version', '1')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def viz_feature_importance(booster: xgb.Booster, metadata: dict, output_dir: Optional[Path] = None) -> Path:
    """Generate feature importance chart from trained XGBoost model.

    Args:
        booster: Trained XGBoost booster object
        metadata: Model metadata dict
        output_dir: Directory to save chart (default: logs/viz)

    Returns:
        Path to saved PNG image
    """
    if output_dir is None:
        output_dir = Path(__file__).parents[2] / "logs" / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_list = metadata.get("feature_list", [])
    importance = booster.get_score(importance_type="weight")

    # Extract feature names and importance values
    features = []
    importances = []
    for feat_name in feature_list:
        features.append(feat_name)
        importances.append(importance.get(feat_name, 0))

    # Sort by importance descending and take top 15
    sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    top_n = min(15, len(sorted_pairs))
    features_top = [p[0] for p in sorted_pairs[:top_n]]
    importances_top = [p[1] for p in sorted_pairs[:top_n]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Feature Importance Bar Chart
    ax = axes[0]
    y_pos = np.arange(len(features_top))
    colors_imp = plt.cm.viridis(np.linspace(0, 1, len(features_top)))
    bars = ax.barh(y_pos, importances_top, color=colors_imp, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_top)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Weight)", fontsize=11, fontweight="bold")
    ax.set_title("Top 15 Feature Importance", fontsize=12, fontweight="bold")
    for i, (bar, imp) in enumerate(zip(bars, importances_top)):
        ax.text(imp, i, f' {int(imp)}', va='center', fontweight='bold')

    # 2. Summary Panel
    ax = axes[1]
    ax.axis("off")
    hyperparams = metadata.get("hyperparams", {})
    summary_text = f"""
FEATURE ENGINEERING & HYPERPARAMETERS

Feature Engineering:
  • Lag features: [1, 3, 6, 12, 24]h
  • Rolling features: [3, 6, 24]h windows
  • Temporal features: hour, day-of-year (sin/cos)
  • Heat index: Rothfusz formula

Total Features: {len(feature_list)}

Top 5 Most Important:
"""
    for i, (feat, imp) in enumerate(sorted_pairs[:5]):
        summary_text += f"\n  {i+1}. {feat}: {int(imp)}"

    summary_text += f"""

XGBoost Hyperparameters:
  • max_depth: {hyperparams.get('max_depth', 'N/A')}
  • learning_rate: {hyperparams.get('learning_rate', 'N/A')}
  • n_estimators: {hyperparams.get('n_estimators', 'N/A')}
  • subsample: {hyperparams.get('subsample', 'N/A')}
  • colsample_bytree: {hyperparams.get('colsample_bytree', 'N/A')}
  • min_child_weight: {hyperparams.get('min_child_weight', 'N/A')}

Training Approach:
  • Cross-validation: TimeSeriesSplit(5)
  • Optimization: Optuna TPE sampler (30 trials)
  • Random state: 42 (reproducible)
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle("XGBoost Heat Index Forecast — Feature Importance", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    output_path = output_dir / f"features_{metadata.get('version', '1')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def generate_report(metadata_path: Path, booster_path: Optional[Path] = None) -> list[Path]:
    """Generate all visualization charts from a metadata file.

    Args:
        metadata_path: Path to model metadata JSON file
        booster_path: Path to .ubj booster file (optional, for feature importance)

    Returns:
        List of generated image file paths
    """
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_dir = metadata_path.parent.parent / "logs" / "viz"

    results = []

    # Metrics chart
    results.append(viz_model_metrics(metadata, output_dir))

    # Feature importance (if booster provided)
    if booster_path and booster_path.exists():
        try:
            booster = xgb.Booster()
            booster.load_model(str(booster_path))
            results.append(viz_feature_importance(booster, metadata, output_dir))
        except Exception as e:
            print(f"Warning: Could not generate feature importance chart: {e}")

    return results
