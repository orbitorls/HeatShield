"""CLI training script for the HeatShield AI heat-index forecast model.

Usage:
    # Multi-horizon (default, trains all 5 horizons):
    python scripts/train_forecast.py [--station BKK_01] [--trials 30]

    # Custom horizons:
    python scripts/train_forecast.py --horizons 6,24,48

    # Legacy single-horizon (falls back to single-horizon save):
    python scripts/train_forecast.py --horizon 24 [--station BKK_01] [--trials 30]

Reads parquet data from data/raw/, builds features, trains XGBoost,
saves model to app/models/forecast_v{n}/ directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np

from app.data.loaders import read_observations
from app.data.stations import STATIONS
from app.ml.forecast.features import (
    build_features,
    build_X_once,
    build_y_for_horizon,
    _get_lags_for_horizon,
    _get_rolling_for_horizon,
    _subset_features_for_horizon,
)
from app.ml.forecast.hpo.optuna_utils import write_heartbeat
from app.ml.forecast.splitting import split_xy
from app.ml.forecast.evaluation import evaluate_predictions
from app.ml.forecast.readiness import evaluate_readiness
from app.data.quality import calculate_data_quality_metrics
from app.ml.registry import save_model
from app.infrastructure.edr import edr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

logger = logging.getLogger(__name__)

# Thread-safe feature pruning cache using a simple dict with station-specific pruning
_pruned_features_cache: dict[str, list[str]] = {}


def _can_use_lgbm_device(device_type: str) -> bool:
    """Probe whether LightGBM can run on a requested device."""
    try:
        import lightgbm as lgb
        import numpy as np
        lgb.train(
            {"objective": "regression", "device_type": device_type, "verbose": -1, "num_leaves": 4},
            lgb.Dataset(np.zeros((8, 2)), np.zeros(8)),
            num_boost_round=1,
        )
        return True
    except (ImportError, RuntimeError, OSError) as e:
        logger.debug("LightGBM device probe failed: %s", e)
        return False
    except Exception as e:
        # LightGBMError and other backend-specific errors (e.g. CUDA not compiled)
        logger.debug("LightGBM device probe failed: %s", e)
        return False


def _configure_training_device(requested: str) -> None:
    """Configure env vars so training prefers the requested device."""
    req = requested.lower().strip()
    if req == "cpu":
        os.environ["HEATSHIELD_FORCE_CPU"] = "1"
        os.environ.pop("LGBM_DEVICE", None)
        logger.info("Training device configured: CPU")
        return

    os.environ["HEATSHIELD_FORCE_CPU"] = "0"
    if req == "auto":
        os.environ.pop("LGBM_DEVICE", None)
        logger.info("Training device configured: auto-detect")
        return

    if req in {"gpu", "cuda"} and _can_use_lgbm_device(req):
        os.environ["LGBM_DEVICE"] = req
        logger.info("Training device configured: LGBM_DEVICE=%s", req)
        return

    alternate = "gpu" if req == "cuda" else "cuda"
    if req in {"gpu", "cuda"} and _can_use_lgbm_device(alternate):
        os.environ["LGBM_DEVICE"] = alternate
        logger.warning(
            "Requested LightGBM device '%s' unavailable. Falling back to '%s'.",
            req,
            alternate,
        )
        return

    os.environ.pop("LGBM_DEVICE", None)
    logger.warning(
        "Requested LightGBM device '%s' unavailable. Falling back to auto-detect.",
        req,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        logger.debug("Git SHA retrieval failed: %s", e)
        return "unknown"


def _extract_hour_from_X(X: pd.DataFrame, n: int) -> list[int]:
    sin_col, cos_col = "local_hour_sin", "local_hour_cos"
    if sin_col in X.columns and cos_col in X.columns:
        angles = np.arctan2(X[sin_col].values, X[cos_col].values)
        angles_norm = np.where(angles < 0, angles + 2 * np.pi, angles)
        hours = np.round(angles_norm / (2 * np.pi) * 24).astype(int) % 24
        return hours.tolist()
    return [0] * n


def _fmt_metrics(metrics: dict) -> dict:
    return {k: f"{v:.4f}" if not (isinstance(v, float) and v != v) else "nan" for k, v in metrics.items()}


def _adaptive_trials(horizon_h: int, requested: int | None = None, quick_mode: bool = False) -> int:
    """Return trial budget adapted to horizon complexity.
    
    Args:
        horizon_h: Forecast horizon in hours
        requested: User-specified trial count (takes precedence)
        quick_mode: If True, use reduced trial budget for faster iteration
    """
    _MIN_TRIALS = {6: 40, 12: 40, 24: 35, 48: 25, 72: 25}
    _QUICK_TRIALS = {6: 10, 12: 10, 24: 10, 48: 8, 72: 8}

    trial_dict = _QUICK_TRIALS if quick_mode else _MIN_TRIALS
    base = trial_dict.get(horizon_h, 10 if quick_mode else 20)
    if requested is not None and requested > 0:
        # In quick mode, allow user to go *below* the base for speed
        return requested if quick_mode else max(base, requested)
    return base


def _compute_tail_sample_weights(
    y_hi: np.ndarray,
    *,
    danger_threshold_c: float = 40.0,
    base_weight: float = 1.0,
    tail_weight: float = 5.0,
) -> np.ndarray:
    """Compute sample weights to give higher importance to tail (danger) examples.
    
    Safer alternative to oversampling - avoids creating synthetic data and
    prevents data leakage. Weights are computed based on heat index values.
    
    Args:
        y_hi: Heat index values for samples
        danger_threshold_c: Threshold for danger class (default 40°C)
        base_weight: Weight for normal examples
        tail_weight: Weight for danger/extreme examples
        
    Returns:
        Sample weights array matching y_hi length
    """
    weights = np.full(len(y_hi), base_weight)
    # Apply higher weight to danger and extreme danger examples
    tail_mask = y_hi >= danger_threshold_c
    weights[tail_mask] = tail_weight
    return weights


def _quality_status(metrics: dict, horizon_h: int = 6) -> str:
    return evaluate_readiness(metrics, horizon_h).status


def _group_stations_by_climate(station_ids: list[str], group_name: str | None = None) -> dict[str, list[str]]:
    """Group stations by climate similarity (latitude/longitude zones).
    
    Args:
        station_ids: List of station IDs to group
        group_name: Optional group name filter ('north', 'central', 'northeast', 'south')
        
    Returns:
        Dict mapping group name to list of station IDs
    """
    from app.data.stations import STATIONS
    
    # Define climate zones for Thailand
    climate_zones = {
        "north": ["CNX_01", "HKT_01"],
        "central": ["BKK_01", "LPT_01", "NMA_01"],
        "northeast": ["KKN_01", "UBN_01", "UDN_01", "NST_01", "NSW_01"],
        "south": ["RYG_01", "SPB_01"],
        "east": ["CEI_01", "CBI_01"],
        "north_east": ["HYI_01", "JTI_01"],
    }
    
    # If group_name specified, return only that group
    if group_name:
        return {group_name: climate_zones.get(group_name, [])}
    
    # Otherwise, return all groups
    return climate_zones


def _merge_station_data(station_ids: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Merge observation data from multiple stations for global training.
    
    Args:
        station_ids: List of station IDs to merge
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Combined DataFrame with station_id column
    """
    all_obs = []
    for sid in station_ids:
        try:
            obs = read_observations(
                station_id=sid,
                start_date=start_date,
                end_date=end_date,
            )
            if not obs.empty:
                obs["station_id"] = sid
                all_obs.append(obs)
        except Exception as e:
            logger.warning("Failed to read observations for station=%s: %s", sid, e)
    
    if not all_obs:
        return pd.DataFrame()
    
    return pd.concat(all_obs, ignore_index=True)


def _status_rank(status: str) -> int:
    order = {"not_ready": 0, "candidate": 1, "ready": 2}
    return order.get(status, -1)


def _row_from_existing_slot(slot_dir: Path, station_id: str, horizon_h: int) -> dict:
    """Build a leaderboard row from existing v3 artifacts for resume mode."""
    bundle_path = slot_dir / "bundle.json"
    registry_path = slot_dir / "registry.json"
    backend = "unknown"
    status = "skipped_existing"
    mae = float("nan")
    skill = float("nan")
    danger_recall = float("nan")
    eval_dir = ""

    if bundle_path.exists():
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            backend = bundle.get("backend_name", backend)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.debug("Failed to read bundle.json: %s", e)

    if registry_path.exists():
        try:
            registry_meta = json.loads(registry_path.read_text(encoding="utf-8"))
            backend = registry_meta.get("backend_name", backend)
            status = registry_meta.get("status", status)
            metrics = registry_meta.get("evaluation", {})
            mae = metrics.get("regression", {}).get("mae", mae)
            skill = metrics.get("baselines", {}).get("skill_score", skill)
            danger_recall = metrics.get("safety", {}).get("danger_42", {}).get("recall", danger_recall)
            eval_dir = metrics.get("paths", {}).get("out_dir", eval_dir)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Failed to parse registry metadata: %s", e)

    return {
        "backend": backend,
        "station": station_id,
        "horizon_h": horizon_h,
        "mae": float(mae) if mae is not None else float("nan"),
        "skill_score": float(skill) if skill is not None else float("nan"),
        "danger_recall_42": float(danger_recall) if danger_recall is not None else float("nan"),
        "danger_recall_40": float("nan"),  # not available for existing slots
        "status": status,
        "eval_dir": eval_dir,
    }


_EVAL_REPORT_NAME = "report.pdf"

def _next_version_dir() -> Path:
    """Return the next logs/eval/v{n}/ folder that does not yet exist."""
    base = Path("logs") / "eval"
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name[1:])
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]
    n = (max(existing) + 1) if existing else 1
    return base / f"v{n}"


def _export_run_to_versioned_dir(run_dir: Path, primary_station: str = "BKK_01", primary_h: int = 24) -> Path:
    """Copy primary slot PDF report into a new logs/eval/v{n}/ folder.

    Structure written:
      logs/eval/v{n}/report.pdf
      logs/eval/v{n}/summary.png
      logs/eval/v{n}/leaderboard.md
      logs/eval/v{n}/leaderboard.json

    The report comes from the primary slot (primary_station/h{primary_h}).
    Falls back to the first available slot if primary is missing.
    """
    import shutil

    dest = _next_version_dir()
    dest.mkdir(parents=True, exist_ok=True)

    # Locate primary slot dir: run_dir/{backend}/{station}/h{H}/
    primary_dir: Path | None = None
    for candidate in sorted(run_dir.rglob(f"h{primary_h}")):
        if candidate.is_dir() and candidate.parent.name == primary_station:
            primary_dir = candidate
            break
    if primary_dir is None:
        hits = sorted(run_dir.rglob(_EVAL_REPORT_NAME))
        if hits:
            primary_dir = hits[0].parent

    # Copy primary report flat into dest
    if primary_dir:
        src = primary_dir / _EVAL_REPORT_NAME
        if src.exists():
            shutil.copy2(src, dest / _EVAL_REPORT_NAME)

    # Copy summary + leaderboard
    for fname in ("summary.png", "leaderboard.md", "leaderboard.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)

    logger.info("Run exported to %s", dest.resolve())
    return dest


def _display_eval_charts(run_dir: Path) -> None:
    """Print all eval PDF report paths."""
    found = sorted(run_dir.rglob(_EVAL_REPORT_NAME))
    if not found:
        return

    logger.info("=" * 60)
    logger.info("POST-TRAINING EVAL REPORTS (%d files)", len(found))
    logger.info("=" * 60)
    prev_slot = None
    for p in found:
        slot = str(p.parent.relative_to(run_dir))
        if slot != prev_slot:
            logger.info("")
            logger.info("[%s]", slot)
            prev_slot = slot
        logger.info("  EVAL_REPORT: %s", p.resolve())

    logger.info("")
    logger.info("=" * 60)

    if sys.platform == "win32":
        for p in found:
            try:
                os.startfile(str(p))
            except OSError:
                pass


def _generate_summary_chart(run_dir: Path, rows: list[dict]) -> Path:
    """Create a single-page summary: MAE / skill / danger-recall heatmaps across station × horizon."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    stations = sorted({r["station"] for r in rows})
    horizons = sorted({r["horizon_h"] for r in rows})

    def _grid(metric: str, default=float("nan")) -> np.ndarray:
        def _safe(v):
            return float("nan") if v is None else float(v)
        lookup = {(r["station"], r["horizon_h"]): _safe(r.get(metric, default)) for r in rows}
        return np.array([[lookup.get((s, h), float("nan")) for h in horizons] for s in stations], dtype=float)

    mae_grid = _grid("mae")
    skill_grid = _grid("skill_score")
    danger_grid = _grid("danger_recall_42")
    danger40_grid = _grid("danger_recall_40")

    fig, axes = plt.subplots(1, 4, figsize=(20, max(3, len(stations) * 1.2 + 2)))
    fig.suptitle(f"Training Run Summary  —  {run_dir.name}", fontsize=13, fontweight="bold")

    h_labels = [f"h{h}" for h in horizons]

    def _heatmap(ax, data, title, cmap, vmin, vmax, fmt=".2f", bad_color="#cccccc"):
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(horizons))); ax.set_xticklabels(h_labels, fontsize=9)
        ax.set_yticks(range(len(stations))); ax.set_yticklabels(stations, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(stations)):
            for j in range(len(horizons)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                            color="white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.2 else "black")

    _heatmap(axes[0], mae_grid,     "MAE (°C)\nlow = good",       "RdYlGn_r", 1.0, 4.0)
    _heatmap(axes[1], skill_grid,   "Skill Score\nhigh = good",   "RdYlGn",   0.0, 1.0)
    _heatmap(axes[2], danger_grid,  "Danger Recall ≥42°C\nhigh = good", "RdYlGn", 0.0, 1.0)
    _heatmap(axes[3], danger40_grid, "Danger Recall ≥40°C\nhigh = good", "RdYlGn", 0.0, 1.0)

    plt.tight_layout()
    out = run_dir / "summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def _write_leaderboard(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "leaderboard.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(json_tmp, json_path)
    lines = [
        "| Backend | Station | Horizon | MAE | Skill | Danger Recall >=42C | Danger Recall >=40C | Status |",
        "|---|---|---:|---:|---:|---:|---|",  
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['station']} | {row['horizon_h']} | "
            f"{row.get('mae', float('nan')):.3f} | {row.get('skill_score', float('nan')):.3f} | "
            f"{(row['danger_recall_42'] if row.get('danger_recall_42') is not None else float('nan')):.3f} | "
            f"{(row['danger_recall_40'] if row.get('danger_recall_40') is not None else float('nan')):.3f} | {row['status']} |"
        )
    md_path = run_dir / "leaderboard.md"
    md_tmp = md_path.with_suffix(".md.tmp")
    md_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(md_tmp, md_path)


def _extract_feature_importance(forecaster) -> dict[str, float] | None:
    """Try to extract feature importance from the forecaster's internal boosters."""
    try:
        boosters = getattr(forecaster, "_boosters", {})
        features = getattr(forecaster, "feature_list", [])
        if not boosters or not features:
            return None

        flat = []
        for v in boosters.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, list):
                        flat.extend(item)
                    else:
                        flat.append(item)

        for b in flat:
            if hasattr(b, "feature_importance"):
                imp = b.feature_importance(importance_type="gain")
                return {f: float(i) for f, i in zip(features, imp)}
            if hasattr(b, "get_score"):
                scores = b.get_score(importance_type="gain")
                mapped = {}
                for k, v in scores.items():
                    if k.startswith("f") and k[1:].isdigit():
                        idx = int(k[1:])
                        if idx < len(features):
                            mapped[features[idx]] = float(v)
                    elif k in features:
                        mapped[k] = float(v)
                return mapped if mapped else None
        return None
    except Exception:
        return None


def _read_champion_mae(slot_dir: Path) -> float | None:
    """Return the existing slot's test MAE from registry.json, or None if no slot exists."""
    registry_path = slot_dir / "registry.json"
    if not registry_path.exists():
        return None
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        mae = data.get("evaluation", {}).get("regression", {}).get("mae")
        return float(mae) if mae is not None else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeatShield AI forecast model")
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Legacy single-horizon flag in hours. If set, overrides --horizons and uses single-horizon save.",
    )
    parser.add_argument(
        "--horizons", type=str, default="6,12,24,48,72",
        help="Comma-separated forecast horizons in hours (default: 6,12,24,48,72)",
    )
    parser.add_argument("--station", type=str, default=None, help="Single station to train on (default: all)")
    parser.add_argument("--trials", type=int, default=30, help="Number of optuna trials (default: 30)")
    parser.add_argument("--quick-mode", action="store_true", help="Use reduced trial budget for faster iteration (20/20/20/15/15 trials)")
    parser.add_argument("--start", type=str, default=None, help="Data start date YYYY-MM-DD (default: 5 years ago)")
    parser.add_argument("--end", type=str, default=None, help="Data end date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--backend", type=str, default="xgboost_safety",
                        choices=["xgboost_safety", "xgboost", "lightgbm", "lightgbm_hi"],
                        help="Model backend to train (default: xgboost_safety)")
    parser.add_argument("--target-kind", type=str, default="th",
                        choices=["hi", "th"],
                        dest="target_kind",
                        help="Prediction target: hi=direct heat-index, th=temp+rh two-head (default: th)")
    parser.add_argument("--gate-backend", type=str, default="lightgbm",
                        choices=["lightgbm", "brf"],
                        dest="gate_backend",
                        help="DangerGate backend: lightgbm (default) or brf (BalancedRandomForest)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel station training workers (default: 1). "
                             "GPU training with workers>1 may cause device contention.")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "gpu", "cuda"],
                        help="Training device (default: cuda for NVIDIA GPU). Use 'cpu' to force CPU training.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for logs/eval/runs/<run_id>. Default: UTC timestamp.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain slots even when bundle already exists AND bypass champion-challenger (always save).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        dest="no_skip",
        help="Bypass the bundle-exists skip check but still respect champion-challenger (save only if MAE improves).",
    )
    parser.add_argument(
        "--model-version", type=str, default="v3",
        dest="model_version",
        help="Registry layout version under app/models/forecast_<ver>/ (default: v3).",
    )
    parser.add_argument(
        "--backends",
        default="lgbm",
        help="Comma-separated list of backends to train: lgbm,catboost (default: lgbm)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        dest="no_pdf",
        help="Skip PDF report generation during bulk training (faster for large runs).",
    )
    parser.add_argument(
        "--enhanced-features",
        action="store_true",
        dest="enhanced_features",
        help="Add 2nd/3rd diurnal harmonics and day-of-week sin/cos to the feature matrix.",
    )
    parser.add_argument(
        "--changepoint",
        action="store_true",
        dest="changepoint",
        help="Add PELT changepoint distance/regime features (requires: pip install ruptures).",
    )
    parser.add_argument(
        "--asinh",
        action="store_true",
        dest="asinh",
        help="(Reserved) Apply asinh target transform during training. Requires registry support.",
    )
    parser.add_argument(
        "--adaptive-conformal",
        action="store_true",
        dest="adaptive_conformal",
        help="Apply MAPIE ACI calibration to prediction intervals after base model fit (requires: pip install 'mapie>=0.9,<2').",
    )
    args = parser.parse_args()
    os.environ["HEATSHIELD_FORECAST_VERSION"] = args.model_version

    # Configure quick mode BEFORE device config so _get_median_seeds() reads the right env.
    if args.quick_mode:
        os.environ["HEATSHIELD_QUICK_MODE"] = "1"       # explicit quick mode signal
        os.environ["HEATSHIELD_CV_SPLITS"] = "1"        # signals quick mode to lgbm_backend
        os.environ["HEATSHIELD_EARLY_STOPPING_ROUNDS"] = "10"  # aggressive early stopping
        logger.info("Quick mode enabled: CV splits=1, early_stop=10, single seed, 10 trials")
    else:
        os.environ.setdefault("HEATSHIELD_CV_SPLITS", "2")
        os.environ.setdefault("HEATSHIELD_EARLY_STOPPING_ROUNDS", "20")

    _configure_training_device(args.device)

    end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=365 * 3)

    # Resolve horizons list — legacy --horizon overrides --horizons
    if args.horizon is not None:
        horizons = [args.horizon]
        legacy_single = True
        logger.info("Legacy --horizon mode: training single horizon=%dh", args.horizon)
    else:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        legacy_single = len(horizons) == 1
        logger.info("Multi-horizon mode: training horizons=%s", horizons)

    station_ids = [args.station] if args.station else list(STATIONS.keys())
    os.environ["HEATSHIELD_GLOBAL_TRAINING"] = "0"
    
    logger.info("Training for stations=%s trials=%d", station_ids, args.trials)
    try:
        from app.ml.forecast.train import _gpu_xgb_params
        logger.info("XGBoost device mode: %s", _gpu_xgb_params().get("device", "cpu"))
    except Exception:
        pass
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    eval_run_dir = Path("logs") / "eval" / "runs" / run_id
    leaderboard: list[dict] = []
    leaderboard_lock = Lock()
    os.environ["HEATSHIELD_RUN_ID"] = run_id
    write_heartbeat(run_id, phase="start", extra={"model_version": args.model_version})

    def _record_leaderboard_row(row: dict) -> None:
        with leaderboard_lock:
            leaderboard.append(row)
            _write_leaderboard(eval_run_dir, leaderboard)
            drive_root = os.environ.get("HEATSHIELD_DRIVE_ROOT")
            if drive_root:
                try:
                    import shutil
                    dst = Path(drive_root) / "logs" / "train" / run_id
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(eval_run_dir / "leaderboard.json", dst / "leaderboard.json")
                    shutil.copy2(eval_run_dir / "leaderboard.md", dst / "leaderboard.md")
                except Exception:
                    pass

    if args.backend in {"lightgbm", "lightgbm_hi", "xgboost", "xgboost_safety"}:
        if args.backend == "xgboost":
            from app.ml.forecast.backends.xgb_train_backend import XGBoostTrainForecaster
            from app.ml.forecast.backends.lgbm.utils import _compute_hi_array
            from app.ml import registry
            target_kind = "hi"
        elif args.backend == "xgboost_safety":
            from app.ml.forecast.backends.xgb_safety_backend import XGBoostSafetyForecaster
            from app.ml.forecast.backends.lgbm.utils import _compute_hi_array
            from app.ml import registry
            target_kind = "hi"
        else:
            from app.ml.forecast.backends.lgbm import LGBMDirectHIForecaster, LGBMForecaster
            from app.ml import registry
            target_kind = "hi" if args.backend == "lightgbm_hi" else args.target_kind
            from app.ml.forecast.backends.lgbm.utils import _compute_hi_array

        # Allow parallel station training even with GPU
        # Each worker process gets its own GPU context, avoiding memory contention
        if args.workers is None:
            _in_colab = (
                os.environ.get("COLAB_RELEASE_TAG") is not None
                or os.path.exists("/content")
                or os.environ.get("COLAB_GPU") is not None
            )
            _cores = os.cpu_count() or 1
            if _in_colab:
                # Colab VMs are small (2 vCPU typical); avoid oversubscription
                args.workers = min(2, len(station_ids))
                logger.info("Colab detected: capping workers to %d", args.workers)
            else:
                args.workers = min(6, len(station_ids), max(1, _cores - 1))
        logger.info("Parallel station training: workers=%d stations=%s", args.workers, station_ids)

        def _train_station(sid: str) -> list[dict]:
            """Train all horizons for one station; returns leaderboard rows."""
            rows: list[dict] = []
            # Enable hourly grid normalization for time-aware lag features
            obs = read_observations(sid, start_date, end_date, normalize_hourly_grid=True)
            if len(obs) < 500:
                logger.warning("Too few rows (%d) for station=%s — skipping", len(obs), sid)
                return rows

            # Calculate data quality metrics on raw observations BEFORE feature engineering
            # This ensures ts_utc and raw columns are preserved for accurate gap/outlier detection
            raw_quality_metrics = calculate_data_quality_metrics(obs)

            # Build the feature matrix ONCE with the union of all per-horizon
            # lags/rolling windows, then compute the correct y target PER HORIZON.
            #
            # The previous approach called build_features(horizon_h=max_horizon) which
            # produces y = shift(-max_horizon) and reused that same y for all shorter
            # horizons.  A h6 model trained with a 72h-ahead target performed far worse
            # than the naive persistence baseline (negative skill).  We now separate:
            #   1. build_X_once → X feature matrix + df_augmented (no target)
            #   2. build_y_for_horizon(h) → correct shift(-h) target per horizon
            all_lags = sorted({l for h in horizons for l in _get_lags_for_horizon(h)})
            all_rolls = sorted({w for h in horizons for w in _get_rolling_for_horizon(h)})
            logger.info(
                "Building features once (union lags=%s rolls=%s) for station=%s",
                all_lags, all_rolls, sid,
            )

 
            X_base, df_augmented = build_X_once(obs, all_lags, all_rolls)

            if args.enhanced_features:
                _ts = pd.to_datetime(df_augmented["ts_utc"].loc[X_base.index])
                _hour = _ts.dt.hour.values
                _doy = _ts.dt.dayofyear.values
                _dow = _ts.dt.dayofweek.values
                X_base = X_base.copy()
                X_base["hour_sin_2"] = np.sin(4 * np.pi * _hour / 24)
                X_base["hour_cos_2"] = np.cos(4 * np.pi * _hour / 24)
                X_base["hour_sin_3"] = np.sin(6 * np.pi * _hour / 24)
                X_base["hour_cos_3"] = np.cos(6 * np.pi * _hour / 24)
                X_base["doy_sin_2"] = np.sin(4 * np.pi * _doy / 365)
                X_base["doy_cos_2"] = np.cos(4 * np.pi * _doy / 365)
                X_base["dow_sin"] = np.sin(2 * np.pi * _dow / 7)
                X_base["dow_cos"] = np.cos(2 * np.pi * _dow / 7)
                logger.info("Added 8 enhanced temporal harmonic columns to X_base for station=%s", sid)

            for h in horizons:
                current_slot = f"{sid}:h{h}"
                write_heartbeat(run_id, current_slot=current_slot, phase="slot_start")
                slot_dir = Path("app") / "models" / f"forecast_{args.model_version}" / sid / f"h{h}"
                bundle_path = slot_dir / "bundle.json"
                registry_path = slot_dir / "registry.json"
                if not args.force and not args.no_skip and bundle_path.exists() and registry_path.exists():
                    logger.warning(
                        "SKIP station=%s h=%d — bundle already exists. Pass --force or --no-skip to retrain. (%s)",
                        sid,
                        h,
                        slot_dir,
                    )
                    row = _row_from_existing_slot(slot_dir, sid, h)
                    rows.append(row)
                    _record_leaderboard_row(row)
                    write_heartbeat(run_id, current_slot=current_slot, phase="skipped_existing")
                    continue

                backend_name = "XGBoost" if args.backend == "xgboost" else "LightGBM"
                logger.info("Training %s backend: station=%s horizon=%d target=%s",
                            backend_name, sid, h, target_kind)

                # Build the correct y for this specific horizon (shift by h, not max_h).
                # NaN rows differ per horizon (last h rows of the series have no target),
                # so we filter after building y.
                y_raw = build_y_for_horizon(df_augmented, X_base.index, h, target_kind)
                _valid_mask = (
                    y_raw.notna().all(axis=1)
                    if isinstance(y_raw, pd.DataFrame)
                    else y_raw.notna()
                )
                X_h = X_base[_valid_mask].reset_index(drop=True)
                y_full_h = y_raw[_valid_mask].reset_index(drop=True)

                # Subset X columns to the lag/rolling set appropriate for this horizon
                lags_h = _get_lags_for_horizon(h)
                rolling_h = _get_rolling_for_horizon(h)
                X, y = _subset_features_for_horizon(X_h, y_full_h, h, lags_h, rolling_h)

                # Apply feature pruning if available from previous stations
                if _pruned_features_cache:
                    # Use union of all pruned features from previous stations
                    all_pruned = set()
                    for pruned in _pruned_features_cache.values():
                        all_pruned.update(pruned)
                    intersect = [c for c in X.columns if c in all_pruned]
                    if intersect:
                        X = X[intersect]

                if len(X) < 500:
                    logger.warning(
                        "Too few rows (%d) after target NaN-drop for station=%s h=%d — skipping",
                        len(X), sid, h,
                    )
                    continue
                if target_kind == "th":
                    y_hi_for_audit = _compute_hi_array(y["temp_c"].values, y["rh"].values)
                else:
                    y_hi_for_audit = y.values
                danger_support = int((y_hi_for_audit >= 42.0).sum())
                if danger_support < 20:
                    logger.warning(
                        "Low danger support for station=%s h=%d: only %d rows with HI>=42C",
                        sid, h, danger_support,
                    )

                trials = _adaptive_trials(h, args.trials, args.quick_mode)
                logger.info("Adaptive trials for station=%s h=%d: %d", sid, h, trials)

                with edr.training_run(station=sid, horizon=h, backend=args.backend) as run:
                    # Log data quality metrics for full dataset (computed from raw observations)
                    run.log_data_quality(**raw_quality_metrics)

                    if args.backend == "xgboost":
                        forecaster = XGBoostTrainForecaster(
                            n_trials=trials,
                            random_state=42,
                        )
                    elif args.backend == "xgboost_safety":
                        forecaster = XGBoostSafetyForecaster(
                            n_trials=trials,
                            random_state=42,
                            gate_backend=args.gate_backend,
                        )
                    else:
                        forecaster_cls = LGBMDirectHIForecaster if args.backend == "lightgbm_hi" else LGBMForecaster
                        forecaster = forecaster_cls(
                            n_trials=trials,
                            gate_backend=args.gate_backend,
                        )
                    write_heartbeat(run_id, current_slot=current_slot, phase="fit")
                    forecaster.fit(X, y, station_id=sid, horizon_h=h)

                    split = split_xy(X, y, horizon_h=h)
                    # Note: Split-level data quality metrics are not logged separately
                    # because they would be computed on feature matrices without ts_utc/raw columns
                    # The full dataset metrics (computed from raw observations) are logged above
                    bundle = forecaster.predict_with_pi(split.X_test)
                    if target_kind == "th":
                        y_true = _compute_hi_array(split.y_test["temp_c"].values, split.y_test["rh"].values)
                    else:
                        y_true = split.y_test.values
                    out_dir = eval_run_dir / forecaster.backend_name / sid / f"h{h}"
                    best_params = getattr(forecaster, "best_params_", None) or getattr(forecaster, "_best_params", None) or {}
                    feat_imp = _extract_feature_importance(forecaster)
                    eval_metrics = evaluate_predictions(
                        y_true,
                        bundle.hi_mean,
                        split.X_test,
                        out_dir=out_dir,
                        horizon_h=h,
                        station_labels={0: sid},
                        q05=bundle.hi_lower,
                        q95=bundle.hi_upper,
                        runtime={"train_seconds": forecaster._metadata.get("train_seconds", 0.0)},
                        split_metadata=split.metadata,
                        station=sid,
                        backend=forecaster.backend_name,
                        hyperparams=best_params,
                        feature_importance=feat_imp,
                        model_version=args.model_version,
                        run_id=run_id,
                        eval_dpi=80,
                    )
                    write_heartbeat(run_id, current_slot=current_slot, phase="eval")
                    readiness = evaluate_readiness(eval_metrics, h)
                    status = readiness.status

                    # Feature pruning: after first horizon of first station, extract importance & prune
                    if sid not in _pruned_features_cache and hasattr(forecaster, '_boosters'):
                        try:
                            booster = forecaster._boosters['temp_c'][1][0]
                            if booster is not None:
                                importances = booster.feature_importance(importance_type='gain')
                                sorted_idx = np.argsort(importances)[::-1]
                                cumsum = np.cumsum(importances[sorted_idx]) / importances.sum()
                                top_n = min(60, int((cumsum < 0.95).sum()) + 1)
                                pruned = [X.columns[i] for i in sorted_idx[:top_n]]
                                # Always include required features (temp_c, rh, etc.)
                                _required = ['temp_c', 'rh', 'heat_index_c', 'station_enc', 'lat', 'lon']
                                for col in _required:
                                    if col not in pruned and col in X.columns:
                                        pruned.append(col)
                                _pruned_features_cache[sid] = pruned
                                logger.info("Feature pruning: %d->%d features for station=%s",
                                             len(X.columns), len(pruned), sid)
                        except Exception:
                            pass

                    metrics = {
                        "run_id": run_id,
                        "status": status,
                        "n_train_rows": split.metadata["row_counts"]["train"],
                        "horizons": horizons,
                        "evaluation": eval_metrics,
                    }
                    new_mae = eval_metrics.get("regression", {}).get("mae", float("inf"))
                    champion_mae = _read_champion_mae(slot_dir)
                    champion_kept = False
                    challenger_wins = True
                    champion_status = None
                    if champion_mae is not None and (slot_dir / "registry.json").exists():
                        try:
                            champ_data = json.loads((slot_dir / "registry.json").read_text(encoding="utf-8"))
                            champion_eval = champ_data.get("evaluation", {})
                            champion_status = evaluate_readiness(champion_eval, h).status
                        except Exception:
                            champion_status = None

                    if not args.force and champion_mae is not None:
                        if champion_status is not None:
                            better_status = _status_rank(status) > _status_rank(champion_status)
                            same_status = _status_rank(status) == _status_rank(champion_status)
                            better_mae = (champion_mae - new_mae) >= 0.005
                            challenger_wins = better_status or (same_status and better_mae)
                        else:
                            challenger_wins = (champion_mae - new_mae) >= 0.005

                    if not args.force and champion_mae is not None and not challenger_wins:
                        logger.info(
                            "Champion-challenger: keeping existing model (champion MAE=%.4f, challenger MAE=%.4f, "
                            "champion_status=%s, challenger_status=%s) "
                            "for station=%s h=%d",
                            champion_mae, new_mae, champion_status, status, sid, h,
                        )
                        champion_kept = True
                        run.log_champion_result(won=False, mae_delta=new_mae - champion_mae, reason="mae_not_better")
                    else:
                        registry.save_model_v3(forecaster, metrics, sid, h)
                        logger.info(
                            "Champion-challenger: saved challenger (MAE=%.4f%s) for station=%s h=%d",
                            new_mae,
                            f" < champion {champion_mae:.4f}" if champion_mae is not None else " — new slot",
                            sid, h,
                        )
                        mae_delta = (new_mae - champion_mae) if champion_mae is not None else 0.0
                        run.log_champion_result(won=True, mae_delta=mae_delta)
                        run.log_model_artifact(str(slot_dir))

                        # Fit calibration using select_calibration() — auto-picks best
                        # level L1-L4 based on validation MAE (not hardcoded L2).
                        try:
                            from app.ml.calibration import select_calibration as _select_cal
                            _bundle_val = forecaster.predict_with_pi(split.X_val)
                            _bundle_train = forecaster.predict_with_pi(split.X_train)
                            if target_kind == "th":
                                _y_val_hi = _compute_hi_array(
                                    split.y_val["temp_c"].values, split.y_val["rh"].values
                                )
                                _y_train_hi = _compute_hi_array(
                                    split.y_train["temp_c"].values, split.y_train["rh"].values
                                )
                            else:
                                _y_val_hi = split.y_val.values
                                _y_train_hi = split.y_train.values
                            _n_val = len(_y_val_hi)
                            _n_train = len(_y_train_hi)
                            _val_hours = _extract_hour_from_X(split.X_val, _n_val)
                            _train_hours = _extract_hour_from_X(split.X_train, _n_train)
                            _calib = _select_cal(
                                y_true_train=_y_train_hi,
                                y_pred_train=_bundle_train.hi_mean,
                                station_ids_train=[sid] * _n_train,
                                horizons_train=[h] * _n_train,
                                hours_train=_train_hours,
                                y_true_val=_y_val_hi,
                                y_pred_val=_bundle_val.hi_mean,
                                station_ids_val=[sid] * _n_val,
                                horizons_val=[h] * _n_val,
                                hours_val=_val_hours,
                            )
                            _calib_path = slot_dir / "calibration.json"
                            _calib_path.write_text(
                                json.dumps(_calib.to_json(), ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            _bias_desc = f"{_calib.level} shift={_calib.shifts.get(f'{sid}|{h}', 0.0):.3f}C" if _calib.shifts else f"{_calib.level} isotonic"
                            logger.info(
                                "Calibration saved: station=%s h=%d %s",
                                sid, h, _bias_desc,
                            )
                        except Exception as _calib_exc:
                            logger.warning(
                                "Could not fit bias calibration for station=%s h=%d: %s",
                                sid, h, _calib_exc,
                            )

                    # Log key metrics to EDR
                    run.log_metric("mae", new_mae)
                    skill = eval_metrics.get("baselines", {}).get("skill_score", float("nan"))
                    if not (skill != skill):  # not NaN
                        run.log_metric("skill_score", skill)
                    danger_recall = eval_metrics.get("safety", {}).get("danger_42", {}).get("recall")
                    if danger_recall is not None:
                        run.log_metric("danger_recall_42", danger_recall)
                    if hasattr(forecaster, "best_params_") and forecaster.best_params_:
                        run.log_hyperparams(forecaster.best_params_)

                    # When champion is kept, surface its stored metrics so the leaderboard
                    # reflects the model that is actually on disk, not the rejected challenger.
                    row_eval = eval_metrics
                    row_status = status
                    if champion_kept:
                        try:
                            champ_data = json.loads((slot_dir / "registry.json").read_text(encoding="utf-8"))
                            row_eval = champ_data.get("evaluation", eval_metrics)
                            row_status = champ_data.get("status", status)
                        except Exception:
                            pass

                    row = {
                        "backend": forecaster.backend_name,
                        "station": sid,
                        "horizon_h": h,
                        "mae": row_eval.get("regression", {}).get("mae",
                                 eval_metrics.get("regression", {}).get("mae", float("nan"))),
                        "skill_score": row_eval.get("baselines", {}).get("skill_score",
                                         eval_metrics.get("baselines", {}).get("skill_score", float("nan"))),
                        "danger_recall_42": row_eval.get("safety", {}).get("danger_42", {}).get("recall",
                                              eval_metrics.get("safety", {}).get("danger_42", {}).get("recall")),
                        "danger_recall_40": row_eval.get("safety", {}).get("danger_40", {}).get("recall",
                                              eval_metrics.get("safety", {}).get("danger_40", {}).get("recall")),
                        "status": row_status,
                        "eval_dir": str(out_dir),
                        **({"champion_kept": True} if champion_kept else {}),
                    }
                    rows.append(row)
                    _record_leaderboard_row(row)
                    write_heartbeat(run_id, current_slot=current_slot, phase="slot_done")
                    logger.info("Saved %s v3: station=%s h=%d", backend_name, sid, h)
                    del forecaster, bundle, split
                    gc.collect()

                # Optional CatBoost training when --backends includes "catboost".
                # Trained head-to-head against the LGBM champion; the better backend wins
                # the slot and updates choice_matrix.json.
                backends_to_train = [b.strip() for b in getattr(args, "backends", "lgbm").split(",")]
                if "catboost" in backends_to_train:
                    try:
                        from app.ml.forecast.backends.catboost_backend import CatBoostForecaster
                        logger.info("Training CatBoost backend: station=%s horizon=%d", sid, h)
                        if target_kind == "th":
                            y_cb = pd.Series(
                                _compute_hi_array(y["temp_c"].values, y["rh"].values),
                                index=y.index, name="heat_index_c",
                            )
                        else:
                            y_cb = y
                        cb_forecaster = CatBoostForecaster(
                            n_trials=getattr(args, "trials", 10),
                            random_state=42,
                        )
                        cb_forecaster.fit(X, y_cb, station_id=sid, horizon_h=h)
                        cb_split = split_xy(X, y_cb, horizon_h=h)
                        cb_bundle = cb_forecaster.predict_with_pi(cb_split.X_test)
                        cb_y_true = cb_split.y_test.values
                        cb_out_dir = eval_run_dir / cb_forecaster.backend_name / sid / f"h{h}"
                        cb_best_params = getattr(cb_forecaster, "best_params_", None) or getattr(cb_forecaster, "_best_params", None) or {}
                        cb_feat_imp = _extract_feature_importance(cb_forecaster)
                        cb_eval = evaluate_predictions(
                            cb_y_true,
                            cb_bundle.hi_mean,
                            cb_split.X_test,
                            out_dir=cb_out_dir,
                            horizon_h=h,
                            station_labels={0: sid},
                            q05=cb_bundle.hi_lower,
                            q95=cb_bundle.hi_upper,
                            runtime={"train_seconds": cb_forecaster._metadata.get("train_seconds", 0.0)},
                            split_metadata=cb_split.metadata,
                            station=sid,
                            backend=cb_forecaster.backend_name,
                            hyperparams=cb_best_params,
                            feature_importance=cb_feat_imp,
                            model_version=args.model_version,
                            run_id=run_id,
                            eval_dpi=80,
                        )
                        cb_readiness = evaluate_readiness(cb_eval, h)
                        cb_status = cb_readiness.status
                        cb_metrics = {
                            "run_id": run_id,
                            "status": cb_status,
                            "n_train_rows": cb_split.metadata["row_counts"]["train"],
                            "horizons": horizons,
                            "evaluation": cb_eval,
                        }
                        cb_new_mae = cb_eval.get("regression", {}).get("mae", float("inf"))
                        cb_champion_mae = _read_champion_mae(slot_dir)
                        cb_champion_kept = False
                        cb_challenger_wins = True
                        cb_champion_status = None
                        if cb_champion_mae is not None and (slot_dir / "registry.json").exists():
                            try:
                                champ_data = json.loads((slot_dir / "registry.json").read_text(encoding="utf-8"))
                                champ_eval = champ_data.get("evaluation", {})
                                cb_champion_status = evaluate_readiness(champ_eval, h).status
                            except Exception:
                                cb_champion_status = None

                        if not args.force and cb_champion_mae is not None:
                            if cb_champion_status is not None:
                                better_status = _status_rank(cb_status) > _status_rank(cb_champion_status)
                                same_status = _status_rank(cb_status) == _status_rank(cb_champion_status)
                                better_mae = (cb_champion_mae - cb_new_mae) >= 0.005
                                cb_challenger_wins = better_status or (same_status and better_mae)
                            else:
                                cb_challenger_wins = (cb_champion_mae - cb_new_mae) >= 0.005

                        if not args.force and cb_champion_mae is not None and not cb_challenger_wins:
                            logger.info(
                                "CatBoost champion-challenger: keeping existing model "
                                "(champion MAE=%.4f, challenger MAE=%.4f, champion_status=%s, challenger_status=%s) "
                                "for station=%s h=%d",
                                cb_champion_mae, cb_new_mae, cb_champion_status, cb_status, sid, h,
                            )
                            cb_champion_kept = True
                        else:
                            registry.save_model_v3(cb_forecaster, cb_metrics, sid, h)
                            logger.info(
                                "CatBoost saved challenger (MAE=%.4f%s) for station=%s h=%d",
                                cb_new_mae,
                                f" < champion {cb_champion_mae:.4f}" if cb_champion_mae is not None else " — new slot",
                                sid, h,
                            )

                        cb_row_eval = cb_eval
                        cb_row_status = cb_status
                        if cb_champion_kept:
                            try:
                                champ_data = json.loads((slot_dir / "registry.json").read_text(encoding="utf-8"))
                                cb_row_eval = champ_data.get("evaluation", cb_eval)
                                cb_row_status = champ_data.get("status", cb_status)
                            except Exception:
                                pass

                        cb_row = {
                            "backend": cb_forecaster.backend_name,
                            "station": sid,
                            "horizon_h": h,
                            "mae": cb_row_eval.get("regression", {}).get("mae",
                                     cb_eval.get("regression", {}).get("mae", float("nan"))),
                            "skill_score": cb_row_eval.get("baselines", {}).get("skill_score",
                                             cb_eval.get("baselines", {}).get("skill_score", float("nan"))),
                            "danger_recall_42": cb_row_eval.get("safety", {}).get("danger_42", {}).get("recall",
                                                  cb_eval.get("safety", {}).get("danger_42", {}).get("recall")),
                            "danger_recall_40": cb_row_eval.get("safety", {}).get("danger_40", {}).get("recall",
                                                  cb_eval.get("safety", {}).get("danger_40", {}).get("recall")),
                            "status": cb_row_status,
                            "eval_dir": str(cb_out_dir),
                            **({"champion_kept": True} if cb_champion_kept else {}),
                        }
                        rows.append(cb_row)
                        _record_leaderboard_row(cb_row)
                        logger.info("CatBoost v3 training done: station=%s h=%d", sid, h)
                        del cb_forecaster, cb_bundle, cb_split, y_cb
                        gc.collect()
                    except Exception as exc:
                        logger.warning("CatBoost training failed (%s) — skipping", exc)
            return rows

        if args.workers > 1:
            logger.info("Parallel station training: workers=%d stations=%s", args.workers, station_ids)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_train_station, sid): sid for sid in station_ids}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error("Station %s training failed: %s", futures[fut], exc)
        else:
            for sid in station_ids:
                _train_station(sid)

        _write_leaderboard(eval_run_dir, leaderboard)
        logger.info("LightGBM v3 training complete")
        try:
            summary_path = _generate_summary_chart(eval_run_dir, leaderboard)
            logger.info("Summary chart: %s", summary_path.resolve())
        except Exception as e:
            logger.warning("Summary chart failed: %s", e)
        try:
            _export_run_to_versioned_dir(eval_run_dir)
        except Exception as e:
            logger.warning("versioned export failed: %s", e)
        if not args.no_pdf:
            _display_eval_charts(eval_run_dir)

        return  # don't fall through to XGBoost path

    # Load and concatenate observations from all stations
    frames: list[pd.DataFrame] = []
    for sid in station_ids:
        df = read_observations(sid, start_date, end_date)
        if df.empty:
            logger.warning("No data found for station=%s — skipping", sid)
            continue
        df["station_id"] = sid
        frames.append(df)

    if not frames:
        logger.error("No data loaded. Run scripts/build_historical.py first to ingest data.")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d total observations across %d stations", len(all_data), len(frames))

    # ------------------------------------------------------------------ #
    # Train each horizon
    # ------------------------------------------------------------------ #
    all_horizon_results: dict[int, dict] = {}
    for horizon_h in horizons:
        write_heartbeat(run_id, current_slot=f"all:h{horizon_h}", phase="slot_start")
        logger.info("=" * 50)
        logger.info("Training horizon = %dh", horizon_h)

        X, y = build_features(
            all_data.copy(),
            horizon_h=horizon_h,
        )
        logger.info("Feature matrix shape for h%d: %s", horizon_h, X.shape)

        if len(X) < 500:
            logger.error(
                "Only %d training rows for horizon=%dh after feature engineering. "
                "Need at least 500. Ingest more historical data first.",
                len(X), horizon_h,
            )
            sys.exit(1)

        result = train(X, y, station_id=",".join(station_ids), horizon_h=horizon_h, n_trials=args.trials)
        write_heartbeat(run_id, current_slot=f"all:h{horizon_h}", phase="eval")
        all_horizon_results[horizon_h] = result
        logger.info("h%d metrics: %s", horizon_h, _fmt_metrics(result["metrics"]))
        preds = result["test_predictions"]
        test_data = result["test_data"]
        out_dir = eval_run_dir / "xgboost_hi" / "all" / f"h{horizon_h}"
        xgb_best_params = result.get("hyperparams", {})
        eval_metrics = evaluate_predictions(
            test_data["y"].values,
            preds["mean"],
            test_data["X"],
            out_dir=out_dir,
            horizon_h=horizon_h,
            station_labels={i: sid for i, sid in enumerate(sorted(station_ids))},
            q05=preds.get("q05"),
            q95=preds.get("q95"),
            runtime={"train_seconds": result["training"]["train_seconds"]},
            split_metadata=result["split_metadata"],
            station="all",
            backend="XGBoost",
            hyperparams=xgb_best_params,
            feature_importance=None,
            model_version=args.model_version,
            run_id=run_id,
            eval_dpi=80,
        )
        status = _quality_status(eval_metrics, horizon_h=horizon_h)
        result["evaluation"] = eval_metrics
        result["status"] = status
        row = {
            "backend": "xgboost_hi",
            "station": "all",
            "horizon_h": horizon_h,
            "mae": eval_metrics["regression"]["mae"],
            "skill_score": eval_metrics["baselines"]["skill_score"],
            "danger_recall_42": eval_metrics["safety"]["danger_42"]["recall"],
            "danger_recall_40": eval_metrics["safety"]["danger_40"]["recall"],
            "status": status,
            "eval_dir": str(out_dir),
        }
        _record_leaderboard_row(row)
        write_heartbeat(run_id, current_slot=f"all:h{horizon_h}", phase="slot_done")
        gc.collect()

    logger.info("=" * 50)

    # ------------------------------------------------------------------ #
    # Save model
    # ------------------------------------------------------------------ #
    if legacy_single:
        # Single-horizon save (original v2 single-horizon behavior)
        only_h = horizons[0]
        result = all_horizon_results[only_h]
        feature_medians = result["feature_medians"]
        metadata = {
            "horizon_h": only_h,
            "stations": station_ids,
            "data_window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "n_train_rows": len(X),
            "feature_list": result["feature_list"],
            "feature_medians": feature_medians,
            "hyperparams": result["hyperparams"],
            "metrics": result["metrics"],
            "evaluation": result["evaluation"],
            "status": result["status"],
            "git_sha": _git_sha(),
            "roles": ["mean", "q05", "q95"],
            "seeds": [42, 123, 7],
        }
        ensemble_payload = {"all_boosters": result["all_boosters"]}
        model_path = save_model(ensemble_payload, metadata, kind="forecast")
    else:
        # Multi-horizon save
        horizons_payload = {
            "horizons": {
                h: {"all_boosters": all_horizon_results[h]["all_boosters"]}
                for h in horizons
            }
        }

        # Use h24 as primary for backward compat, fall back to first horizon
        h24_result = all_horizon_results.get(24, next(iter(all_horizon_results.values())))
        primary_h = 24 if 24 in all_horizon_results else horizons[0]

        metadata = {
            "horizon_h": primary_h,
            "horizons": horizons,
            "stations": station_ids,
            "data_window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "n_train_rows": sum(len(all_data) for _ in horizons),  # approximate
            "feature_list": h24_result["feature_list"],
            "feature_medians": h24_result["feature_medians"],
            "hyperparams": h24_result["hyperparams"],
            "metrics": h24_result["metrics"],
            "evaluation": h24_result["evaluation"],
            "status": h24_result["status"],
            "git_sha": _git_sha(),
            "roles": ["mean", "q05", "q95"],
            "seeds": [42, 123, 7],
            "per_horizon_metrics": {h: all_horizon_results[h]["metrics"] for h in horizons},
            "per_horizon_evaluation": {h: all_horizon_results[h]["evaluation"] for h in horizons},
        }
        model_path = save_model(horizons_payload, metadata, kind="forecast")

    logger.info("Model saved to %s", model_path)
    _write_leaderboard(eval_run_dir, leaderboard)
    logger.info("Evaluation artifacts saved to %s", eval_run_dir)

    # ------------------------------------------------------------------ #
    # Final quality gate (use primary horizon metrics)
    # ------------------------------------------------------------------ #
    primary_result = all_horizon_results.get(primary_h, next(iter(all_horizon_results.values())))
    skill = primary_result["metrics"]["skill_score"]
    if skill >= 0.15:
        logger.info("skill_score=%.3f >= 0.15 — model is READY to ship.", skill)
    else:
        logger.warning(
            "skill_score=%.3f < 0.15 — model does NOT beat baselines enough. "
            "Do not use in production. Ingest more data or tune features.",
            skill,
        )


def run(argv: list[str]) -> None:
    """Call main() with explicit argv — allows notebook to avoid subprocess overhead."""
    _saved = sys.argv[:]
    sys.argv = ["train_forecast"] + list(argv)
    try:
        main()
    finally:
        sys.argv = _saved


if __name__ == "__main__":
    main()
