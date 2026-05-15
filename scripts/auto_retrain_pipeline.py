"""Automated retraining pipeline for HeatShield AI.

Checks data freshness and model performance, triggers retraining when needed,
and deploys new models if they show significant improvement.

Usage:
    python scripts/auto_retrain_pipeline.py --station BKK_01 --horizon 24
    python scripts/auto_retrain_pipeline.py --all-stations
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from app.data.loaders import read_observations
from app.data.stations import STATIONS
from app.core.edr import edr

logger = logging.getLogger(__name__)

# Configuration
_MIN_COVERAGE_DAYS = 30  # Minimum days of data required
_MIN_COVERAGE_PCT = 0.90  # 90% coverage threshold
_IMPROVEMENT_THRESHOLD = 0.95  # Deploy if new MAE < 0.95 * current MAE
_EVALUATION_DAYS = 14  # Holdout evaluation period


def check_data_coverage(station_id: str, days: int = 30) -> float:
    """Check data coverage percentage for a station over the last N days."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    df = read_observations(station_id, start_date, end_date)
    if df.empty:
        return 0.0

    expected_hours = days * 24
    actual_hours = len(df)
    return actual_hours / expected_hours


def get_latest_model_metrics(station_id: str, horizon: int) -> dict | None:
    """Load the latest model evaluation metrics from registry."""
    registry_path = (
        Path(__file__).parents[1]
        / "app"
        / "models"
        / "forecast_v3"
        / station_id
        / f"h{horizon}"
        / "registry.json"
    )

    if not registry_path.exists():
        return None

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load registry for %s h%d: %s", station_id, horizon, exc)
        return None


def evaluate_model_on_holdout(station_id: str, horizon: int, days: int = 14) -> float | None:
    """Evaluate current model on recent holdout data.

    Returns MAE or None if evaluation fails.
    """
    from app.ml.forecast.predict import predict
    from app.data.schemas import StationObservation

    end_date = date.today()
    start_date = end_date - timedelta(days=days + 7)  # Extra days for lags

    df = read_observations(station_id, start_date, end_date)
    if df.empty or len(df) < 48:
        return None

    # Use the last available observation point to predict forward
    df = df.sort_values("ts_utc").reset_index(drop=True)
    cutoff_idx = len(df) - horizon - 1
    if cutoff_idx < 25:
        return None

    train_df = df.iloc[:cutoff_idx + 1]
    actual_df = df.iloc[cutoff_idx + 1 : cutoff_idx + 1 + horizon]

    if len(actual_df) < horizon:
        return None

    recent_obs = [
        StationObservation(
            station_id=station_id,
            ts_utc=row.ts_utc,
            temp_c=float(row.temp_c),
            rh=float(row.rh),
            wind_ms=float(row.wind_ms) if pd.notna(row.wind_ms) else None,
            precip_mm=float(row.precip_mm) if pd.notna(row.precip_mm) else None,
            source=row.get("source", "cache") if hasattr(row, "source") else "cache",
        )
        for row in train_df.itertuples(index=False)
    ]

    try:
        forecasts, _, _ = predict(
            station_id=station_id,
            recent_obs=recent_obs,
            horizons=[horizon],
        )
        if not forecasts:
            return None

        predicted_hi = forecasts[0].heat_index_c
        # Calculate actual heat index from actual_df
        from app.core.heat_index import calculate_heat_index

        actual_hi = calculate_heat_index(
            float(actual_df.iloc[-1].temp_c),
            float(actual_df.iloc[-1].rh),
        )
        return abs(predicted_hi - actual_hi)
    except Exception as exc:
        logger.warning("Evaluation failed for %s h%d: %s", station_id, horizon, exc)
        return None


async def retrain_station_horizon(station_id: str, horizon: int) -> dict:
    """Trigger retraining for a specific station and horizon.

    Returns result dict with success status and metrics.
    """
    with edr.training_run(
        station=station_id, horizon=horizon, backend="lightgbm"
    ) as run:
        logger.info("Starting retrain for %s h%d", station_id, horizon)

        # Check data coverage
        coverage = check_data_coverage(station_id, days=_MIN_COVERAGE_DAYS)
        run.log_metric("data_coverage", coverage)

        if coverage < _MIN_COVERAGE_PCT:
            msg = f"Insufficient data coverage: {coverage:.1%} < {_MIN_COVERAGE_PCT:.1%}"
            logger.warning(msg)
            run.log_exception(RuntimeError(msg))
            return {"success": False, "reason": msg}

        # Get current model metrics
        current_metrics = get_latest_model_metrics(station_id, horizon)
        current_mae = current_metrics.get("mae") if current_metrics else None

        if current_mae:
            run.log_metric("current_mae", current_mae)
            logger.info("Current model MAE: %.2f°C", current_mae)

        # Evaluate current model on recent data
        eval_mae = evaluate_model_on_holdout(station_id, horizon, days=_EVALUATION_DAYS)
        if eval_mae:
            run.log_metric("holdout_mae", eval_mae)
            logger.info("Holdout evaluation MAE: %.2f°C", eval_mae)

        # Trigger training script
        try:
            from scripts.train_forecast import train_station_horizon

            new_model_path = await asyncio.to_thread(
                train_station_horizon,
                station_id=station_id,
                horizon=horizon,
            )
            run.log_model_artifact(str(new_model_path))
            logger.info("New model saved to %s", new_model_path)
        except Exception as exc:
            msg = f"Training failed: {exc}"
            logger.error(msg)
            run.log_exception(exc)
            return {"success": False, "reason": msg}

        # Evaluate new model
        new_eval_mae = evaluate_model_on_holdout(station_id, horizon, days=_EVALUATION_DAYS)
        if new_eval_mae:
            run.log_metric("new_holdout_mae", new_eval_mae)
            logger.info("New model holdout MAE: %.2f°C", new_eval_mae)

            # Decide whether to deploy
            if current_mae and new_eval_mae < current_mae * _IMPROVEMENT_THRESHOLD:
                logger.info(
                    "Deploying new model: %.2f°C < %.2f°C * %.2f",
                    new_eval_mae, current_mae, _IMPROVEMENT_THRESHOLD,
                )
                run.log_champion_result(
                    won=True,
                    mae_delta=new_eval_mae - current_mae,
                    reason=f"MAE improved from {current_mae:.2f} to {new_eval_mae:.2f}",
                )
                return {
                    "success": True,
                    "deployed": True,
                    "old_mae": current_mae,
                    "new_mae": new_eval_mae,
                    "model_path": str(new_model_path),
                }
            else:
                logger.info(
                    "Keeping current model (new MAE %.2f°C not significantly better)",
                    new_eval_mae,
                )
                run.log_champion_result(
                    won=False,
                    mae_delta=(new_eval_mae - current_mae) if current_mae else 0,
                    reason="Insufficient improvement",
                )
                return {
                    "success": True,
                    "deployed": False,
                    "old_mae": current_mae,
                    "new_mae": new_eval_mae,
                    "reason": "No significant improvement",
                }

        return {"success": True, "deployed": False, "reason": "Could not evaluate new model"}


async def retrain_all_stations(horizons: list[int] | None = None) -> list[dict]:
    """Retrain all stations and horizons."""
    if horizons is None:
        horizons = [6, 12, 24]

    results = []
    for station_id in sorted(STATIONS.keys()):
        for horizon in horizons:
            try:
                result = await retrain_station_horizon(station_id, horizon)
                results.append({
                    "station_id": station_id,
                    "horizon": horizon,
                    **result,
                })
            except Exception as exc:
                logger.error("Unexpected error for %s h%d: %s", station_id, horizon, exc)
                results.append({
                    "station_id": station_id,
                    "horizon": horizon,
                    "success": False,
                    "reason": str(exc),
                })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="HeatShield AI Auto-Retrain Pipeline")
    parser.add_argument("--station", help="Specific station ID to retrain")
    parser.add_argument("--horizon", type=int, help="Specific horizon (6, 12, 24)")
    parser.add_argument("--all-stations", action="store_true", help="Retrain all stations")
    parser.add_argument("--horizons", nargs="+", type=int, default=[6, 12, 24])
    parser.add_argument("--coverage-days", type=int, default=_MIN_COVERAGE_DAYS)
    parser.add_argument("--coverage-pct", type=float, default=_MIN_COVERAGE_PCT)
    parser.add_argument("--improvement-threshold", type=float, default=_IMPROVEMENT_THRESHOLD)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Update config from args (module-level constants updated directly)
    import __main__
    __main__._MIN_COVERAGE_DAYS = args.coverage_days
    __main__._MIN_COVERAGE_PCT = args.coverage_pct
    __main__._IMPROVEMENT_THRESHOLD = args.improvement_threshold

    if args.all_stations:
        results = asyncio.run(retrain_all_stations(args.horizons))
    elif args.station and args.horizon:
        results = [asyncio.run(retrain_station_horizon(args.station, args.horizon))]
    else:
        parser.print_help()
        sys.exit(1)

    # Print summary
    successful = sum(1 for r in results if r.get("success"))
    deployed = sum(1 for r in results if r.get("deployed"))
    print(f"\n{'='*50}")
    print(f"Retraining Summary")
    print(f"{'='*50}")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Models deployed: {deployed}")
    print(f"Failed: {len(results) - successful}")

    for r in results:
        status = "DEPLOYED" if r.get("deployed") else ("OK" if r.get("success") else "FAIL")
        print(f"  {r['station_id']} h{r['horizon']}: {status} - {r.get('reason', '')}")


if __name__ == "__main__":
    main()
