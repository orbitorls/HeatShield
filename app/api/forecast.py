"""POST /forecast/heat-index — heat index forecast endpoint."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException

from app.data.loaders import read_observations
from app.data.schemas import ForecastRequest, ForecastResponse, StationObservation
from app.data.stations import STATIONS
from app.ml.forecast.predict import predict
from app.core.edr import edr
from app.core.monitoring import record_prediction

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple in-memory LRU cache for identical forecast requests.
# Keyed by (station_id, tuple of horizons, tuple of recent_obs hashes).
# TTL is implicitly short — process restart or cache eviction only.
_FORECAST_CACHE: dict = {}
_FORECAST_CACHE_ORDER: list = []
_FORECAST_CACHE_MAX = 32
_FORECAST_CACHE_LOCK = asyncio.Lock()


def _request_cache_key(req: ForecastRequest) -> str:
    """Stable cache key for a forecast request."""
    obs_key = ""
    if req.recent_obs:
        # Hash by count + first/last timestamp + last temp/rh for speed
        first = req.recent_obs[0]
        last = req.recent_obs[-1]
        obs_key = f"{len(req.recent_obs)}_{first.ts_utc}_{last.ts_utc}_{last.temp_c:.1f}_{last.rh:.1f}"
    return f"{req.station_id}_{tuple(req.horizons)}_{obs_key}"


@router.post("/heat-index", response_model=ForecastResponse)
async def forecast_heat_index(req: ForecastRequest) -> ForecastResponse:
    """Forecast heat index for a station at requested horizons.

    If `recent_obs` is not provided, loads the last 48 hours from the local parquet store.
    Requires a trained model (run scripts/train_forecast.py first).
    """
    if req.station_id not in STATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown station_id '{req.station_id}'. Valid: {list(STATIONS.keys())}"
        )

    recent_obs: list[StationObservation] | None = req.recent_obs

    # Check in-memory cache for identical requests (thread-safe)
    cache_key = _request_cache_key(req)
    async with _FORECAST_CACHE_LOCK:
        if cache_key in _FORECAST_CACHE:
            _FORECAST_CACHE_ORDER.remove(cache_key)
            _FORECAST_CACHE_ORDER.append(cache_key)
            cached = _FORECAST_CACHE[cache_key]
            # Update generated_at timestamp so it reflects reality
            cached.generated_at = datetime.now(timezone.utc)
            return cached

    # Load from parquet if not provided (offload blocking I/O to thread pool)
    if recent_obs is None:
        today = date.today()
        # Pull 3 days to ensure we have 48+ hours of recent observations
        # (predict() requires max(lags_h)+2 = 26 minimum)
        start = today - timedelta(days=3)
        try:
            df = await asyncio.to_thread(read_observations, req.station_id, start, today)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Failed to load observations: {exc}")

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No local observations for station '{req.station_id}'. "
                    "Provide recent_obs in the request body, or run scripts/ingest_tmd.py first."
                )
            )

        from app.data.schemas import StationObservation as SO
        import pandas as pd
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        recent_obs = [
            SO(
                station_id=req.station_id,
                ts_utc=row.ts_utc,
                temp_c=float(row.temp_c),
                rh=float(row.rh),
                wind_ms=float(row.wind_ms) if row.wind_ms is not None and str(row.wind_ms) != "nan" else None,
                precip_mm=float(row.precip_mm) if row.precip_mm is not None and str(row.precip_mm) != "nan" else None,
                source="cache",
            )
            for row in df.itertuples(index=False)
        ]

    with edr.prediction_request(station=req.station_id) as run:
        t0 = time.perf_counter()
        try:
            # predict() may load models — run in thread pool to avoid blocking event loop
            forecasts, low_confidence, confidence_reason = await asyncio.to_thread(
                predict,
                station_id=req.station_id,
                recent_obs=recent_obs,
                horizons=req.horizons,
            )
        except FileNotFoundError:
            run.log_exception(FileNotFoundError("No trained forecast model found"))
            raise HTTPException(
                status_code=503,
                detail="No trained forecast model found. Run scripts/train_forecast.py first."
            )
        except ValueError as exc:
            run.log_exception(exc)
            raise HTTPException(status_code=422, detail=str(exc))

        latency_ms = (time.perf_counter() - t0) * 1000
        model_version = forecasts[0].model_version if forecasts else "unknown"
        run.log_metric("horizon_count", len(req.horizons))
        run.log_metric("forecast_count", len(forecasts))
        run.log_metric("low_confidence", int(low_confidence))
        run.log_metric("latency_ms", latency_ms)
        run.log_model_artifact(model_version)

        # Determine data source from observations
        data_source = "cache"
        if recent_obs:
            sources = {getattr(o, "source", "unknown") for o in recent_obs if hasattr(o, "source")}
            if "nasa_power" in sources and "tmd" not in sources:
                data_source = "nasa_power"
            elif "tmd" in sources:
                data_source = "tmd"

        # Compute staleness
        staleness_min = 0.0
        if recent_obs:
            newest_ts = max(o.ts_utc for o in recent_obs)
            staleness_min = (datetime.now(timezone.utc) - newest_ts).total_seconds() / 60

        record_prediction(
            station_id=req.station_id,
            model_version=model_version,
            horizons=req.horizons,
            forecasts=forecasts,
            latency_ms=latency_ms,
            low_confidence=low_confidence,
            confidence_reason=confidence_reason,
            data_source=data_source,
            staleness_min=staleness_min,
            fallback_used=data_source == "nasa_power",
        )

        response = ForecastResponse(
            station_id=req.station_id,
            generated_at=datetime.now(timezone.utc),
            forecasts=forecasts,
            model_version=model_version,
            low_confidence=low_confidence,
            confidence_reason=confidence_reason,
        )

    # Cache the response (thread-safe)
    async with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE[cache_key] = response
        _FORECAST_CACHE_ORDER.append(cache_key)
        while len(_FORECAST_CACHE_ORDER) > _FORECAST_CACHE_MAX:
            evict = _FORECAST_CACHE_ORDER.pop(0)
            _FORECAST_CACHE.pop(evict, None)

    return response
