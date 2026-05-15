import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import heat_index, events, events_auto, risk, whatif, action_card, forecast, stations, profiles

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: validate config, then pre-warm models on startup."""
    from app.core.config import ConfigValidationError, validate_config
    
    try:
        validate_config()
    except ConfigValidationError as exc:
        logger.error(f"Configuration validation failed: {exc}")
        raise  # Fail fast if config is invalid
    
    _prewarm_v3_forecasters()
    yield


def _prewarm_v3_forecasters() -> None:
    """Pre-load v3 h=24 forecasters in parallel at startup to avoid cold-start latency."""
    from app.ml import registry
    from app.data.stations import STATIONS

    def _load_one(sid: str) -> None:
        try:
            registry.load_latest_v3(sid, 24)
            logger.info("Pre-warmed v3 forecaster: station=%s h=24", sid)
        except FileNotFoundError:
            pass  # v3 not trained yet — v2 fallback will handle it
        except (IOError, OSError, ValueError) as exc:
            logger.warning("Could not pre-warm v3 for %s due to file/format error: %s", sid, exc)
        except Exception as exc:
            logger.warning("Unexpected error pre-warming v3 for %s: %s", sid, exc)

    loop = asyncio.new_event_loop()
    with ThreadPoolExecutor(max_workers=len(STATIONS)) as pool:
        loop.run_until_complete(
            asyncio.gather(*[loop.run_in_executor(pool, _load_one, sid) for sid in STATIONS])
        )
    loop.close()


app = FastAPI(
    title="HeatShield AI",
    description=(
        "Adaptive Heat-Health Risk Intelligence for School Safety and Outdoor Workers. "
        "Converts weather data into actionable risk scores and decision support."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://heatshield.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(stations.router, prefix="/stations", tags=["Stations"])
app.include_router(profiles.router, prefix="/profiles", tags=["Profiles"])
app.include_router(heat_index.router, prefix="/heat-index", tags=["Heat Index"])
app.include_router(events.router, prefix="/events", tags=["Heatwave Events"])
app.include_router(events_auto.router, prefix="/events", tags=["Heatwave Events"])
app.include_router(risk.router, prefix="/risk", tags=["Risk Scoring"])
app.include_router(whatif.router, prefix="/whatif", tags=["What-if Simulator"])
app.include_router(action_card.router, prefix="/action-card", tags=["Action Card"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "HeatShield AI"}


@app.get("/health/detailed")
def health_detailed():
    """Detailed health check with monitoring metrics and system status."""
    from app.core.monitoring import get_health_report
    from app.data.stations import STATIONS
    from app.data.circuit_breaker import get_tmd_circuit

    return {
        "status": "ok",
        "service": "HeatShield AI",
        "stations_configured": len(STATIONS),
        "station_regions": {
            "central": 3,
            "northern": 3,
            "northeastern": 4,
            "eastern": 3,
            "southern": 3,
        },
        "tmd_circuit_state": get_tmd_circuit().state.value,
        "monitoring": get_health_report(),
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors with structured response."""
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions — returns structured error instead of raw 500."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred. Check server logs."},
    )
