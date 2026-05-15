"""Master data ingestion script — pulls all weather data sources for HeatShield AI.

Sources (in recommended order):
  1. NASA POWER (MERRA-2)  — no auth, solar + cloud included, 1981–present
  2. ERA5 (CDS)            — needs CDS API key, higher accuracy, 1940–present
  3. MODIS LST             — needs Earthdata auth, daily LST, 2000–present
  4. TMD (optional)        — needs TMD_API_KEY, real Thai station data

You can run all sources or pick individual ones with flags.
After ingestion, run: python scripts/train_forecast.py --horizon 24

Usage:
  # Recommended first run (no credentials needed):
  python scripts/ingest_all.py --sources nasa_power --start 2023-01-01

  # All sources if you have credentials:
  python scripts/ingest_all.py --sources nasa_power,era5,modis --start 2022-01-01

  # With TMD data too:
  python scripts/ingest_all.py --sources nasa_power,tmd --start 2024-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.data.stations import STATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ALL_SOURCES = ["nasa_power", "era5", "modis", "tmd"]


def run_nasa_power(station_ids: list[str], start: date, end: date, force: bool) -> None:
    from app.data.nasa_power_client import NASAPowerClient
    from scripts.ingest_nasa_power import ingest_station_range

    logger.info("=" * 60)
    logger.info("SOURCE: NASA POWER (MERRA-2 reanalysis, no auth required)")
    logger.info("=" * 60)

    client = NASAPowerClient()
    with ThreadPoolExecutor(max_workers=min(4, len(station_ids))) as executor:
        future_to_sid = {
            executor.submit(ingest_station_range, client, sid, start, end, force=force): sid
            for sid in station_ids
        }
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            try:
                stats = future.result()
                logger.info("NASA POWER %s: %s", sid, stats)
            except Exception as exc:
                logger.error("NASA POWER %s failed: %s", sid, exc)


def run_era5(station_ids: list[str], start: date, end: date, force: bool) -> None:
    from scripts.ingest_era5 import ingest_range

    logger.info("=" * 60)
    logger.info("SOURCE: ERA5 (ECMWF reanalysis via CDS API — needs ~/.cdsapirc)")
    logger.info("=" * 60)

    try:
        from app.data.era5_client import _require_libs
        _require_libs()
    except ImportError as e:
        logger.error("ERA5 skipped: %s", e)
        return

    ingest_range(station_ids, start, end, force=force, workers=1)


def run_modis(station_ids: list[str], start: date, end: date, force: bool) -> None:
    from app.data.modis_client import MODISClient
    from scripts.ingest_modis import ingest_from_client_result

    logger.info("=" * 60)
    logger.info("SOURCE: MODIS LST (NASA APPEEARS — needs EARTHDATA_USERNAME/PASSWORD)")
    logger.info("=" * 60)

    client = MODISClient()
    try:
        if len(station_ids) == len(STATIONS):
            obs_by_station = client.fetch_all_stations_range(start, end)
        else:
            obs_by_station = {}
            for sid in station_ids:
                obs_by_station[sid] = client.fetch_range(sid, start, end)
    except EnvironmentError as e:
        logger.error("MODIS skipped: %s", e)
        return
    except Exception as e:
        logger.error("MODIS failed: %s", e)
        return

    stats = ingest_from_client_result(obs_by_station, force=force)
    logger.info("MODIS complete: %s", stats)


def run_tmd(station_ids: list[str], start: date, end: date, force: bool) -> None:
    import asyncio
    from scripts.build_historical import backfill

    logger.info("=" * 60)
    logger.info("SOURCE: TMD (Thai Meteorological Department — needs TMD_API_KEY)")
    logger.info("=" * 60)

    api_key = os.getenv("TMD_API_KEY")
    if not api_key:
        logger.error("TMD skipped: TMD_API_KEY not set in environment")
        return

    asyncio.run(backfill(station_ids, start, end, force=force))


def print_coverage_summary(station_ids: list[str], start: date, end: date) -> None:
    """Print a summary of how much data is available per station."""
    from app.data.loaders import read_observations
    import pandas as pd

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATA COVERAGE SUMMARY")
    logger.info("=" * 60)

    for sid in station_ids:
        try:
            df = read_observations(sid, start, end)
            if df.empty:
                logger.info("  %s: NO DATA", sid)
                continue

            n_hours = len(df)
            total_hours = int((end - start).days * 24)
            coverage_pct = 100.0 * n_hours / total_hours if total_hours else 0

            sources = df["source"].value_counts().to_dict() if "source" in df.columns else {}
            has_solar = df["solar_wm2"].notna().sum() if "solar_wm2" in df.columns else 0
            has_lst = df["lst_c"].notna().sum() if "lst_c" in df.columns else 0

            logger.info(
                "  %s: %d/%d hours (%.0f%%) | sources=%s | solar=%d | LST=%d",
                sid, n_hours, total_hours, coverage_pct, sources, has_solar, has_lst,
            )
        except Exception as exc:
            logger.warning("  %s: error reading data — %s", sid, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master data ingestion — pull all weather sources for HeatShield AI"
    )
    parser.add_argument(
        "--sources", type=str, default="nasa_power",
        help=f"Comma-separated list: {','.join(_ALL_SOURCES)} (default: nasa_power)"
    )
    parser.add_argument("--station", type=str, default=None,
                        help="Single station (default: all 5)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if data already exists")
    args = parser.parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=730)

    if args.station:
        if args.station not in STATIONS:
            logger.error("Unknown station '%s'. Valid: %s", args.station, list(STATIONS.keys()))
            sys.exit(1)
        station_ids = [args.station]
    else:
        station_ids = list(STATIONS.keys())

    requested = [s.strip().lower() for s in args.sources.split(",")]
    unknown = [s for s in requested if s not in _ALL_SOURCES]
    if unknown:
        logger.error("Unknown sources: %s. Valid: %s", unknown, _ALL_SOURCES)
        sys.exit(1)

    logger.info("HeatShield AI — Data Ingestion")
    logger.info("Stations: %s | %s → %s | sources: %s", station_ids, start_date, end_date, requested)

    if "nasa_power" in requested:
        run_nasa_power(station_ids, start_date, end_date, args.force)

    if "era5" in requested:
        # ERA5 already parallelises internally; keep sequential entry
        run_era5(station_ids, start_date, end_date, args.force)

    if "modis" in requested:
        run_modis(station_ids, start_date, end_date, args.force)

    if "tmd" in requested:
        run_tmd(station_ids, start_date, end_date, args.force)

    print_coverage_summary(station_ids, start_date, end_date)
    logger.info("")
    logger.info("Next step: python scripts/train_forecast.py --horizon 24")


if __name__ == "__main__":
    main()
