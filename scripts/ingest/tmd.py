"""One-shot TMD data ingestion for a single station + date.

Idempotent: if the parquet file already exists for this station/date, skips the API call.
This makes it safe to re-run without burning API quota.

Usage:
    python scripts/ingest_tmd.py --station BKK_01 --date 2024-05-01
    python scripts/ingest_tmd.py --station BKK_01  # defaults to yesterday
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.data.loaders import _partition_path, write_observations
from app.data.quality import filter_observations
from app.data.stations import STATIONS, get_station
from app.data.clients.tmd_client import TMDClient, TMDAPIError, TMDAuthError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def ingest_one(station_id: str, day: date, force: bool = False) -> dict:
    """Ingest one station-day. Returns summary dict."""
    # Idempotency check
    parquet_path = _partition_path(station_id, day)
    if parquet_path.exists() and not force:
        logger.info("Already exists: %s — skipping (use --force to overwrite)", parquet_path)
        return {"station_id": station_id, "date": day.isoformat(), "status": "skipped", "observations": 0}

    # Fetch
    client = TMDClient()
    try:
        raw_obs = await client.fetch_hourly(station_id, day)
    except TMDAuthError as e:
        logger.error("Auth failed for %s: %s. Check TMD_API_KEY in .env", station_id, e)
        return {"station_id": station_id, "date": day.isoformat(), "status": "auth_error", "observations": 0}
    except TMDAPIError as e:
        logger.error("API error for %s: %s", station_id, e)
        return {"station_id": station_id, "date": day.isoformat(), "status": "api_error", "observations": 0}

    # Quality filter
    kept, dropped = filter_observations(raw_obs)
    if dropped:
        logger.warning("Dropped %d/%d observations for %s %s: %s",
                       len(dropped), len(raw_obs), station_id, day, [d["reasons"] for d in dropped[:3]])

    # Write parquet
    if kept:
        path = write_observations(kept, station_id, day)
        logger.info("Wrote %d observations → %s", len(kept), path)
    else:
        logger.warning("No valid observations for %s on %s", station_id, day)

    return {
        "station_id": station_id,
        "date": day.isoformat(),
        "status": "ok",
        "observations": len(kept),
        "dropped": len(dropped),
    }


async def main_async(station_id: str, day: date, force: bool) -> None:
    get_station(station_id)  # validate station exists
    result = await ingest_one(station_id, day, force=force)
    logger.info("Result: %s", result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one station-day from TMD API")
    parser.add_argument("--station", type=str, required=True, choices=list(STATIONS.keys()),
                        help="Station ID (e.g. BKK_01)")
    parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet file")
    args = parser.parse_args()

    day = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    asyncio.run(main_async(args.station, day, force=args.force))


if __name__ == "__main__":
    main()
