"""Backfill TMD data for multiple stations over a date range.

Downloads one station-day at a time, skipping existing partitions (idempotent).
Sleeps briefly between requests to respect TMD rate limits.
Can be interrupted and restarted safely — already-downloaded partitions are skipped.

Usage:
    python scripts/build_historical.py --stations BKK_01,CNX_01 --start 2023-01-01 --end 2025-12-31
    python scripts/build_historical.py  # all stations, last 2 years
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.data.stations import STATIONS
from scripts.ingest.tmd import ingest_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP_S = 1.0  # seconds between requests (polite default)


async def backfill(
    station_ids: list[str],
    start: date,
    end: date,
    sleep_s: float = _RATE_LIMIT_SLEEP_S,
    force: bool = False,
) -> dict:
    """Backfill all station-days in the date range."""
    total_days = (end - start).days + 1
    total_tasks = len(station_ids) * total_days

    logger.info(
        "Backfill: %d stations × %d days = %d station-days to process",
        len(station_ids), total_days, total_tasks
    )

    stats = {"ok": 0, "skipped": 0, "error": 0}

    for station_id in station_ids:
        current = start
        while current <= end:
            result = await ingest_one(station_id, current, force=force)
            status = result.get("status", "error")
            if status == "ok":
                stats["ok"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["error"] += 1
                logger.warning("Failed: station=%s date=%s status=%s", station_id, current, status)

            current += timedelta(days=1)

            # Rate limiting: only sleep on actual API calls (not skipped)
            if status != "skipped" and sleep_s > 0:
                await asyncio.sleep(sleep_s)

    logger.info("Backfill complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill TMD historical weather data")
    parser.add_argument(
        "--stations", type=str, default=None,
        help="Comma-separated station IDs (default: all 5 stations)"
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--sleep", type=float, default=_RATE_LIMIT_SLEEP_S,
                        help=f"Sleep seconds between API calls (default: {_RATE_LIMIT_SLEEP_S})")
    parser.add_argument("--force", action="store_true", help="Re-download even if parquet exists")
    args = parser.parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=730)

    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",")]
        invalid = [s for s in station_ids if s not in STATIONS]
        if invalid:
            logger.error("Unknown station IDs: %s. Valid: %s", invalid, list(STATIONS.keys()))
            sys.exit(1)
    else:
        station_ids = list(STATIONS.keys())

    if start_date > end_date:
        logger.error("--start (%s) must be before --end (%s)", start_date, end_date)
        sys.exit(1)

    asyncio.run(backfill(station_ids, start_date, end_date, sleep_s=args.sleep, force=args.force))


if __name__ == "__main__":
    main()
