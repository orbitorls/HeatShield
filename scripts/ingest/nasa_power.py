"""Download NASA POWER (MERRA-2) weather data for HeatShield AI stations.

No authentication required. Free API, hourly data from 1981 to ~3 days ago.
Includes solar irradiance, cloud cover, and pressure — not available in TMD.

Usage:
  # All stations, last 2 years (recommended starting point)
  python scripts/ingest_nasa_power.py --start 2023-01-01 --end 2025-04-30

  # Single station, test window
  python scripts/ingest_nasa_power.py --station BKK_01 --start 2024-01-01 --end 2024-01-31

  # Efficient: fetch one month per request (faster than day-by-day)
  python scripts/ingest_nasa_power.py --chunk-days 30 --start 2023-01-01

Performance:
  NASA POWER allows multi-day requests. Default chunk = 90 days per API call.
  Typical throughput: 2 years × 5 stations ≈ 40 requests, ~5 min total.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.data.clients.nasa_power_client import NASAPowerClient
from app.data.loaders import _partition_path, write_observations
from app.data.quality import filter_observations
from app.data.stations import STATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_DAYS = 90  # NASA POWER handles up to ~1 year; 90 days balances memory


def ingest_station_range(
    client: NASAPowerClient,
    station_id: str,
    start: date,
    end: date,
    chunk_days: int = _DEFAULT_CHUNK_DAYS,
    force: bool = False,
) -> dict:
    """Fetch NASA POWER data for a station over a date range, writing daily parquet files."""
    stats = {"ok": 0, "skipped": 0, "error": 0}
    current = start

    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)

        # Check idempotency: skip chunk if all days already exist
        if not force:
            all_exist = all(
                _partition_path(station_id, current + timedelta(days=i)).exists()
                for i in range((chunk_end - current).days + 1)
            )
            if all_exist:
                logger.debug("All days exist for %s %s→%s — skipping", station_id, current, chunk_end)
                stats["skipped"] += (chunk_end - current).days + 1
                current = chunk_end + timedelta(days=1)
                continue

        try:
            obs_list = client.fetch_range(station_id, current, chunk_end)
        except Exception as exc:
            logger.error("NASA POWER fetch failed %s %s→%s: %s", station_id, current, chunk_end, exc)
            stats["error"] += (chunk_end - current).days + 1
            current = chunk_end + timedelta(days=1)
            continue

        # Split by day and write parquet per day
        from collections import defaultdict
        by_day: dict[date, list] = defaultdict(list)
        for obs in obs_list:
            obs_date = obs.ts_utc.date()
            by_day[obs_date].append(obs)

        for day, day_obs in sorted(by_day.items()):
            if not force and _partition_path(station_id, day).exists():
                stats["skipped"] += 1
                continue
            kept, dropped = filter_observations(day_obs)
            if kept:
                write_observations(kept, station_id, day)
                logger.info("Wrote %d obs: station=%s date=%s (solar=%s)",
                            len(kept), station_id, day,
                            "yes" if any(o.solar_wm2 is not None for o in kept) else "no")
                stats["ok"] += 1
            else:
                logger.warning("No valid obs: station=%s date=%s (dropped=%d)", station_id, day, len(dropped))

        current = chunk_end + timedelta(days=1)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NASA POWER (MERRA-2) weather data")
    parser.add_argument("--station", type=str, default=None,
                        help="Single station ID (default: all 5 stations)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--chunk-days", type=int, default=_DEFAULT_CHUNK_DAYS,
                        help=f"Days per API request (default: {_DEFAULT_CHUNK_DAYS})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if parquet already exists")
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

    client = NASAPowerClient()
    total_stats = {"ok": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=min(4, len(station_ids))) as executor:
        future_to_sid = {
            executor.submit(
                ingest_station_range, client, sid, start_date, end_date,
                chunk_days=args.chunk_days, force=args.force
            ): sid
            for sid in station_ids
        }
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            try:
                stats = future.result()
                logger.info("Processing station complete: %s (%s → %s)", sid, start_date, end_date)
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
            except Exception as exc:
                logger.error("Station %s failed: %s", sid, exc)
                total_stats["error"] = total_stats.get("error", 0) + 1

    logger.info("NASA POWER ingestion complete: %s", total_stats)


if __name__ == "__main__":
    main()
