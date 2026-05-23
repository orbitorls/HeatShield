"""Download MODIS Terra LST for HeatShield AI stations via NASA APPEEARS.

MODIS Terra daily land surface temperature (MOD11A1.061, 1 km resolution).
Stored as lst_c field in parquet — used as an extended feature during training.

Authentication: Free NASA Earthdata account required.
  Register: https://urs.earthdata.nasa.gov/
  Set env vars: EARTHDATA_USERNAME, EARTHDATA_PASSWORD

Note: APPEEARS tasks are processed server-side — this script submits a task
and waits for completion (typically 5–30 minutes for multi-year requests).
The task can be interrupted and re-queried using --task-id.

Usage:
  # Submit new task for all stations
  python scripts/ingest_modis.py --start 2023-01-01 --end 2025-04-30

  # Re-query an existing task (avoid re-submitting)
  python scripts/ingest_modis.py --task-id abc123def

  # Single station (creates separate task per station)
  python scripts/ingest_modis.py --station BKK_01 --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.data.clients.modis_client import MODISClient
from app.data.loaders import _partition_path, write_observations
from app.data.stations import STATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TASK_ID_CACHE = Path(__file__).parents[2] / "data" / "modis_task_ids.txt"


def save_task_id(task_id: str) -> None:
    _TASK_ID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_TASK_ID_CACHE, "a") as f:
        f.write(task_id + "\n")
    logger.info("Task ID saved to %s — use --task-id %s to resume", _TASK_ID_CACHE, task_id)


def ingest_from_client_result(
    obs_by_station: dict,
    force: bool = False,
) -> dict:
    """Write MODIS observations to parquet, one record per day per station."""
    stats = {"ok": 0, "skipped": 0}
    for station_id, obs_list in obs_by_station.items():
        for obs in obs_list:
            day = obs.ts_utc.date()
            # MODIS LST is daily — we store it at 03:00 UTC
            # Check if the parquet for this day already has lst_c data
            path = _partition_path(station_id, day)
            if path.exists() and not force:
                # Parquet exists — check if it already has lst_c column with data
                import pandas as pd
                existing = pd.read_parquet(str(path))
                if "lst_c" in existing.columns and existing["lst_c"].notna().any():
                    logger.debug("LST already stored: station=%s date=%s", station_id, day)
                    stats["skipped"] += 1
                    continue
                # Parquet exists but no lst_c — merge the MODIS record in
                new_row = {
                    "station_id": obs.station_id,
                    "ts_utc": obs.ts_utc,
                    "temp_c": obs.temp_c,
                    "rh": obs.rh,
                    "source": obs.source,
                    "lst_c": obs.lst_c,
                }
                import pandas as pd
                new_df = pd.DataFrame([new_row])
                merged = pd.concat([existing, new_df], ignore_index=True)
                merged.to_parquet(str(path), index=False)
                logger.info("Merged LST=%.1f°C into station=%s date=%s", obs.lst_c or 0, station_id, day)
                stats["ok"] += 1
            else:
                # Write new parquet with this MODIS record
                write_observations([obs], station_id, day)
                logger.info("Wrote MODIS LST=%.1f°C: station=%s date=%s",
                            obs.lst_c or 0, station_id, day)
                stats["ok"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MODIS LST via NASA APPEEARS")
    parser.add_argument("--station", type=str, default=None,
                        help="Single station ID (default: all stations in one task)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Resume/download from existing APPEEARS task ID")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing LST data")
    args = parser.parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=730)

    client = MODISClient()

    if args.task_id:
        # Resume existing task — skip submission
        logger.info("Resuming APPEEARS task: %s", args.task_id)
        token = client._login()
        csv_data = client._wait_and_download(token, args.task_id)
        obs_by_station = client._parse_csv_all(csv_data)
    elif args.station:
        if args.station not in STATIONS:
            logger.error("Unknown station '%s'", args.station)
            sys.exit(1)
        obs_by_station = {
            args.station: client.fetch_range(args.station, start_date, end_date)
        }
    else:
        obs_by_station = client.fetch_all_stations_range(start_date, end_date)

    total = sum(len(v) for v in obs_by_station.values())
    logger.info("Received %d MODIS daily records across %d stations", total, len(obs_by_station))

    stats = ingest_from_client_result(obs_by_station, force=args.force)
    logger.info("MODIS ingestion complete: %s", stats)


if __name__ == "__main__":
    main()
