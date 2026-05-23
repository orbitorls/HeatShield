"""Check per-station data coverage across the full training window.

Fails with exit code 1 if any station has < --min-coverage hourly coverage.
Run after data ingestion before training:
  python scripts/check_coverage.py --start 2018-01-01 --end 2025-04-30
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.data.stations import STATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_EXTENDED_COLS = ["solar_wm2", "cloud_cover", "pressure_hpa", "blh_m", "lst_c"]


def check_station(station_id: str, start: date, end: date, min_coverage: float) -> bool:
    from app.data.loaders import read_observations

    df = read_observations(station_id, start, end)
    total_hours = (end - start).days * 24
    n_hours = len(df)
    coverage = n_hours / total_hours if total_hours else 0.0

    source_counts = {}
    if "source" in df.columns:
        source_counts = df["source"].value_counts().to_dict()

    ext_counts = {col: int(df[col].notna().sum()) for col in _EXTENDED_COLS if col in df.columns}

    status = "OK" if coverage >= min_coverage else "FAIL"
    logger.info(
        "%s [%s]: %d/%d h (%.1f%%) | sources=%s | ext=%s",
        station_id, status, n_hours, total_hours, coverage * 100, source_counts, ext_counts,
    )
    return coverage >= min_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Check hourly data coverage per station")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--min-coverage", type=float, default=0.90,
                        help="Minimum fraction of hours required (default: 0.90)")
    parser.add_argument("--station", default=None, help="Single station ID (default: all)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    station_ids = [args.station] if args.station else list(STATIONS.keys())

    logger.info("Checking coverage: %s → %s | min=%.1f%%", start, end, args.min_coverage * 100)

    results = {sid: check_station(sid, start, end, args.min_coverage) for sid in station_ids}

    failed = [sid for sid, ok in results.items() if not ok]
    if failed:
        logger.error("FAIL: stations below %.1f%% threshold: %s", args.min_coverage * 100, failed)
        sys.exit(1)
    else:
        logger.info("PASS: all %d stations meet %.1f%% coverage threshold", len(station_ids), args.min_coverage * 100)


if __name__ == "__main__":
    main()
