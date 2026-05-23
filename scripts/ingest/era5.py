"""Download ERA5 reanalysis data for HeatShield AI stations.

ERA5 is ECMWF's global atmospheric reanalysis — free, high-quality historical
data from 1940 to present. No TMD API key required.

One-time setup:
  1. pip install cdsapi xarray netCDF4
  2. Register free account at https://cds.climate.copernicus.eu/
  3. Accept the ERA5 license on the dataset page
  4. Go to My Account → API Tokens → copy your Personal Access Token
  5. Create ~/.cdsapirc:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR_PERSONAL_ACCESS_TOKEN
     OR export CDSAPI_KEY=YOUR_PERSONAL_ACCESS_TOKEN

Usage:
  # All stations, last 2 years (recommended for first run)
  python scripts/ingest_era5.py --start 2023-01-01 --end 2025-04-30

  # Single station, narrow range (quick test)
  python scripts/ingest_era5.py --station BKK_01 --start 2024-01-01 --end 2024-01-07

  # Skip dates already downloaded
  python scripts/ingest_era5.py --start 2023-01-01  # default: skip existing

Performance:
  - Each daily NetCDF (~1 MB) covers all Thai stations, so one CDS download per day.
  - CDS queue time varies: 30 s to 5 min per day. For 2 years = ~730 downloads.
  - Use --workers 2 to parallelize downloads (CDS allows limited concurrency).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.data.clients.era5_client import ERA5Client
from app.data.loaders import _partition_path, write_observations
from app.data.quality import filter_observations
from app.data.stations import STATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_station_day(
    client: ERA5Client,
    station_id: str,
    day: date,
    force: bool = False,
) -> dict:
    """Ingest ERA5 data for one station-day. Returns summary dict.

    If the parquet already has nasa_power rows, ERA5 rows are merged in
    (both sources kept; read_observations deduplicates preferring ERA5).
    """
    parquet_path = _partition_path(station_id, day)
    if parquet_path.exists() and not force:
        existing = pd.read_parquet(parquet_path, engine="pyarrow")
        if "source" in existing.columns and "era5" in existing["source"].values:
            logger.debug("Skip (era5 exists): station=%s date=%s", station_id, day)
            return {"station_id": station_id, "date": str(day), "status": "skipped"}
        # Has nasa_power only — fetch ERA5 and merge below

    try:
        raw_obs = client.fetch_day(station_id, day)
    except Exception as exc:
        logger.error("ERA5 fetch failed: station=%s date=%s error=%s", station_id, day, exc)
        return {"station_id": station_id, "date": str(day), "status": "error", "error": str(exc)}

    kept, dropped = filter_observations(raw_obs)
    if dropped:
        logger.warning("Dropped %d/%d obs for %s %s", len(dropped), len(raw_obs), station_id, day)

    if kept:
        era5_df = pd.DataFrame([o.model_dump() for o in kept])
        if parquet_path.exists() and not force:
            existing = pd.read_parquet(parquet_path, engine="pyarrow")
            merged = pd.concat([existing, era5_df], ignore_index=True)
        else:
            merged = era5_df
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(parquet_path, index=False, engine="pyarrow")
        logger.info("Wrote %d ERA5 obs (merged): station=%s date=%s", len(kept), station_id, day)
    else:
        logger.warning("No valid observations: station=%s date=%s", station_id, day)

    return {"station_id": station_id, "date": str(day), "status": "ok", "observations": len(kept)}


def ingest_range(
    station_ids: list[str],
    start: date,
    end: date,
    force: bool = False,
    workers: int = 1,
) -> dict:
    """Download ERA5 for all station-days in date range.

    ERA5 downloads one NetCDF per calendar day covering the Thailand bounding box,
    so all stations share the same download — we download the file once and
    extract all stations from it.
    """
    client = ERA5Client()
    stats = {"ok": 0, "skipped": 0, "error": 0}

    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)

    total_days = len(days)
    total_tasks = len(station_ids) * total_days
    logger.info(
        "ERA5 backfill: %d stations × %d days = %d station-days",
        len(station_ids), total_days, total_tasks,
    )

    def process_day(day: date) -> list[dict]:
        results = []
        for sid in station_ids:
            results.append(ingest_station_day(client, sid, day, force=force))
        return results

    if workers > 1:
        def process_day_isolated(day: date) -> list[dict]:
            # Each thread gets its own ERA5Client so cdsapi.Client is not shared.
            thread_client = ERA5Client()
            results = []
            for sid in station_ids:
                results.append(ingest_station_day(thread_client, sid, day, force=force))
            return results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_day_isolated, d): d for d in days}
            for fut in as_completed(futures):
                for r in fut.result():
                    stats[r.get("status", "error")] = stats.get(r.get("status", "error"), 0) + 1
    else:
        for i, day in enumerate(days, 1):
            logger.info("Processing day %d/%d: %s", i, total_days, day)
            for r in process_day(day):
                status = r.get("status", "error")
                stats[status] = stats.get(status, 0) + 1

    logger.info("ERA5 backfill complete: %s", stats)
    return stats


def ingest_range_chunked(
    station_ids: list[str],
    start: date,
    end: date,
    force: bool = False,
    workers: int = 2,
    chunk_months: int = 3,
) -> dict:
    """Download ERA5 in N-month chunks (default: quarterly = 3 months/request).

    Quarterly mode: ~17 chunks × 2 CDS calls = 34 total requests (vs 100 for monthly).
    ETA: ~17/2 workers × 6 min/chunk = ~51 min downloads + ~30 min ingestion.
    Falls back to monthly (chunk_months=1) on CDS cost-limit errors.
    """
    from datetime import date as _date
    import calendar

    client = ERA5Client()
    stats: dict[str, int] = {"ok": 0, "skipped": 0, "error": 0}

    all_days: list[date] = []
    cur = start
    while cur <= end:
        all_days.append(cur)
        cur += timedelta(days=1)

    # Group missing days by (year, chunk_idx) where chunk = floor((month-1)/chunk_months)
    chunk_groups: dict[tuple[int, int], dict] = defaultdict(lambda: {"months": set(), "days": []})
    for d in all_days:
        nc_path = client.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc"
        if not nc_path.exists() or force:
            chunk_idx = (d.month - 1) // chunk_months
            key = (d.year, chunk_idx)
            chunk_groups[key]["months"].add(d.month)
            chunk_groups[key]["days"].append(d)

    n_missing = sum(len(v["days"]) for v in chunk_groups.values())
    logger.info("Chunked batch plan: chunk_months=%d, %d chunks, %d/%d days to download",
                chunk_months, len(chunk_groups), n_missing, len(all_days))

    def download_chunk(key: tuple[int, int]) -> None:
        year, _ = key
        months = sorted(chunk_groups[key]["months"])
        days = chunk_groups[key]["days"]
        try:
            client.download_chunk_batch(year, months, days)
        except Exception as exc:
            logger.error("Chunk download failed %d months=%s: %s", year, months, exc)

    sorted_keys = sorted(chunk_groups)
    batch_workers = min(workers, 2)
    if batch_workers > 1:
        with ThreadPoolExecutor(max_workers=batch_workers) as pool:
            futs = {pool.submit(download_chunk, k): k for k in sorted_keys}
            for i, fut in enumerate(as_completed(futs), 1):
                k = futs[fut]
                try:
                    fut.result()
                    logger.info("Chunk %d/%d done: %d-%s", i, len(sorted_keys),
                                k[0], sorted(chunk_groups[k]["months"]))
                except Exception as exc:
                    logger.error("Chunk %s raised: %s", k, exc)
    else:
        for i, k in enumerate(sorted_keys, 1):
            logger.info("Downloading chunk %d/%d: %d months=%s (%d days)",
                        i, len(sorted_keys), k[0], sorted(chunk_groups[k]["months"]),
                        len(chunk_groups[k]["days"]))
            download_chunk(k)

    logger.info("Ingesting %d station-days (%d stations × %d days)…",
                len(all_days) * len(station_ids), len(station_ids), len(all_days))
    for i, d in enumerate(all_days, 1):
        if i % 200 == 0:
            logger.info("Ingest progress: %d/%d days", i, len(all_days))
        for sid in station_ids:
            r = ingest_station_day(client, sid, d, force=force)
            s = r.get("status", "error")
            stats[s] = stats.get(s, 0) + 1

    logger.info("Chunked ingest complete: %s", stats)
    return stats


def ingest_range_yearly(
    station_ids: list[str],
    start: date,
    end: date,
    force: bool = False,
    workers: int = 2,
) -> dict:
    """Download ERA5 per-year (2 CDS calls/year vs 24/year for monthly), then ingest parquet.

    Reduces CDS calls from ~100 to ~10 for a 4-year backfill.
    ETA: ~20-30 min downloads + ~30 min parquet ingestion = ~1 hour total.
    """
    client = ERA5Client()
    stats: dict[str, int] = {"ok": 0, "skipped": 0, "error": 0}

    # Build full day list and find missing
    all_days: list[date] = []
    cur = start
    while cur <= end:
        all_days.append(cur)
        cur += timedelta(days=1)

    missing_by_year: dict[int, list[date]] = defaultdict(list)
    for d in all_days:
        nc_path = client.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc"
        if not nc_path.exists() or force:
            missing_by_year[d.year].append(d)

    n_missing = sum(len(v) for v in missing_by_year.values())
    logger.info("Yearly batch plan: %d years, %d/%d days to download",
                len(missing_by_year), n_missing, len(all_days))

    def download_year(year: int) -> None:
        try:
            client.download_year_batch(year, missing_by_year[year])
        except Exception as exc:
            logger.error("Year batch download failed %d: %s", year, exc)

    sorted_years = sorted(missing_by_year)
    batch_workers = min(workers, 2)
    if batch_workers > 1:
        with ThreadPoolExecutor(max_workers=batch_workers) as pool:
            futs = {pool.submit(download_year, y): y for y in sorted_years}
            for i, fut in enumerate(as_completed(futs), 1):
                y = futs[fut]
                try:
                    fut.result()
                    logger.info("Year batch %d/%d complete: %d", i, len(sorted_years), y)
                except Exception as exc:
                    logger.error("Year %d raised: %s", y, exc)
    else:
        for i, y in enumerate(sorted_years, 1):
            logger.info("Downloading year %d/%d: %d (%d days)",
                        i, len(sorted_years), y, len(missing_by_year[y]))
            download_year(y)

    # Ingest all station-days to parquet
    logger.info("Ingesting %d station-days (%d stations × %d days)…",
                len(all_days) * len(station_ids), len(station_ids), len(all_days))
    for i, d in enumerate(all_days, 1):
        if i % 200 == 0:
            logger.info("Ingest progress: %d/%d days", i, len(all_days))
        for sid in station_ids:
            r = ingest_station_day(client, sid, d, force=force)
            s = r.get("status", "error")
            stats[s] = stats.get(s, 0) + 1

    logger.info("Yearly ingest complete: %s", stats)
    return stats


def ingest_range_monthly(
    station_ids: list[str],
    start: date,
    end: date,
    force: bool = False,
    workers: int = 1,
) -> dict:
    """Download ERA5 in monthly batches (1 CDS request per month), then ingest all station-days.

    Reduces CDS API calls from ~N_days to ~N_months, cutting wall time from ~25h to ~2-3h.
    """
    client = ERA5Client()
    stats: dict[str, int] = {"ok": 0, "skipped": 0, "error": 0}

    # Build full day list
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    # Group missing days by (year, month)
    month_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for d in days:
        nc_path = client.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc"
        if not nc_path.exists() or force:
            month_groups[(d.year, d.month)].append(d.day)

    n_missing = sum(len(v) for v in month_groups.values())
    logger.info(
        "Monthly batch plan: %d months, %d/%d days to download",
        len(month_groups), n_missing, len(days),
    )

    def download_month(ym: tuple[int, int]) -> None:
        year, month = ym
        try:
            client.download_month_batch(year, month, month_groups[ym])
        except Exception as exc:
            logger.error("Batch download failed %d-%02d: %s", year, month, exc)

    sorted_months = sorted(month_groups)
    if workers > 1:
        batch_workers = min(workers, 2)  # CDS allows at most 2 concurrent requests
        with ThreadPoolExecutor(max_workers=batch_workers) as pool:
            futs = {pool.submit(download_month, ym): ym for ym in sorted_months}
            for i, fut in enumerate(as_completed(futs), 1):
                ym = futs[fut]
                logger.info("Batch %d/%d complete: %d-%02d", i, len(sorted_months), *ym)
                try:
                    fut.result()
                except Exception as exc:
                    logger.error("Batch %d-%02d raised: %s", *ym, exc)
    else:
        for i, ym in enumerate(sorted_months, 1):
            logger.info("Downloading batch %d/%d: %d-%02d (%d days)",
                        i, len(sorted_months), ym[0], ym[1], len(month_groups[ym]))
            download_month(ym)

    # Ingest all station-days (using already-cached NC files)
    logger.info("Ingesting %d station-days (%d stations × %d days)…",
                len(days) * len(station_ids), len(station_ids), len(days))
    for i, d in enumerate(days, 1):
        if i % 200 == 0:
            logger.info("Ingest progress: %d/%d days", i, len(days))
        for sid in station_ids:
            r = ingest_station_day(client, sid, d, force=force)
            s = r.get("status", "error")
            stats[s] = stats.get(s, 0) + 1

    logger.info("Monthly ingest complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ERA5 reanalysis data for Thai stations")
    parser.add_argument("--station", type=str, default=None,
                        help="Single station ID (default: all 5 stations)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if parquet already exists")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel CDS download workers (default: 1, max recommended: 2)")
    parser.add_argument("--monthly", action="store_true",
                        help="Use monthly batch downloads (1 CDS request/month — much faster for large ranges)")
    parser.add_argument("--yearly", action="store_true",
                        help="Use yearly batch downloads (2 CDS requests/year — fastest for multi-year backfills)")
    parser.add_argument("--chunked", action="store_true",
                        help="Use N-month chunk batch downloads (default 3 months/request — fastest safe option)")
    parser.add_argument("--chunk-months", type=int, default=3,
                        help="Months per CDS chunk when using --chunked (default: 3 = quarterly)")
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

    if start_date > end_date:
        logger.error("--start must be before --end")
        sys.exit(1)

    logger.info(
        "Ingesting ERA5: stations=%s start=%s end=%s",
        station_ids, start_date, end_date,
    )

    try:
        if args.chunked:
            ingest_range_chunked(station_ids, start_date, end_date,
                                 force=args.force, workers=args.workers,
                                 chunk_months=args.chunk_months)
        elif args.yearly:
            ingest_range_yearly(station_ids, start_date, end_date,
                                force=args.force, workers=args.workers)
        elif args.monthly:
            ingest_range_monthly(station_ids, start_date, end_date,
                                 force=args.force, workers=args.workers)
        else:
            ingest_range(station_ids, start_date, end_date, force=args.force, workers=args.workers)
    except ImportError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
