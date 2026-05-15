"""ERA5 reanalysis data client — download hourly weather data for Thai stations.

Source: ECMWF ERA5 via Copernicus Climate Data Store (CDS).
Coverage: 1940-present, global, 0.25° × 0.25° grid, hourly.

Setup (one-time):
  pip install cdsapi xarray netCDF4
  Register at https://cds.climate.copernicus.eu/ → My Account → API key
  Create ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: YOUR_PERSONAL_ACCESS_TOKEN
  OR set env vars: CDSAPI_URL, CDSAPI_KEY

Variable mapping:
  ERA5 long name                        → NetCDF short → Unit → stored as
  2m_temperature                        → t2m          → K    → temp_c (°C)
  2m_dewpoint_temperature               → d2m          → K    → rh (% via Magnus)
  10m_u_component_of_wind               → u10          → m/s  → wind_ms
  10m_v_component_of_wind               → v10          → m/s  → (combined with u10)
  total_precipitation                   → tp           → m    → precip_mm
  surface_solar_radiation_downwards     → ssrd         → J/m² → solar_wm2 (W/m²)
  total_cloud_cover                     → tcc          → 0–1  → cloud_cover
  boundary_layer_height                 → blh          → m    → blh_m
  mean_sea_level_pressure               → msl          → Pa   → pressure_hpa
"""
from __future__ import annotations

import logging
import math
import os
from datetime import date, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.data.schemas import StationObservation
from app.data.stations import STATIONS, WeatherStation

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parents[2] / "data" / "era5_cache"

# Instantaneous variables (returned together by CDS in one request)
_ERA5_INSTANT_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",                   # tcc: 0–1 fraction
    "boundary_layer_height",               # blh: metres
    "mean_sea_level_pressure",             # msl: Pa → ÷100 → hPa
]

# Accumulated flux variables — CDS v2 requires a separate request for these
_ERA5_ACCUM_VARS = [
    "total_precipitation",                 # tp: m per hour → ×1000 → mm
    "surface_solar_radiation_downwards",   # ssrd: J/m² per hour → ÷3600 → W/m²
]

# Keep for backward compatibility
_ERA5_VARIABLES = _ERA5_INSTANT_VARS + _ERA5_ACCUM_VARS

# [North, West, South, East] — bounding box covering Thailand + buffer
_THAILAND_AREA = [22.5, 97.0, 5.0, 106.5]


def _rh_from_t_td(t_c: float, td_c: float) -> float:
    """Relative humidity (%) from 2m temperature and dewpoint via Magnus formula."""
    a, b = 17.625, 243.04
    num = math.exp(a * td_c / (b + td_c))
    den = math.exp(a * t_c / (b + t_c))
    return min(100.0, max(0.0, 100.0 * num / den))


class ERA5Client:
    """Download and parse ERA5 hourly reanalysis for HeatShield AI stations.

    Downloads one day at a time as NetCDF, caches locally, then extracts
    the nearest 0.25° grid point to each station's lat/lon.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://cds.climate.copernicus.eu/api",
        cache_dir: Optional[Path] = None,
    ):
        self.api_key = api_key or os.getenv("CDSAPI_KEY")
        self.api_url = api_url or os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_day(self, station_id: str, day: date) -> list[StationObservation]:
        """Download ERA5 for *day* and extract observations at *station_id*.

        Args:
            station_id: One of the 5 registered station IDs.
            day: UTC date to download.

        Returns:
            Up to 24 hourly StationObservation records (source="era5").

        Raises:
            KeyError: Unknown station_id.
            ImportError: cdsapi / xarray / netCDF4 not installed.
            Exception: CDS API auth failure or network error.
        """
        if station_id not in STATIONS:
            raise KeyError(f"Unknown station_id '{station_id}'")
        station = STATIONS[station_id]

        _require_libs()

        nc_path = self._download_nc(day)
        return self._extract_station(nc_path, station, day)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _download_nc(self, day: date) -> Path:
        """Download ERA5 NetCDF for the Thailand bounding box (cached by date).

        CDS API v2 may return a ZIP archive containing the NetCDF — this method
        detects and unpacks it automatically.
        """
        import cdsapi
        import zipfile

        nc_path = self.cache_dir / f"era5_{day.strftime('%Y%m%d')}.nc"
        if nc_path.exists():
            logger.debug("ERA5 cache hit: %s", nc_path)
            return nc_path

        client = cdsapi.Client(url=self.api_url, key=self.api_key, quiet=True)
        logger.info("Downloading ERA5 for %s from CDS (may take 1-5 min)…", day)

        import os as _os

        def _fetch_vars(variables: list, suffix: str) -> Path:
            tmp = self.cache_dir / f"era5_{day.strftime('%Y%m%d')}_{suffix}.tmp"
            out = self.cache_dir / f"era5_{day.strftime('%Y%m%d')}_{suffix}.nc"
            if out.exists():
                return out
            # Remove stale .tmp from a previous interrupted run before downloading.
            tmp.unlink(missing_ok=True)
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": str(day.year),
                    "month": f"{day.month:02d}",
                    "day": f"{day.day:02d}",
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": _THAILAND_AREA,
                    "format": "netcdf",
                },
                str(tmp),
            )
            if zipfile.is_zipfile(tmp):
                extracted = self.cache_dir / f"era5_{day.strftime('%Y%m%d')}_{suffix}.extracted"
                with zipfile.ZipFile(tmp) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                    if not nc_names:
                        raise RuntimeError(f"No .nc in CDS ZIP: {zf.namelist()}")
                    with zf.open(nc_names[0]) as src, open(str(extracted), "wb") as dst:
                        dst.write(src.read())
                tmp.unlink(missing_ok=True)
                _os.replace(str(extracted), str(out))
            else:
                _os.replace(str(tmp), str(out))
            return out

        # CDS v2 requires separate requests for instantaneous vs accumulated vars
        instant_path = _fetch_vars(_ERA5_INSTANT_VARS, "instant")
        logger.info("Merging accumulated variables (tp, ssrd)…")
        try:
            accum_path = _fetch_vars(_ERA5_ACCUM_VARS, "accum")
            import xarray as xr
            ds_instant = xr.open_dataset(str(instant_path))
            ds_accum = xr.open_dataset(str(accum_path))
            merged = xr.merge([ds_instant, ds_accum])
            merged.to_netcdf(str(nc_path))
            ds_instant.close()
            ds_accum.close()
            instant_path.unlink(missing_ok=True)
            accum_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Accumulated vars fetch failed (%s) — proceeding with instantaneous only", exc)
            if not instant_path.exists():
                raise
            import shutil
            shutil.copy(str(instant_path), str(nc_path))
            instant_path.unlink(missing_ok=True)

        logger.info("Downloaded → %s (%.1f MB)", nc_path, nc_path.stat().st_size / 1e6)
        return nc_path

    def download_chunk_batch(
        self,
        year: int,
        months: list[int],
        missing_days: list[date],
    ) -> list[Path]:
        """Download ERA5 for a multi-month chunk (e.g. one quarter) in two CDS requests.

        Requesting 3 months at a time cuts CDS calls by 3× vs monthly.
        CDS cost limit: monthly (1 month) is ~5k fields — quarterly (~3 months) is ~15k,
        which stays within the CDS v2 per-request cost limit (yearly fails at ~60k).

        Returns list of per-day NC paths created or already cached.
        """
        import cdsapi
        import numpy as np
        import os as _os
        import zipfile

        import xarray as xr

        days_to_fetch = [d for d in missing_days
                         if not (self.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc").exists()]
        existing = [self.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc"
                    for d in missing_days if d not in days_to_fetch]

        if not days_to_fetch:
            logger.debug("ERA5 chunk %d-%s: all days cached", year, months)
            return existing

        month_strs = [f"{m:02d}" for m in sorted(months)]
        all_day_strs = [f"{d:02d}" for d in range(1, 32)]
        prefix = f"era5_{year:04d}_m{'_'.join(month_strs)}"
        logger.info("ERA5 chunk batch %d months=%s: %d missing days", year, month_strs, len(days_to_fetch))

        api_url, api_key = self.api_url, self.api_key

        def _fetch_chunk(variables: list, suffix: str) -> Path:
            tmp = self.cache_dir / f"{prefix}_{suffix}.tmp"
            out = self.cache_dir / f"{prefix}_{suffix}.nc"
            if out.exists():
                return out
            tmp.unlink(missing_ok=True)
            _client = cdsapi.Client(url=api_url, key=api_key, quiet=True)
            _client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": str(year),
                    "month": month_strs,
                    "day": all_day_strs,
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": _THAILAND_AREA,
                    "format": "netcdf",
                },
                str(tmp),
            )
            if zipfile.is_zipfile(tmp):
                extracted = self.cache_dir / f"{prefix}_{suffix}.extracted"
                with zipfile.ZipFile(tmp) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                    if not nc_names:
                        raise RuntimeError(f"No .nc in CDS ZIP: {zf.namelist()}")
                    with zf.open(nc_names[0]) as src, open(str(extracted), "wb") as dst:
                        dst.write(src.read())
                tmp.unlink(missing_ok=True)
                _os.replace(str(extracted), str(out))
            else:
                _os.replace(str(tmp), str(out))
            logger.info("Chunk %d %s %s: %.1f MB downloaded", year, month_strs, suffix,
                        Path(out).stat().st_size / 1e6)
            return out

        # Submit instant and accum CDS requests in parallel to halve queue wait time.
        from concurrent.futures import ThreadPoolExecutor as _ChunkTPE
        with _ChunkTPE(max_workers=2) as _exe:
            _fut_i = _exe.submit(_fetch_chunk, _ERA5_INSTANT_VARS, "instant")
            _fut_a = _exe.submit(_fetch_chunk, _ERA5_ACCUM_VARS, "accum")
            instant_path = _fut_i.result()
            accum_path = None
            try:
                accum_path = _fut_a.result()
            except Exception as exc:
                logger.warning("Accum chunk fetch failed (%s) — instantaneous only", exc)

        try:
            if accum_path is not None:
                ds_i = xr.open_dataset(str(instant_path))
                ds_a = xr.open_dataset(str(accum_path))
                merged = xr.merge([ds_i, ds_a]).load()
                ds_i.close()
                ds_a.close()
            else:
                merged = xr.open_dataset(str(instant_path)).load()
        except Exception as exc:
            logger.warning("Merge/load failed (%s) — instant only", exc)
            merged = xr.open_dataset(str(instant_path)).load()
            accum_path = None

        time_dim = "valid_time" if "valid_time" in merged.dims else "time"
        times = pd.DatetimeIndex(merged[time_dim].values)
        created: list[Path] = list(existing)

        for d in days_to_fetch:
            nc_path = self.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc"
            mask = (times.year == d.year) & (times.month == d.month) & (times.day == d.day)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                logger.warning("No data for %s in chunk batch", d)
                continue
            day_ds = merged.isel({time_dim: list(indices)})
            day_ds.to_netcdf(str(nc_path))
            created.append(nc_path)

        merged.close()
        for _batch in (instant_path, accum_path):
            if _batch is not None:
                try:
                    _batch.unlink(missing_ok=True)
                except OSError:
                    pass  # Windows: brief file-handle retention after close

        logger.info("Chunk %d %s done: %d days written", year, month_strs,
                    len(created) - len(existing))
        return created

    def download_month_batch(
        self,
        year: int,
        month: int,
        days: list[int],
    ) -> list[Path]:
        """Download ERA5 for multiple days in one CDS request, split into per-day NC files.

        Returns list of per-day NC paths that were created or already existed.
        """
        import cdsapi
        import numpy as np
        import os as _os
        import zipfile

        import xarray as xr

        days_to_fetch = [d for d in days
                         if not (self.cache_dir / f"era5_{year:04d}{month:02d}{d:02d}.nc").exists()]
        existing = [self.cache_dir / f"era5_{year:04d}{month:02d}{d:02d}.nc"
                    for d in days if d not in days_to_fetch]

        if not days_to_fetch:
            logger.debug("ERA5 month batch %d-%02d: all %d days cached", year, month, len(days))
            return existing

        client = cdsapi.Client(url=self.api_url, key=self.api_key, quiet=True)
        day_strs = [f"{d:02d}" for d in days_to_fetch]
        prefix = f"era5_{year:04d}{month:02d}_batch"
        logger.info("ERA5 batch download: %d-%02d days=%s", year, month, day_strs)

        def _fetch_batch(variables: list, suffix: str) -> Path:
            tmp = self.cache_dir / f"{prefix}_{suffix}.tmp"
            out = self.cache_dir / f"{prefix}_{suffix}.nc"
            if out.exists():
                return out
            tmp.unlink(missing_ok=True)
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": day_strs,
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": _THAILAND_AREA,
                    "format": "netcdf",
                },
                str(tmp),
            )
            if zipfile.is_zipfile(tmp):
                extracted = self.cache_dir / f"{prefix}_{suffix}.extracted"
                with zipfile.ZipFile(tmp) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                    if not nc_names:
                        raise RuntimeError(f"No .nc in CDS ZIP: {zf.namelist()}")
                    with zf.open(nc_names[0]) as src, open(str(extracted), "wb") as dst:
                        dst.write(src.read())
                tmp.unlink(missing_ok=True)
                _os.replace(str(extracted), str(out))
            else:
                _os.replace(str(tmp), str(out))
            return out

        instant_path = _fetch_batch(_ERA5_INSTANT_VARS, "instant")
        accum_path = None
        try:
            accum_path = _fetch_batch(_ERA5_ACCUM_VARS, "accum")
            ds_i = xr.open_dataset(str(instant_path))
            ds_a = xr.open_dataset(str(accum_path))
            merged = xr.merge([ds_i, ds_a])
            ds_i.close()
            ds_a.close()
        except Exception as exc:
            logger.warning("Accum batch fetch failed (%s) — instantaneous only", exc)
            merged = xr.open_dataset(str(instant_path))

        # Split monthly dataset into per-day NC files
        time_dim = "valid_time" if "valid_time" in merged.dims else "time"
        times = pd.DatetimeIndex(merged[time_dim].values)
        created: list[Path] = list(existing)

        for d in days_to_fetch:
            nc_path = self.cache_dir / f"era5_{year:04d}{month:02d}{d:02d}.nc"
            mask = (times.year == year) & (times.month == month) & (times.day == d)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                logger.warning("No data for %d-%02d-%02d in batch", year, month, d)
                continue
            day_ds = merged.isel({time_dim: list(indices)})
            day_ds.to_netcdf(str(nc_path))
            created.append(nc_path)
            logger.debug("Split %d-%02d-%02d → %s", year, month, d, nc_path.name)

        merged.close()
        instant_path.unlink(missing_ok=True)
        if accum_path:
            accum_path.unlink(missing_ok=True)

        logger.info("Batch %d-%02d done: %d/%d days written",
                    year, month, len(created) - len(existing), len(days_to_fetch))
        return created

    def download_year_batch(
        self,
        year: int,
        missing_days: list[date],
    ) -> list[Path]:
        """Download ERA5 for an entire year in two CDS requests (instant + accum).

        One CDS call per variable group per year instead of one per month — ~10x fewer
        requests, reducing wall time from ~3h to ~20-30 min for a 4-year backfill.

        Returns list of per-day NC paths created (already-cached days skipped).
        """
        import cdsapi
        import numpy as np
        import os as _os
        import zipfile

        import xarray as xr

        days_to_fetch = [d for d in missing_days
                         if not (self.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc").exists()]
        existing = [self.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc"
                    for d in missing_days if d not in days_to_fetch]

        if not days_to_fetch:
            logger.debug("ERA5 year batch %d: all %d days cached", year, len(missing_days))
            return existing

        # Collect unique months that have missing days
        months = sorted({d.month for d in days_to_fetch})
        month_strs = [f"{m:02d}" for m in months]
        # Request all 31 possible days — CDS drops days that don't exist in the month
        all_day_strs = [f"{d:02d}" for d in range(1, 32)]

        client = cdsapi.Client(url=self.api_url, key=self.api_key, quiet=True)
        prefix = f"era5_{year:04d}_year"
        logger.info("ERA5 year batch %d: %d months, %d missing days", year, len(months), len(days_to_fetch))

        def _fetch_year(variables: list, suffix: str) -> Path:
            tmp = self.cache_dir / f"{prefix}_{suffix}.tmp"
            out = self.cache_dir / f"{prefix}_{suffix}.nc"
            if out.exists():
                return out
            tmp.unlink(missing_ok=True)
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": str(year),
                    "month": month_strs,
                    "day": all_day_strs,
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": _THAILAND_AREA,
                    "format": "netcdf",
                },
                str(tmp),
            )
            if zipfile.is_zipfile(tmp):
                extracted = self.cache_dir / f"{prefix}_{suffix}.extracted"
                with zipfile.ZipFile(tmp) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                    if not nc_names:
                        raise RuntimeError(f"No .nc in CDS ZIP: {zf.namelist()}")
                    with zf.open(nc_names[0]) as src, open(str(extracted), "wb") as dst:
                        dst.write(src.read())
                tmp.unlink(missing_ok=True)
                _os.replace(str(extracted), str(out))
            else:
                _os.replace(str(tmp), str(out))
            logger.info("ERA5 year %d %s downloaded: %.1f MB", year, suffix,
                        Path(out).stat().st_size / 1e6)
            return out

        instant_path = _fetch_year(_ERA5_INSTANT_VARS, "instant")
        accum_path = None
        try:
            accum_path = _fetch_year(_ERA5_ACCUM_VARS, "accum")
            ds_i = xr.open_dataset(str(instant_path))
            ds_a = xr.open_dataset(str(accum_path))
            merged = xr.merge([ds_i, ds_a])
            ds_i.close()
            ds_a.close()
        except Exception as exc:
            logger.warning("Accum year fetch failed (%s) — instantaneous only", exc)
            merged = xr.open_dataset(str(instant_path))

        time_dim = "valid_time" if "valid_time" in merged.dims else "time"
        times = pd.DatetimeIndex(merged[time_dim].values)
        created: list[Path] = list(existing)

        for d in days_to_fetch:
            nc_path = self.cache_dir / f"era5_{d.strftime('%Y%m%d')}.nc"
            mask = (times.year == d.year) & (times.month == d.month) & (times.day == d.day)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                logger.warning("No data for %s in year batch %d", d, year)
                continue
            day_ds = merged.isel({time_dim: list(indices)})
            day_ds.to_netcdf(str(nc_path))
            created.append(nc_path)

        merged.close()
        instant_path.unlink(missing_ok=True)
        if accum_path:
            accum_path.unlink(missing_ok=True)

        logger.info("Year batch %d done: %d/%d days written",
                    year, len(created) - len(existing), len(days_to_fetch))
        return created

    def _extract_station(
        self,
        nc_path: Path,
        station: WeatherStation,
        day: date,
    ) -> list[StationObservation]:
        """Extract nearest ERA5 grid point for a station from a downloaded NetCDF."""
        import numpy as np
        import xarray as xr

        ds = xr.open_dataset(str(nc_path))
        try:
            lats = ds.latitude.values
            lons = ds.longitude.values
            lat_idx = int(np.argmin(np.abs(lats - station.lat)))
            lon_idx = int(np.argmin(np.abs(lons - station.lon)))

            time_dim = "valid_time" if "valid_time" in ds.dims else "time"
            n_times = len(ds[time_dim])

            obs_list: list[StationObservation] = []
            for t_idx in range(n_times):
                raw_ts = ds[time_dim].values[t_idx]
                ts = pd.Timestamp(raw_ts).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                t2m_k = float(ds["t2m"].values[t_idx, lat_idx, lon_idx])
                t2m_c = t2m_k - 273.15

                d2m_k = float(ds["d2m"].values[t_idx, lat_idx, lon_idx])
                d2m_c = d2m_k - 273.15
                rh = _rh_from_t_td(t2m_c, d2m_c)

                u10 = float(ds["u10"].values[t_idx, lat_idx, lon_idx])
                v10 = float(ds["v10"].values[t_idx, lat_idx, lon_idx])
                wind_ms = math.sqrt(u10 ** 2 + v10 ** 2)

                # Extended variables — best-effort (may be missing in older cache files)
                def _safe_era5(var: str, scale: float = 1.0) -> float | None:
                    try:
                        v = float(ds[var].values[t_idx, lat_idx, lon_idx])
                        return round(v * scale, 2)
                    except (KeyError, IndexError):
                        return None

                # ERA5 tp is in metres; hourly slice → mm (optional — accumulated var)
                tp_raw = _safe_era5("tp")
                precip_mm = max(0.0, tp_raw * 1000.0) if tp_raw is not None else 0.0

                # ssrd is hourly accumulation (J/m²); divide by 3600 → W/m²
                solar = _safe_era5("ssrd", 1.0 / 3600.0)
                solar = max(0.0, solar) if solar is not None else None

                tcc = _safe_era5("tcc")
                tcc = min(1.0, max(0.0, tcc)) if tcc is not None else None

                blh = _safe_era5("blh")
                msl_pa = _safe_era5("msl")
                pressure_hpa = round(msl_pa / 100.0, 1) if msl_pa is not None else None

                obs_list.append(
                    StationObservation(
                        station_id=station.station_id,
                        ts_utc=ts,
                        temp_c=round(t2m_c, 2),
                        rh=round(rh, 1),
                        wind_ms=round(wind_ms, 2),
                        precip_mm=round(precip_mm, 2),
                        source="era5",
                        solar_wm2=solar,
                        cloud_cover=tcc,
                        blh_m=blh,
                        pressure_hpa=pressure_hpa,
                    )
                )
        finally:
            ds.close()

        logger.debug("Extracted %d observations for %s %s", len(obs_list), station.station_id, day)
        return obs_list


def _require_libs() -> None:
    missing = []
    for lib in ("cdsapi", "xarray", "netCDF4"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        raise ImportError(
            f"ERA5 ingestion requires: pip install {' '.join(missing)}\n"
            "Then register at https://cds.climate.copernicus.eu/ and create ~/.cdsapirc"
        )
