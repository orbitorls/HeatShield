"""NASA POWER API client — MERRA-2 based hourly weather data.

No authentication required. Free for research use.
Coverage: 1981 to ~3 days ago, global, point extraction.

Docs: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/

Parameters fetched:
  T2M            — Temperature at 2m (°C)
  RH2M           — Relative Humidity at 2m (%)
  WS10M          — Wind Speed at 10m (m/s)
  PRECTOTCORR    — Bias-corrected precipitation (mm/hr)
  ALLSKY_SFC_SW_DWN — All-sky solar irradiance (W/m²)
  CLOUD_AMT      — Cloud amount (% sky cover → stored as 0–1)
  PS             — Surface pressure (kPa → stored as hPa)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from app.data.schemas import StationObservation
from app.data.stations import STATIONS

logger = logging.getLogger(__name__)

_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
_PARAMETERS = "T2M,RH2M,WS10M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,CLOUD_AMT,PS"
_FILL_VALUE = -999.0  # POWER API sentinel for missing data


class NASAPowerClient:
    """Fetch NASA POWER (MERRA-2) hourly data for a single station-day.

    Rate limit: ~30 requests/minute. Default timeout 120s (CDS server can be slow).
    """

    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s

    def fetch_day(self, station_id: str, day: date) -> list[StationObservation]:
        """Download one station-day from NASA POWER.

        Args:
            station_id: Registered station ID (from STATIONS registry).
            day: Calendar date (UTC).

        Returns:
            Up to 24 hourly StationObservation records (source="nasa_power").
        """
        if station_id not in STATIONS:
            raise KeyError(f"Unknown station_id '{station_id}'")
        station = STATIONS[station_id]

        params = {
            "parameters": _PARAMETERS,
            "community": "RE",
            "longitude": station.lon,
            "latitude": station.lat,
            "start": day.strftime("%Y%m%d"),
            "end": day.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.get(_BASE_URL, params=params)

        if resp.status_code != 200:
            raise RuntimeError(
                f"NASA POWER returned {resp.status_code} for {station_id} {day}: {resp.text[:200]}"
            )

        return self._parse(resp.json(), station_id, day)

    def fetch_range(
        self,
        station_id: str,
        start: date,
        end: date,
    ) -> list[StationObservation]:
        """Fetch a multi-day range in one API call (more efficient than day-by-day).

        NASA POWER allows up to ~1-year per request.
        """
        if station_id not in STATIONS:
            raise KeyError(f"Unknown station_id '{station_id}'")
        station = STATIONS[station_id]

        params = {
            "parameters": _PARAMETERS,
            "community": "RE",
            "longitude": station.lon,
            "latitude": station.lat,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }

        logger.info("Fetching NASA POWER: station=%s %s → %s", station_id, start, end)
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.get(_BASE_URL, params=params)

        if resp.status_code != 200:
            raise RuntimeError(
                f"NASA POWER returned {resp.status_code}: {resp.text[:300]}"
            )

        return self._parse_range(resp.json(), station_id, start, end)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse(self, payload: dict, station_id: str, day: date) -> list[StationObservation]:
        """Parse single-day JSON response."""
        return self._parse_range(payload, station_id, day, day)

    def _parse_range(
        self, payload: dict, station_id: str, start: date, end: date
    ) -> list[StationObservation]:
        """Parse multi-day POWER API JSON response into StationObservation list."""
        try:
            params_data: dict[str, dict] = (
                payload.get("properties", {}).get("parameter", {})
                or payload.get("features", [{}])[0].get("properties", {}).get("parameter", {})
            )
        except (IndexError, AttributeError):
            logger.warning("Unexpected NASA POWER response structure")
            return []

        if not params_data:
            logger.warning("Empty NASA POWER response for %s", station_id)
            return []

        obs_list: list[StationObservation] = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            for hour in range(24):
                # API key format: YYYYMMDDHH (zero-padded hour)
                dt_key = f"{date_str}{hour:02d}"

                def _get(param: str, _key: str = dt_key) -> float | None:
                    v = params_data.get(param, {}).get(_key)
                    if v is None or float(v) <= _FILL_VALUE:
                        return None
                    return float(v)

                t2m = _get("T2M")
                rh2m = _get("RH2M")

                if t2m is None or rh2m is None:
                    continue  # skip hours with missing core data

                ts = datetime(current.year, current.month, current.day, hour, 0, 0, tzinfo=timezone.utc)
                solar = _get("ALLSKY_SFC_SW_DWN")
                cloud_pct = _get("CLOUD_AMT")
                ps_kpa = _get("PS")

                obs_list.append(
                    StationObservation(
                        station_id=station_id,
                        ts_utc=ts,
                        temp_c=round(t2m, 2),
                        rh=round(min(100.0, max(0.0, rh2m)), 1),
                        wind_ms=round(_get("WS10M") or 0.0, 2) or None,
                        precip_mm=round(max(0.0, _get("PRECTOTCORR") or 0.0), 2),
                        source="nasa_power",
                        solar_wm2=round(solar, 1) if solar is not None and solar >= 0 else None,
                        cloud_cover=round(cloud_pct / 100.0, 3) if cloud_pct is not None else None,
                        pressure_hpa=round(ps_kpa * 10.0, 1) if ps_kpa is not None else None,
                    )
                )
            current += timedelta(days=1)

        return obs_list
