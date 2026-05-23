"""MODIS Land Surface Temperature client via NASA APPEEARS API.

APPEEARS (Application for Extracting and Exploring Analysis Ready Samples)
provides time-series extraction for MODIS/VIIRS products at arbitrary points.

Authentication: Free NASA Earthdata account required.
  Register at: https://urs.earthdata.nasa.gov/
  Set env vars: EARTHDATA_USERNAME, EARTHDATA_PASSWORD
  OR add to ~/.netrc:
    machine urs.earthdata.nasa.gov login <user> password <pass>

Product used: MOD11A1.061 — MODIS Terra Land Surface Temperature & Emissivity (daily, 1 km)
  Layer: LST_Day_1km (scale factor 0.02, unit K → subtract 273.15 for °C)
  QC layer: QC_Day (quality flags; bit 0-1 = 00 means good data)

APPEEARS workflow (async — task runs server-side):
  1. POST /login           → bearer token
  2. POST /task            → task_id (point sample task)
  3. GET  /task/{task_id}  → poll until status == "done"
  4. GET  /bundle/{task_id} → list download files
  5. GET  /bundle/{task_id}/{filename} → download CSV
"""
from __future__ import annotations

import csv
import io
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Optional

import httpx

from app.data.schemas import StationObservation
from app.data.stations import STATIONS

logger = logging.getLogger(__name__)

_APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
_MODIS_PRODUCT = "MOD11A1.061"
_MODIS_LAYER_LST = "LST_Day_1km"
_MODIS_LAYER_QC = "QC_Day"
_LST_SCALE = 0.02      # Kelvin × 0.02 → actual Kelvin
_LST_FILL = 0          # fill value in scaled product
_POLL_INTERVAL_S = 30  # seconds between status polls
_MAX_WAIT_S = 3600     # 1 hour max wait


class MODISClient:
    """Extract MODIS daily LST for registered Thai stations via APPEEARS.

    Usage:
        client = MODISClient()  # reads EARTHDATA_* env vars
        obs = client.fetch_range("BKK_01", date(2024, 1, 1), date(2024, 3, 31))
        # obs: list[StationObservation] with lst_c set, one record per day per station
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout_s: float = 60.0,
    ):
        self.username = username or os.getenv("EARTHDATA_USERNAME", "")
        self.password = password or os.getenv("EARTHDATA_PASSWORD", "")
        self._token: str | None = None
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_range(
        self,
        station_id: str,
        start: date,
        end: date,
    ) -> list[StationObservation]:
        """Submit APPEEARS task and wait for MODIS LST data.

        Returns one StationObservation per available day (source="modis").
        Days with QC failure or cloud contamination are omitted.
        Note: MODIS Terra Day LST is ~10:30 local time — stored at 03:00 UTC
        (closest UTC hour for Thailand).
        """
        if station_id not in STATIONS:
            raise KeyError(f"Unknown station_id '{station_id}'")
        station = STATIONS[station_id]

        token = self._login()
        task_id = self._submit_task(token, station, start, end)
        logger.info("APPEEARS task submitted: %s (waiting for processing…)", task_id)
        csv_data = self._wait_and_download(token, task_id)
        return self._parse_csv(csv_data, station_id)

    def fetch_all_stations_range(
        self,
        start: date,
        end: date,
    ) -> dict[str, list[StationObservation]]:
        """Fetch MODIS LST for all 5 Thai stations in a single APPEEARS task.

        More efficient than 5 separate requests — APPEEARS processes all points together.
        Returns {station_id: [StationObservation, …]}.
        """
        token = self._login()
        task_id = self._submit_task_all_stations(token, start, end)
        logger.info("APPEEARS multi-station task: %s", task_id)
        csv_data = self._wait_and_download(token, task_id)
        return self._parse_csv_all(csv_data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _login(self) -> str:
        """Authenticate with APPEEARS and return bearer token."""
        if not self.username or not self.password:
            raise EnvironmentError(
                "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars, "
                "or register free at https://urs.earthdata.nasa.gov/"
            )
        resp = httpx.post(
            f"{_APPEEARS_BASE}/login",
            auth=(self.username, self.password),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        token = resp.json().get("token")
        if not token:
            raise RuntimeError(f"APPEEARS login failed: {resp.text[:200]}")
        logger.debug("APPEEARS authenticated")
        return token

    def _submit_task(
        self, token: str, station, start: date, end: date
    ) -> str:
        return self._submit_task_points(
            token,
            points=[{"id": station.station_id, "latitude": station.lat, "longitude": station.lon}],
            start=start,
            end=end,
            task_name=f"modis_lst_{station.station_id}_{start}_{end}",
        )

    def _submit_task_all_stations(self, token: str, start: date, end: date) -> str:
        points = [
            {"id": s.station_id, "latitude": s.lat, "longitude": s.lon}
            for s in STATIONS.values()
        ]
        return self._submit_task_points(
            token, points, start, end,
            task_name=f"modis_lst_all_{start}_{end}",
        )

    def _submit_task_points(
        self,
        token: str,
        points: list[dict],
        start: date,
        end: date,
        task_name: str,
    ) -> str:
        payload = {
            "task_type": "point",
            "task_name": task_name,
            "params": {
                "dates": [{"startDate": start.strftime("%m-%d-%Y"), "endDate": end.strftime("%m-%d-%Y")}],
                "layers": [
                    {"product": _MODIS_PRODUCT, "layer": _MODIS_LAYER_LST},
                    {"product": _MODIS_PRODUCT, "layer": _MODIS_LAYER_QC},
                ],
                "coordinates": points,
                "output": {"format": {"type": "csv"}, "projection": {"type": "geographic"}},
            },
        }
        resp = httpx.post(
            f"{_APPEEARS_BASE}/task",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["task_id"]

    def _wait_and_download(self, token: str, task_id: str) -> str:
        """Poll task until done, then download the CSV bundle."""
        headers = {"Authorization": f"Bearer {token}"}
        elapsed = 0
        while elapsed < _MAX_WAIT_S:
            status_resp = httpx.get(
                f"{_APPEEARS_BASE}/task/{task_id}",
                headers=headers,
                timeout=self.timeout_s,
            )
            status_resp.raise_for_status()
            status = status_resp.json().get("status", "")
            logger.info("APPEEARS task %s status: %s (%ds elapsed)", task_id, status, elapsed)

            if status == "done":
                break
            if status in ("error", "deleted"):
                raise RuntimeError(f"APPEEARS task {task_id} failed with status: {status}")

            time.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S
        else:
            raise TimeoutError(f"APPEEARS task {task_id} did not complete within {_MAX_WAIT_S}s")

        # List bundle files
        bundle_resp = httpx.get(
            f"{_APPEEARS_BASE}/bundle/{task_id}",
            headers=headers,
            timeout=self.timeout_s,
        )
        bundle_resp.raise_for_status()
        files = bundle_resp.json().get("files", [])

        # Find the LST CSV file
        lst_file = next(
            (f for f in files if _MODIS_LAYER_LST in f.get("file_name", "") and f["file_name"].endswith(".csv")),
            None,
        )
        if not lst_file:
            # Fall back to first CSV
            lst_file = next((f for f in files if f["file_name"].endswith(".csv")), None)
        if not lst_file:
            raise FileNotFoundError(f"No CSV in APPEEARS bundle {task_id}: {[f['file_name'] for f in files]}")

        dl_resp = httpx.get(
            f"{_APPEEARS_BASE}/bundle/{task_id}/{lst_file['file_id']}",
            headers=headers,
            timeout=300.0,
            follow_redirects=True,
        )
        dl_resp.raise_for_status()
        return dl_resp.text

    def _parse_csv(self, csv_text: str, station_id: str) -> list[StationObservation]:
        """Parse APPEEARS CSV for a single station."""
        all_obs = self._parse_csv_all(csv_text)
        return all_obs.get(station_id, [])

    def _parse_csv_all(self, csv_text: str) -> dict[str, list[StationObservation]]:
        """Parse APPEEARS CSV for all stations. Returns {station_id: [obs, …]}."""
        result: dict[str, list[StationObservation]] = {}
        reader = csv.DictReader(io.StringIO(csv_text))

        for row in reader:
            sid = row.get("ID") or row.get("id") or row.get("Category", "")
            if sid not in STATIONS:
                continue

            # Date column — APPEEARS format: "YYYY-MM-DD"
            date_str = row.get("Date") or row.get("date") or ""
            try:
                obs_date = date.fromisoformat(date_str[:10])
            except (ValueError, TypeError):
                continue

            # LST value (raw DN, needs × scale_factor then − 273.15)
            lst_raw = row.get(f"{_MODIS_PRODUCT}_{_MODIS_LAYER_LST}", "")
            try:
                lst_raw_f = float(lst_raw)
            except (ValueError, TypeError):
                continue
            if lst_raw_f == 0 or lst_raw_f < 7500:  # fill or unrealistically cold
                continue

            lst_k = lst_raw_f * _LST_SCALE
            lst_c = lst_k - 273.15
            if not (-10 <= lst_c <= 80):
                continue

            # Store at 03:00 UTC (Terra overpass ~10:30 local = 03:30 UTC for Thailand)
            ts = datetime(obs_date.year, obs_date.month, obs_date.day, 3, 0, 0, tzinfo=timezone.utc)

            if sid not in result:
                result[sid] = []

            station = STATIONS[sid]
            result[sid].append(
                StationObservation(
                    station_id=sid,
                    ts_utc=ts,
                    temp_c=lst_c,   # LST used as temp proxy for MODIS-only records
                    rh=50.0,        # unknown — placeholder (quality filter will drop if needed)
                    source="modis",
                    lst_c=round(lst_c, 2),
                )
            )

        return result
