"""TMD (Thai Meteorological Department) API Client

Provides async client for accessing TMD weather data including:
- Current weather observations
- 7-day weather forecasts
- Weather station information
"""

import httpx
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class TMDClient:
    """Client for Thai Meteorological Department API

    API Documentation: https://data.tmd.go.th/api.html
    Note: Some endpoints may require API key registration
    """

    BASE_URL = "https://data.tmd.go.th/api"

    # Cache for API responses
    _cache: Dict[str, tuple[Any, float]] = {}
    _cache_ttl = 300  # 5 minutes default cache TTL

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 300,
        timeout: float = 30.0
    ):
        """Initialize TMD client.

        Args:
            api_key: Optional API key for TMD API (if required)
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=headers,
                timeout=self._timeout,
                follow_redirects=True
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        """Set value in cache with current timestamp."""
        self._cache[key] = (value, datetime.now().timestamp())

    def _invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries."""
        if pattern is None:
            self._cache.clear()
        else:
            keys_to_delete = [k for k in self._cache if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]

    async def get_current_weather(
        self,
        station_id: Optional[str] = None
    ) -> Dict:
        """Get current weather observation.

        Args:
            station_id: Weather station ID (optional, returns all if not specified)

        Returns:
            Dictionary containing current weather data including:
            - temperature, humidity, wind speed, conditions
        """
        cache_key = f"current_weather:{station_id or 'all'}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            client = await self._get_client()

            # TMD current weather endpoint
            if station_id:
                response = await client.get(
                    f"/api/weather/currentweather/v1",
                    params={"StationNumber": station_id}
                )
            else:
                response = await client.get("/api/weather/currentweather/v1")

            response.raise_for_status()
            data = response.json()

            result = self._parse_current_weather(data)
            self._set_cache(cache_key, result)
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"TMD API HTTP error: {e.response.status_code}")
            raise TMDError(f"API returned status {e.response.status_code}") from e
        except httpx.TimeoutException:
            logger.error("TMD API timeout")
            raise TMDError("API request timed out") from None
        except Exception as e:
            logger.error(f"TMD API error: {e}")
            raise TMDError(f"Failed to get current weather: {str(e)}") from e

    async def get_7day_forecast(
        self,
        province: Optional[str] = None
    ) -> List[Dict]:
        """Get 7-day weather forecast.

        Args:
            province: Province name in Thai (optional)

        Returns:
            List of daily forecast dictionaries containing:
            - date, temperature_min, temperature_max, conditions, precipitation
        """
        cache_key = f"forecast_7day:{province or 'all'}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            client = await self._get_client()

            # TMD 7-day forecast endpoint
            params = {}
            if province:
                params["Province"] = province

            response = await client.get(
                "/api/weather/7days-forecast/72hours",
                params=params
            )

            response.raise_for_status()
            data = response.json()

            result = self._parse_7day_forecast(data)
            self._set_cache(cache_key, result)
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"TMD API HTTP error: {e.response.status_code}")
            raise TMDError(f"API returned status {e.response.status_code}") from e
        except httpx.TimeoutException:
            logger.error("TMD API timeout")
            raise TMDError("API request timed out") from None
        except Exception as e:
            logger.error(f"TMD API error: {e}")
            raise TMDError(f"Failed to get forecast: {str(e)}") from e

    async def get_3hour_forecast(self) -> List[Dict]:
        """Get 3-hour interval forecast.

        Returns:
            List of 3-hourly forecast dictionaries
        """
        cache_key = "forecast_3hour"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            client = await self._get_client()
            response = await client.get("/api/weather/3hours-forecast")
            response.raise_for_status()
            data = response.json()

            result = self._parse_3hour_forecast(data)
            self._set_cache(cache_key, result)
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"TMD API HTTP error: {e.response.status_code}")
            raise TMDError(f"API returned status {e.response.status_code}") from e
        except httpx.TimeoutException:
            logger.error("TMD API timeout")
            raise TMDError("API request timed out") from None
        except Exception as e:
            logger.error(f"TMD API error: {e}")
            raise TMDError(f"Failed to get 3-hour forecast: {str(e)}") from e

    async def get_station_list(self) -> List[Dict]:
        """Get list of weather stations.

        Returns:
            List of station dictionaries containing:
            - station_id, name, name_th, lat, lon, province, region
        """
        cache_key = "station_list"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            client = await self._get_client()
            response = await client.get("/api/weather/stations")
            response.raise_for_status()
            data = response.json()

            result = self._parse_station_list(data)
            self._set_cache(cache_key, result)
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"TMD API HTTP error: {e.response.status_code}")
            raise TMDError(f"API returned status {e.response.status_code}") from e
        except httpx.TimeoutException:
            logger.error("TMD API timeout")
            raise TMDError("API request timed out") from None
        except Exception as e:
            logger.error(f"TMD API error: {e}")
            raise TMDError(f"Failed to get station list: {str(e)}") from e

    async def find_nearest_station(
        self,
        lat: float,
        lon: float
    ) -> Optional[Dict]:
        """Find nearest weather station to given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Nearest station dictionary or None if not found
        """
        stations = await self.get_station_list()
        if not stations:
            return None

        import math

        def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate distance between two points in km."""
            R = 6371  # Earth radius in km
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlon / 2) ** 2)
            return 2 * R * math.asin(math.sqrt(a))

        nearest = min(
            stations,
            key=lambda s: haversine(lat, lon, s.get("lat", 0), s.get("lon", 0))
        )
        return nearest

    def _parse_current_weather(self, data: Dict) -> Dict:
        """Parse TMD current weather response."""
        try:
            # TMD response structure varies, try common patterns
            if "Stations" in data:
                stations = data["Stations"]
                if isinstance(stations, list) and len(stations) > 0:
                    station = stations[0]
                    return {
                        "station_id": station.get("StationNumber", station.get("station_id")),
                        "station_name": station.get("StationNameThai", station.get("name", "")),
                        "temperature": station.get("Temperature", station.get("temperature", 0)),
                        "humidity": station.get("RelativeHumidity", station.get("humidity", 0)),
                        "wind_speed": station.get("WindSpeed", station.get("wind_speed", 0)),
                        "pressure": station.get("Pressure", station.get("pressure", 0)),
                        "conditions": station.get("Weather", station.get("conditions", "Unknown")),
                        "observation_time": station.get("ObservationDate", datetime.now().isoformat()),
                        "lat": station.get("Lat", 0),
                        "lon": station.get("Lon", 0)
                    }

            # Alternative response format
            if "data" in data:
                return self._parse_current_weather(data["data"])

            return data

        except Exception as e:
            logger.warning(f"Failed to parse current weather: {e}")
            return data

    def _parse_7day_forecast(self, data: Dict) -> List[Dict]:
        """Parse TMD 7-day forecast response."""
        try:
            forecasts = []

            # Look for common forecast data structures
            if "data" in data:
                data = data["data"]

            if "WeatherForecasts" in data:
                raw_forecasts = data["WeatherForecasts"]
            elif isinstance(data, list):
                raw_forecasts = data
            else:
                return []

            for item in raw_forecasts:
                forecast = {
                    "date": item.get("Date", item.get("date", "")),
                    "temperature_min": item.get("MinTemperature", item.get("min_temp", 0)),
                    "temperature_max": item.get("MaxTemperature", item.get("max_temp", 0)),
                    "conditions": item.get("WeatherDescription", item.get("conditions", "")),
                    "precipitation": item.get("Precipitation", item.get("precipitation", 0)),
                    "humidity": item.get("RelativeHumidity", item.get("humidity", 70)),
                    "wind_speed": item.get("WindSpeed", item.get("wind_speed", 0)),
                    "province": item.get("Province", item.get("province", "")),
                }
                forecasts.append(forecast)

            return forecasts

        except Exception as e:
            logger.warning(f"Failed to parse 7-day forecast: {e}")
            return []

    def _parse_3hour_forecast(self, data: Dict) -> List[Dict]:
        """Parse TMD 3-hour forecast response."""
        try:
            forecasts = []

            if "data" in data:
                data = data["data"]

            if "WeatherForecasts" in data:
                raw_forecasts = data["WeatherForecasts"]
            elif isinstance(data, list):
                raw_forecasts = data
            else:
                return []

            for item in raw_forecasts:
                forecast = {
                    "time": item.get("Time", item.get("time", "")),
                    "temperature": item.get("Temperature", item.get("temperature", 0)),
                    "humidity": item.get("RelativeHumidity", item.get("humidity", 0)),
                    "conditions": item.get("Weather", item.get("conditions", "")),
                    "precipitation": item.get("RainfallProbability", item.get("precipitation", 0)),
                    "wind_speed": item.get("WindSpeed", item.get("wind_speed", 0)),
                }
                forecasts.append(forecast)

            return forecasts

        except Exception as e:
            logger.warning(f"Failed to parse 3-hour forecast: {e}")
            return []

    def _parse_station_list(self, data: Dict) -> List[Dict]:
        """Parse TMD station list response."""
        try:
            stations = []

            if "data" in data:
                data = data["data"]

            if "Stations" in data:
                raw_stations = data["Stations"]
            elif isinstance(data, list):
                raw_stations = data
            else:
                return []

            for item in raw_stations:
                station = {
                    "station_id": item.get("StationNumber", item.get("station_id", "")),
                    "name": item.get("StationNameThai", item.get("name", "")),
                    "name_th": item.get("StationNameThai", item.get("name_th", "")),
                    "name_en": item.get("StationNameEng", item.get("name_en", "")),
                    "lat": float(item.get("Lat", item.get("lat", 0))),
                    "lon": float(item.get("Lon", item.get("lon", 0))),
                    "province": item.get("Province", item.get("province", "")),
                    "region": item.get("Region", item.get("region", "")),
                }
                stations.append(station)

            return stations

        except Exception as e:
            logger.warning(f"Failed to parse station list: {e}")
            return []


class TMDError(Exception):
    """Exception raised for TMD API errors."""
    pass


# Convenience function for quick access
async def get_tmd_client(
    api_key: Optional[str] = None,
    cache_ttl: int = 300
) -> TMDClient:
    """Create and return a TMD client instance."""
    return TMDClient(api_key=api_key, cache_ttl=cache_ttl)