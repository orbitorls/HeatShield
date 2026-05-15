"""Hard-coded registry of TMD weather stations used by HeatShield AI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeatherStation:
    station_id: str
    name_th: str
    lat: float
    lon: float
    elevation_m: float


STATIONS: dict[str, WeatherStation] = {
    # === Central Thailand (ภาคกลาง) ===
    "BKK_01": WeatherStation(
        station_id="BKK_01",
        name_th="กรุงเทพมหานคร (Don Mueang)",
        lat=13.9132,
        lon=100.6067,
        elevation_m=9.5,
    ),
    "NSW_01": WeatherStation(
        station_id="NSW_01",
        name_th="นครสวรรค์",
        lat=15.7031,
        lon=100.1372,
        elevation_m=35.0,
    ),
    "SPB_01": WeatherStation(
        station_id="SPB_01",
        name_th="สุพรรณบุรี",
        lat=14.4744,
        lon=100.1217,
        elevation_m=11.0,
    ),
    # === Northern Thailand (ภาคเหนือ) ===
    "CNX_01": WeatherStation(
        station_id="CNX_01",
        name_th="เชียงใหม่",
        lat=18.7761,
        lon=98.9769,
        elevation_m=310.0,
    ),
    "CEI_01": WeatherStation(
        station_id="CEI_01",
        name_th="เชียงราย",
        lat=19.9072,
        lon=99.8329,
        elevation_m=390.0,
    ),
    "LPT_01": WeatherStation(
        station_id="LPT_01",
        name_th="ลำปาง",
        lat=18.2916,
        lon=99.4878,
        elevation_m=235.0,
    ),
    # === Northeastern Thailand (ภาคตะวันออกเฉียงเหนือ/อีสาน) ===
    "KKN_01": WeatherStation(
        station_id="KKN_01",
        name_th="ขอนแก่น",
        lat=16.4419,
        lon=102.8359,
        elevation_m=182.0,
    ),
    "UDN_01": WeatherStation(
        station_id="UDN_01",
        name_th="อุดรธานี",
        lat=17.4139,
        lon=102.7784,
        elevation_m=179.0,
    ),
    "NMA_01": WeatherStation(
        station_id="NMA_01",
        name_th="นครราชสีมา (โคราช)",
        lat=14.9799,
        lon=102.0977,
        elevation_m=197.0,
    ),
    "UBN_01": WeatherStation(
        station_id="UBN_01",
        name_th="อุบลราชธานี",
        lat=15.2287,
        lon=104.8570,
        elevation_m=124.0,
    ),
    # === Eastern Thailand (ภาคตะวันออก) ===
    "RYG_01": WeatherStation(
        station_id="RYG_01",
        name_th="ระยอง",
        lat=12.6815,
        lon=101.2816,
        elevation_m=14.0,
    ),
    "JTI_01": WeatherStation(
        station_id="JTI_01",
        name_th="จันทบุรี",
        lat=12.6113,
        lon=102.1038,
        elevation_m=7.0,
    ),
    "CBI_01": WeatherStation(
        station_id="CBI_01",
        name_th="ชลบุรี",
        lat=13.3611,
        lon=100.9847,
        elevation_m=10.0,
    ),
    # === Southern Thailand (ภาคใต้) ===
    "HYI_01": WeatherStation(
        station_id="HYI_01",
        name_th="หาดใหญ่ (สงขลา)",
        lat=6.9269,
        lon=100.4370,
        elevation_m=8.0,
    ),
    "HKT_01": WeatherStation(
        station_id="HKT_01",
        name_th="ภูเก็ต",
        lat=8.1132,
        lon=98.2760,
        elevation_m=3.0,
    ),
    "NST_01": WeatherStation(
        station_id="NST_01",
        name_th="นครศรีธรรมราช",
        lat=8.4303,
        lon=99.9628,
        elevation_m=9.0,
    ),
}

# Regional grouping for convenience
STATIONS_BY_REGION: dict[str, list[str]] = {
    "central": ["BKK_01", "NSW_01", "SPB_01"],
    "northern": ["CNX_01", "CEI_01", "LPT_01"],
    "northeastern": ["KKN_01", "UDN_01", "NMA_01", "UBN_01"],
    "eastern": ["RYG_01", "JTI_01", "CBI_01"],
    "southern": ["HYI_01", "HKT_01", "NST_01"],
}


def get_station(station_id: str) -> WeatherStation:
    """Return a WeatherStation by ID.

    Raises:
        KeyError: if station_id is not in the registry.
    """
    if station_id not in STATIONS:
        known = ", ".join(sorted(STATIONS))
        raise KeyError(
            f"Unknown station_id '{station_id}'. Known stations: {known}"
        )
    return STATIONS[station_id]
