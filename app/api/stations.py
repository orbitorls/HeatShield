from dataclasses import asdict

from fastapi import APIRouter

from app.data.stations import STATIONS, STATIONS_BY_REGION

router = APIRouter()


@router.get("", summary="List all weather stations")
def list_stations() -> list[dict]:
    """Return all Thailand weather stations with Thai names and regions."""
    stations_list = []
    for station in STATIONS.values():
        station_dict = asdict(station)
        # Add region information
        for region, station_ids in STATIONS_BY_REGION.items():
            if station.station_id in station_ids:
                station_dict["region"] = region
                station_dict["region_th"] = {
                    "central": "ภาคกลาง",
                    "northern": "ภาคเหนือ",
                    "northeastern": "ภาคตะวันออกเฉียงเหนือ",
                    "eastern": "ภาคตะวันออก",
                    "southern": "ภาคใต้",
                }.get(region)
                break
        stations_list.append(station_dict)
    return stations_list
