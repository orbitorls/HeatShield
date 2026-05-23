"""External API clients for HeatShield AI data sources."""
from app.data.clients.circuit_breaker import CircuitBreaker, CircuitOpenError, get_tmd_circuit
from app.data.clients.tmd_client import TMDClient, TMDAPIError, TMDAuthError
from app.data.clients.nasa_power_client import NASAPowerClient
from app.data.clients.era5_client import ERA5Client
from app.data.clients.modis_client import MODISClient

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "get_tmd_circuit",
    "TMDClient",
    "TMDAPIError",
    "TMDAuthError",
    "NASAPowerClient",
    "ERA5Client",
    "MODISClient",
]
