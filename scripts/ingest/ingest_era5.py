"""Shim — use scripts.ingest.era5 instead."""
import warnings as _warnings
_warnings.warn(
    "scripts.ingest.ingest_era5 has moved to scripts.ingest.era5",
    DeprecationWarning, stacklevel=2,
)
from scripts.ingest.era5 import *  # noqa: F401, F403
