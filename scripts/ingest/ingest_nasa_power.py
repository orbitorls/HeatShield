"""Shim — use scripts.ingest.nasa_power instead."""
import warnings as _warnings
_warnings.warn(
    "scripts.ingest.ingest_nasa_power has moved to scripts.ingest.nasa_power",
    DeprecationWarning, stacklevel=2,
)
from scripts.ingest.nasa_power import *  # noqa: F401, F403
