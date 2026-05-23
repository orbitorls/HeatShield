"""Shim — use scripts.ingest.modis instead."""
import warnings as _warnings
_warnings.warn(
    "scripts.ingest.ingest_modis has moved to scripts.ingest.modis",
    DeprecationWarning, stacklevel=2,
)
from scripts.ingest.modis import *  # noqa: F401, F403
