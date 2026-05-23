"""Shim — use scripts.ingest.run_all instead."""
import warnings as _warnings
_warnings.warn(
    "scripts.ingest.ingest_all has moved to scripts.ingest.run_all",
    DeprecationWarning, stacklevel=2,
)
from scripts.ingest.run_all import *  # noqa: F401, F403
