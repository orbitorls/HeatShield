"""Shim — use scripts.ingest.tmd instead."""
import warnings as _warnings
_warnings.warn(
    "scripts.ingest.ingest_tmd has moved to scripts.ingest.tmd",
    DeprecationWarning, stacklevel=2,
)
from scripts.ingest.tmd import *  # noqa: F401, F403
