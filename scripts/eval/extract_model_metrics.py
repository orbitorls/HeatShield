"""Shim — use scripts.eval.extract_metrics instead."""
import warnings as _warnings
_warnings.warn(
    "scripts.eval.extract_model_metrics has moved to scripts.eval.extract_metrics",
    DeprecationWarning, stacklevel=2,
)
from scripts.eval.extract_metrics import *  # noqa: F401, F403
