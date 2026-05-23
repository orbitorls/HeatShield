"""Shim — use scripts.eval.run instead."""
import warnings as _warnings
_warnings.warn(
    "scripts.eval.evaluate_model has moved to scripts.eval.run",
    DeprecationWarning, stacklevel=2,
)
from scripts.eval.run import *  # noqa: F401, F403
