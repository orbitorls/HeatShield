"""Bias Calibration Module — moved to app.ml.calibration.

This file is a backward-compatible shim. Import from app.ml.calibration instead.
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "app.core.calibration has moved to app.ml.calibration. "
    "Import from app.ml.calibration instead.",
    DeprecationWarning,
    stacklevel=2,
)

from app.ml.calibration import (  # noqa: F401
    Calibration,
    CalibrationLevel,
    fit_calibration,
    select_calibration,
)
