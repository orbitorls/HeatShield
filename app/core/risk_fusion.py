"""Risk-Fusion Module — moved to app.ml.risk_fusion.

This file is a backward-compatible shim. Import from app.ml.risk_fusion instead.
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "app.core.risk_fusion has moved to app.ml.risk_fusion. "
    "Import from app.ml.risk_fusion instead.",
    DeprecationWarning,
    stacklevel=2,
)

from app.ml.risk_fusion import (  # noqa: F401
    DEFAULT_THRESHOLDS,
    RiskLevel,
    RiskOutput,
    UncertaintyFlag,
    fuse_risk,
)
