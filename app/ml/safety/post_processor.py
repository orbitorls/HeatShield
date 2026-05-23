"""SafetyPostProcessor — post-processing layer for forecast safety.

Applies per-station, per-horizon safety rules after model inference:
1. Adaptive blending with persistence (based on skill score)
2. Danger recall boost for at-risk stations
3. PI sanity clamp
"""
from __future__ import annotations

import logging
import numpy as np
from app.ml.safety.config import get_safety_config, StationHorizonConfig
from app.ml.ensemble_blending import blend_with_persistence

logger = logging.getLogger(__name__)

_DEFAULT_WARNING_THRESHOLD = 38.0


class SafetyPostProcessor:
    """Post-processes model predictions to improve safety and robustness.

    Pipeline steps applied in order:
        1. Persistence blending — blend model with last-observed HI
        2. Danger recall boost — push mean higher when HI >= threshold
        3. PI sanity clamp — enforce sensible bounds

    Designed as a stateless processor — one instance per horizon per call.
    """

    def __init__(self, station_id: str, horizon_h: int) -> None:
        self.station_id = station_id
        self.horizon_h = horizon_h
        self.config: StationHorizonConfig = get_safety_config(station_id, horizon_h)

    def process(
        self,
        *,
        hi_mean: float,
        hi_lower: float,
        hi_upper: float,
        last_observed_hi: float | None = None,
        danger_proba: float | None = None,
    ) -> tuple[float, float, float, str | None]:
        """Run the post-processing pipeline on a single forecast point.

        Args:
            hi_mean: Point forecast heat index (°C).
            hi_lower: Lower prediction interval bound (°C).
            hi_upper: Upper prediction interval bound (°C).
            last_observed_hi: Most recent observed heat index for blending.
            danger_proba: P(tier=2) from DangerGate (0-1), or None.

        Returns:
            (hi_mean, hi_lower, hi_upper, extra_reason)
            where extra_reason is a short string describing any modification.
        """
        extra_reason: str | None = None
        reasons: list[str] = []

        hi_mean_f = float(hi_mean)
        hi_lower_f = float(hi_lower)
        hi_upper_f = float(hi_upper)

        # ----------------------------------------------------------------
        # Step 1: Persistence blending
        # ----------------------------------------------------------------
        if self.config.blend_weight < 1.0 and last_observed_hi is not None:
            blended = blend_with_persistence(
                np.array([hi_mean_f]),
                np.array([float(last_observed_hi)]),
                weight_model=self.config.blend_weight,
            )
            hi_mean_f = float(blended[0])
            reasons.append(f"blend_w={self.config.blend_weight:.1f}")

        # ----------------------------------------------------------------
        # Step 2: Danger recall boost
        # ----------------------------------------------------------------
        if self.config.danger_boost > 0.0 and hi_mean_f >= _DEFAULT_WARNING_THRESHOLD:
            gap = hi_upper_f - hi_mean_f
            boost = gap * self.config.danger_boost
            hi_mean_f = hi_mean_f + boost
            # Clamp: never exceed upper PI bound
            hi_mean_f = min(hi_mean_f, hi_upper_f)
            reasons.append(f"danger_boost={boost:.2f}C")

        # ----------------------------------------------------------------
        # Step 3: PI sanity clamp
        # ----------------------------------------------------------------
        pi_width = hi_upper_f - hi_lower_f
        if pi_width > 20.0 or pi_width < 0.0:
            half_width = 5.0
            hi_lower_f = hi_mean_f - half_width
            hi_upper_f = hi_mean_f + half_width
            reasons.append(f"pi_clamped(w={pi_width:.1f})")
        elif hi_lower_f > hi_mean_f:
            hi_lower_f = hi_mean_f - 1.0
            reasons.append("pi_lower>mean")
        elif hi_upper_f < hi_mean_f:
            hi_upper_f = hi_mean_f + 1.0
            reasons.append("pi_upper<mean")

        if reasons:
            extra_reason = "; ".join(reasons)

        return hi_mean_f, hi_lower_f, hi_upper_f, extra_reason
