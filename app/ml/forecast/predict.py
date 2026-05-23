"""Heat-index forecast using the trained XGBoost model.

Loads the latest model via registry, builds features from recent observations,
returns ForecastPoint list for each requested horizon.

low_confidence is set True when:
  - prediction interval width > 4°C  (uncertain forecast)
  - newest observation is older than 1 hour (stale data)

v2 models carry separate quantile boosters (q05, q95) trained with
quantile regression loss, giving per-forecast prediction intervals.
v1 models fall back to a flat PI derived from training metadata.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from app.ml.calibration import Calibration  # bias-correction module / โมดูลแก้อคติ
from app.ml.safety import SafetyPostProcessor, MAX_SUPPORTED_HORIZON
from app.data.schemas import ForecastPoint, StationObservation
from app.ml.forecast.features import (
    build_features,
    _DEFAULT_LAGS_H,
    _DEFAULT_ROLLING_H,
    _get_lags_for_horizon,
    _get_rolling_for_horizon,
    calculate_required_history_hours,
)
from app.ml.registry import load_latest

# Root directory of v3 model artifacts — ตำแหน่งไฟล์โมเดล v3
_V3_ROOT = Path(__file__).resolve().parents[3] / "app" / "models" / "forecast_v3"

logger = logging.getLogger(__name__)

_PI_WIDTH_THRESHOLD = 4.0  # degrees C — flag low confidence above this
_STALE_MINUTES = 60         # minutes — flag low confidence if newest obs is older than this


def _compute_last_hi(obs_sorted: list) -> float | None:
    """Compute the last observed heat index from recent observations.

    Uses Rothfusz equation: HI from temp_c and rh.
    Returns None if temp_c or rh is missing.
    """
    if not obs_sorted:
        return None
    last = obs_sorted[-1]
    if last.temp_c is None or last.rh is None:
        return None
    from app.core.heat_index import compute as _hi_compute
    return float(_hi_compute(last.temp_c, last.rh).heat_index)


def _ensemble_predict(boosters: list, dmat) -> float:
    """Average prediction across an ensemble of boosters."""
    return float(np.mean([b.predict(dmat)[0] for b in boosters]))


def _load_calibration(station_id: str, horizon_h: int) -> Calibration | None:
    """โหลด calibration.json สำหรับ (station, horizon) ถ้ามี — คืน None ถ้าไม่มีไฟล์.

    Load calibration artifact for a (station, horizon) pair if it exists.
    Returns None silently when the file is absent so callers can skip gracefully.
    """
    calib_path = _V3_ROOT / station_id / f"h{horizon_h}" / "calibration.json"
    if not calib_path.exists():
        return None
    try:
        with calib_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return Calibration.from_json(payload)
    except Exception as exc:  # noqa: BLE001 — ไม่ให้ calibration ที่เสียทำให้ crash
        logger.warning(
            "Could not load calibration for %s h%d — skipping: %s",
            station_id, horizon_h, exc,
        )
        return None


def predict(
    station_id: str,
    recent_obs: list[StationObservation],
    horizons: list[int] | None = None,
) -> tuple[list[ForecastPoint], bool, str | None]:
    """Forecast heat index for requested horizons.

    Args:
        station_id: Station identifier.
        recent_obs: Recent observations. Must cover at least max(lags_h)+1 hours for the longest horizon.
        horizons: Forecast horizons in hours.

    Returns:
        (forecasts, low_confidence, confidence_reason)

    Raises:
        FileNotFoundError: If no trained model exists (run train_forecast.py first).
        ValueError: If recent_obs is insufficient for feature engineering.
    """
    if horizons is None:
        horizons = [6, 24, 48]

    # Calculate required history based on the longest requested horizon
    required_history_hours = calculate_required_history_hours(horizons)

    if len(recent_obs) < required_history_hours + 1:
        max_horizon = max(horizons)
        raise ValueError(
            f"Need at least {required_history_hours + 1} recent observations for h{max_horizon} forecast (got {len(recent_obs)}). "
            f"Provide at least {required_history_hours} hours of hourly data."
        )

    loaded = load_latest("forecast")
    # load_latest returns a dict for v1, v2 single-horizon, and v2 multi-horizon.
    # Multi-horizon: loaded["horizons"] = {6: {"mean": [...], "q05": [...], "q95": [...]}, ...}
    # Single-horizon v2: loaded["mean"], loaded["q05"], loaded["q95"] at top level.
    # v1: loaded["booster"] only.
    metadata = loaded["metadata"]
    model_version = f"forecast_v{metadata.get('version', '?')}"
    is_multi_horizon = "horizons" in loaded

    # For single-horizon models (v1 or v2), extract boosters once at load time.
    # For multi-horizon models, boosters are dispatched per-horizon inside the loop.
    if not is_multi_horizon:
        _default_mean_boosters: list = loaded.get("mean", [loaded["booster"]])
        _default_q05_boosters: list = loaded.get("q05", [])
        _default_q95_boosters: list = loaded.get("q95", [])

    # Sort and convert to DataFrame
    obs_sorted = sorted(recent_obs, key=lambda o: o.ts_utc)
    df = pd.DataFrame([o.model_dump() for o in obs_sorted])
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["station_id"] = station_id

    # Stale data check — ensure timezone-aware comparison
    now_utc = datetime.now(timezone.utc)
    newest_ts = pd.to_datetime(df["ts_utc"].max())
    if newest_ts.tzinfo is None:
        newest_ts = newest_ts.replace(tzinfo=timezone.utc)
    staleness_min = (now_utc - newest_ts).total_seconds() / 60
    low_confidence = False
    confidence_reason: str | None = None

    if staleness_min > _STALE_MINUTES:
        low_confidence = True
        confidence_reason = f"Most recent observation is {staleness_min:.0f} min old (>{_STALE_MINUTES} min threshold)"
        logger.warning("Stale data: %s", confidence_reason)

    # ------------------------------------------------------------------
    # v3 path: route through model-agnostic BaseForecaster if available
    # ------------------------------------------------------------------
    from app.ml import registry as _reg
    _last_obs = obs_sorted[-1]
    _results_v3: list[ForecastPoint] = []
    for h in horizons:
        try:
            _forecaster = _reg.load_latest_v3(station_id, h)
            # Use horizon-specific lags/rolling for feature building
            _horizon_lags = _get_lags_for_horizon(h)
            _horizon_rolling = _get_rolling_for_horizon(h)
            try:
                _X_v3, _ = build_features(
                    df,
                    horizon_h=h,
                    lags_h=_horizon_lags,
                    rolling_h=_horizon_rolling,
                )
            except (ValueError, KeyError) as exc:
                logger.warning("v3 forecaster failed for station=%s h=%d: %s", station_id, h, exc)
                break
            if _X_v3.empty:
                break
            _bundle = _forecaster.predict_with_pi(_X_v3, alpha=0.10)
            _hi = float(_bundle.hi_mean[-1])
            _lo = float(_bundle.hi_lower[-1])
            _hi_up = float(_bundle.hi_upper[-1])
            _pi_width = _hi_up - _lo
            _meta = getattr(_forecaster, '_metadata', {})
            _med_pi = float(_meta.get('median_pi_width', 4.0))
            _low_conf_h = bool(_pi_width > 1.6 * _med_pi)
            _temp_v3 = (
                round(float(_bundle.temp_mean[-1]), 1)
                if _bundle.temp_mean is not None
                else round(_last_obs.temp_c, 1)
            )
            _rh_v3 = (
                round(float(_bundle.rh_mean[-1]), 1)
                if _bundle.rh_mean is not None
                else round(_last_obs.rh, 1)
            )
            _forecast_ts_v3 = newest_ts + timedelta(hours=h)

            # ----------------------------------------------------------------
            # Calibration (v3 path only) — แก้อคติของ q50 แล้วเลื่อน PI ตาม delta
            # Apply bias correction to q50; shift q05/q95 by the same delta
            # so the prediction-interval width is preserved exactly.
            # Skip silently when calibration.json is absent for this pair.
            # ----------------------------------------------------------------
            _calib = _load_calibration(station_id, h)
            if _calib is not None:
                _forecast_hour = int(_forecast_ts_v3.hour)  # ชั่วโมงของจุดพยากรณ์
                _hi_cal = float(
                    _calib.apply(
                        [_hi],           # y_pred (scalar wrapped in list)
                        [station_id],    # station_ids
                        [h],             # horizons
                        [_forecast_hour],# hours (UTC hour of the forecast target)
                    )[0]
                )
                _delta = _hi_cal - _hi   # shift ที่ใช้เลื่อน PI ด้วย / PI shift delta
                _hi = _hi_cal
                _lo = _lo + _delta       # เลื่อน lower bound ตาม delta / preserve PI width
                _hi_up = _hi_up + _delta # เลื่อน upper bound ตาม delta / preserve PI width

            # ----------------------------------------------------------------
            # Safety PostProcessor — blending + danger boost + PI sanity
            # ----------------------------------------------------------------
            _last_hi = _compute_last_hi(obs_sorted) if h <= MAX_SUPPORTED_HORIZON else None
            if h <= MAX_SUPPORTED_HORIZON:
                _pp = SafetyPostProcessor(station_id, h)
                _hi, _lo, _hi_up, _pp_reason = _pp.process(
                    hi_mean=_hi,
                    hi_lower=_lo,
                    hi_upper=_hi_up,
                    last_observed_hi=_last_hi,
                    danger_proba=float(_bundle.danger_proba[-1]) if _bundle.danger_proba is not None else None,
                )
                if _pp_reason and low_confidence:
                    confidence_reason = f"{confidence_reason}; post_process: {_pp_reason}"
                elif _pp_reason:
                    low_confidence = True
                    confidence_reason = f"post_process: {_pp_reason}"
            else:
                _pp_reason = f"horizon h{h} > max ({MAX_SUPPORTED_HORIZON}h) — blocked"
                if not low_confidence:
                    low_confidence = True
                    confidence_reason = _pp_reason

            _results_v3.append(ForecastPoint(
                station_id=station_id,
                ts_utc=_forecast_ts_v3,
                horizon_h=h,
                heat_index_c=round(_hi, 2),
                temp_c=_temp_v3,
                rh=_rh_v3,
                pi_lower=round(_lo, 2),
                pi_upper=round(_hi_up, 2),
                model_version=f"forecast_v3_{getattr(_forecaster, 'backend_name', 'unknown')}",
            ))
            if _low_conf_h and not low_confidence:
                low_confidence = True
                confidence_reason = (
                    f"v3 PI width {_pi_width:.1f}°C > 1.6× median ({_med_pi:.1f}°C)"
                )
        except (FileNotFoundError, KeyError, AttributeError):
            break  # no v3 for this horizon → fall through to v2 path entirely

    if len(_results_v3) == len(horizons):
        return _results_v3, low_confidence, confidence_reason
    # else: fall through to v2/v1 XGBoost path below
    # ------------------------------------------------------------------

    forecasts: list[ForecastPoint] = []

    import xgboost as xgb

    for horizon_h in horizons:
        # Dispatch boosters per horizon.
        # Multi-horizon model: use per-horizon ensemble when available; fall back to
        # shim booster for horizons not present in the model.
        # Single-horizon v2 / v1: use the same boosters for every horizon
        # (different feature-engineering targets — legacy behavior).
        if is_multi_horizon:
            horizon_data: dict = loaded["horizons"]
            if horizon_h in horizon_data:
                mean_boosters: list = horizon_data[horizon_h]["mean"]
                q05_boosters: list = horizon_data[horizon_h].get("q05", [])
                q95_boosters: list = horizon_data[horizon_h].get("q95", [])
            else:
                logger.warning(
                    "Horizon %dh not found in multi-horizon model; falling back to shim booster.",
                    horizon_h,
                )
                mean_boosters = [loaded["booster"]]
                q05_boosters = []
                q95_boosters = []
        else:
            mean_boosters = _default_mean_boosters
            q05_boosters = _default_q05_boosters
            q95_boosters = _default_q95_boosters

        try:
            X, _ = build_features(
                df.copy(),
                horizon_h=horizon_h,
                lags_h=_DEFAULT_LAGS_H,
                rolling_h=_DEFAULT_ROLLING_H,
            )
        except Exception as exc:
            logger.error("Feature build failed for horizon=%dh: %s", horizon_h, exc)
            continue

        if X.empty:
            logger.warning("No valid feature rows for horizon=%dh", horizon_h)
            continue

        # Align columns with what the model was trained on.
        # Extended columns (solar_wm2, cloud_cover, …) may be absent at inference —
        # fill with column median from training metadata, else 0.
        model_feature_list: list[str] = metadata.get("feature_list", list(X.columns))
        for col in model_feature_list:
            if col not in X.columns:
                fill = metadata.get("feature_medians", {}).get(col, 0.0)
                X[col] = fill
        X = X[model_feature_list]  # exact order required by XGBoost

        # Use the last row of features — that is the most recent observation's feature vector
        X_last = X.iloc[[-1]]
        dmat = xgb.DMatrix(X_last.values, feature_names=list(X.columns))

        # Mean prediction: average across seeds
        hi_pred = _ensemble_predict(mean_boosters, dmat)

        # Quantile PI: average across seeds, fall back to flat PI if no quantile boosters
        if q05_boosters and q95_boosters:
            pi_lower = _ensemble_predict(q05_boosters, dmat)
            pi_upper = _ensemble_predict(q95_boosters, dmat)
        else:
            # v1 / shim fallback: flat PI from training metadata
            p90_err = metadata.get("metrics", {}).get("p90_abs_err", 3.0)
            pi_lower = hi_pred - p90_err
            pi_upper = hi_pred + p90_err

        # Estimate temp_c and rh from recent trend (last observation as proxy)
        last_obs = obs_sorted[-1]
        estimated_temp = last_obs.temp_c
        estimated_rh = last_obs.rh

        pi_width = pi_upper - pi_lower
        if pi_width > _PI_WIDTH_THRESHOLD and not low_confidence:
            low_confidence = True
            confidence_reason = f"Prediction interval width {pi_width:.1f}°C > {_PI_WIDTH_THRESHOLD}°C threshold"

        forecast_ts = newest_ts + timedelta(hours=horizon_h)
        forecasts.append(ForecastPoint(
            station_id=station_id,
            ts_utc=forecast_ts,
            horizon_h=horizon_h,
            heat_index_c=round(hi_pred, 2),
            temp_c=round(estimated_temp, 1),
            rh=round(estimated_rh, 1),
            pi_lower=round(pi_lower, 2),
            pi_upper=round(pi_upper, 2),
            model_version=model_version,
        ))

    return forecasts, low_confidence, confidence_reason
