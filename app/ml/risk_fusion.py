"""
Risk-Fusion Module — รวม regression + quantile + classifier เป็น risk signal

Pure function ที่รวม output จากโมเดลพยากรณ์หลายชนิด (heat-index regression
แบบ q50/q90, classifier P(High-Watch) / P(Danger), local anomaly excess)
ให้กลายเป็น risk_level เดียวพร้อม reason_codes ที่ตรวจสอบย้อนกลับได้

Design goals:
  - ML-free, no I/O, no hidden globals → testable, deterministic
  - Explainable → reason_codes บอกทุก rule ที่ trip
  - Conservative when uncertain → uncertainty_flag จาก PI width

Combine regression / quantile / classifier outputs into a single
explainable risk signal. Pure function — no I/O, no logging side
effects, no global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Default thresholds — every knob explicitly documented
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict = {
    # q50 ≥ 42°C → Danger (median exceeds danger HI)
    "q50_danger": 42.0,
    # q90 ≥ 42°C (with q50 below) → Danger Watch
    "q90_danger": 42.0,
    # P(Danger) classifier ≥ this → trip P_DANGER_GATE reason
    "p_danger_gate": 0.20,
    # P(High-Watch) classifier ≥ this → trip P_HIGH_WATCH_GATE reason
    "p_high_watch_gate": 0.40,
    # PI width below this is "low" uncertainty
    "pi_width_low": 2.0,
    # PI width above this is "high" uncertainty
    "pi_width_high": 4.0,
    # Horizons (hours) classified as "awareness" instead of "operational"
    "awareness_horizons": {48, 72},
}


RiskLevel = Literal["Normal", "Caution", "Watch", "Warning", "Danger"]
HorizonType = Literal["operational", "awareness"]
UncertaintyFlag = Literal["low", "medium", "high"]


# Ordered ladder for monotonic escalation
_LEVEL_ORDER: tuple[RiskLevel, ...] = (
    "Normal",
    "Caution",
    "Watch",
    "Warning",
    "Danger",
)


@dataclass(frozen=True)
class RiskOutput:
    risk_level: RiskLevel
    confidence: float
    reason_codes: list[str]
    horizon_type: HorizonType
    uncertainty_flag: UncertaintyFlag


def _escalate(current: RiskLevel, target: RiskLevel) -> RiskLevel:
    """Move risk level up toward `target`; never demote."""
    cur_idx = _LEVEL_ORDER.index(current)
    tgt_idx = _LEVEL_ORDER.index(target)
    return _LEVEL_ORDER[max(cur_idx, tgt_idx)]


def _uncertainty_from_pi(pi_width: float, low: float, high: float) -> UncertaintyFlag:
    if pi_width < low:
        return "low"
    if pi_width > high:
        return "high"
    return "medium"


def _confidence_from_uncertainty(flag: UncertaintyFlag) -> float:
    return {"low": 0.85, "medium": 0.65, "high": 0.45}[flag]


def fuse_risk(
    *,
    hi_q50: float,
    hi_q90: float,
    p_high_watch: float,
    p_danger: float,
    hi_anomaly_p95_excess: float,
    horizon_h: int,
    pi_width: float,
    thresholds: dict | None = None,
) -> RiskOutput:
    """
    รวมผลลัพธ์จากหลายโมเดลให้เป็น risk signal เดียว (Explainable Fusion)

    Inputs:
      hi_q50: heat-index quantile p50 (°C, จาก quantile regressor)
      hi_q90: heat-index quantile p90 (°C, upper bound)
      p_high_watch: ความน่าจะเป็นเข้า class High-Watch ∈ [0,1]
      p_danger: ความน่าจะเป็นเข้า class Danger ∈ [0,1]
      hi_anomaly_p95_excess: ส่วนเกิน HI เหนือ station-month-hour p95 (°C);
                             >0 หมายถึง anomaly local
      horizon_h: ระยะพยากรณ์ (ชั่วโมง); 48/72 = awareness, อื่น ๆ = operational
      pi_width: ความกว้างของ prediction interval (q90-q05 หรือ q90-q50; °C)
      thresholds: dict override ของ DEFAULT_THRESHOLDS (optional)

    Returns RiskOutput พร้อม reason_codes อธิบายทุก rule ที่ทำงาน

    -----------------------------------------------------------------
    Fuse regression (q50), quantile (q90), classifier (P(HighWatch),
    P(Danger)) and a local anomaly excess into one explainable signal.

    Rule table:
      q50 >= q50_danger              -> Danger
      q90 >= q90_danger & q50 < ...  -> Watch (Danger Watch)
      P(Danger) >= gate              -> Watch (escalate to Warning if also q90 hot)
      P(HighWatch) >= gate           -> Caution (or Watch if other rules trip)
      anomaly_p95_excess > 0         -> add LOCAL_ANOMALY_P95 reason

    horizon_type = "awareness" if horizon_h in awareness_horizons else "operational"
    uncertainty_flag from pi_width: <low → "low", >high → "high", else "medium"
    """
    cfg: dict = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    reason_codes: list[str] = []
    risk_level: RiskLevel = "Normal"

    # --- Rule 1: q50 ≥ danger threshold → Danger -----------------------
    if hi_q50 >= cfg["q50_danger"]:
        risk_level = _escalate(risk_level, "Danger")
        reason_codes.append("Q50_DANGER_THRESHOLD")

    # --- Rule 2: q90 ≥ danger threshold (Danger Watch) -----------------
    # Even when q50 < threshold, an upper-tail breach is meaningful.
    if hi_q90 >= cfg["q90_danger"] and hi_q50 < cfg["q50_danger"]:
        risk_level = _escalate(risk_level, "Watch")
        reason_codes.append("Q90_DANGER_THRESHOLD")

    # --- Rule 3: P(Danger) gate ---------------------------------------
    if p_danger >= cfg["p_danger_gate"]:
        # If classifier is confident AND quantile already hot, push to Warning;
        # otherwise just Watch.
        if hi_q90 >= cfg["q90_danger"] or hi_q50 >= cfg["q50_danger"]:
            risk_level = _escalate(risk_level, "Warning")
        else:
            risk_level = _escalate(risk_level, "Watch")
        reason_codes.append("P_DANGER_GATE")

    # --- Rule 4: P(High-Watch) gate -----------------------------------
    if p_high_watch >= cfg["p_high_watch_gate"]:
        risk_level = _escalate(risk_level, "Caution")
        reason_codes.append("P_HIGH_WATCH_GATE")

    # --- Rule 5: Local anomaly (does NOT escalate by itself) ----------
    # The anomaly is informational — it adds context but never single-handedly
    # bumps level. It can co-occur with any other reason.
    if hi_anomaly_p95_excess > 0:
        reason_codes.append("LOCAL_ANOMALY_P95")

    # --- Horizon classification ---------------------------------------
    awareness = set(cfg["awareness_horizons"])
    horizon_type: HorizonType = "awareness" if horizon_h in awareness else "operational"

    # --- Uncertainty flag from PI width -------------------------------
    uncertainty_flag = _uncertainty_from_pi(
        pi_width, float(cfg["pi_width_low"]), float(cfg["pi_width_high"])
    )

    confidence = _confidence_from_uncertainty(uncertainty_flag)

    return RiskOutput(
        risk_level=risk_level,
        confidence=confidence,
        reason_codes=reason_codes,
        horizon_type=horizon_type,
        uncertainty_flag=uncertainty_flag,
    )


__all__ = ["DEFAULT_THRESHOLDS", "RiskOutput", "fuse_risk"]
