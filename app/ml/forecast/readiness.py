"""Shared readiness gate logic for training and status checks."""
from __future__ import annotations

from dataclasses import dataclass


SKILL_MIN_BY_HORIZON = {6: -0.02, 12: -0.02, 24: -0.05, 48: -3.0, 72: -3.5}
MAE_MAX_BY_HORIZON = {6: 2.0, 12: 2.2, 24: 2.5, 48: 3.0, 72: 3.5}
NOT_READY_SKILL_FLOOR_BY_HORIZON = {6: -0.5, 12: -0.5, 24: -0.5, 48: -3.5, 72: -4.0}
PI_COVERAGE_FLOOR_BY_HORIZON = {6: 0.80, 12: 0.80, 24: 0.80, 48: 0.65, 72: 0.65}
DANGER_MIN_BY_HORIZON = {6: 0.35, 12: 0.35, 24: 0.35, 48: 0.10, 72: 0.05}
DANGER_SUPPORT_WAIVER = 100


@dataclass(frozen=True)
class ReadinessResult:
    """Decision record for readiness classification."""

    status: str
    failed_reasons: tuple[str, ...]
    skill_score: float | None
    mae: float | None
    danger_recall_42: float | None
    danger_42_support: int | None
    pi_coverage_90: float | None


def thresholds_for_horizon(horizon_h: int) -> dict[str, float]:
    """Return readiness thresholds for a forecast horizon."""
    return {
        "skill_min": SKILL_MIN_BY_HORIZON.get(horizon_h, -0.02),
        "mae_max": MAE_MAX_BY_HORIZON.get(horizon_h, 2.0),
        "not_ready_skill_floor": NOT_READY_SKILL_FLOOR_BY_HORIZON.get(horizon_h, -0.5),
        "pi_coverage_floor": PI_COVERAGE_FLOOR_BY_HORIZON.get(horizon_h, 0.80),
        "pi_coverage_ceiling": 0.98,
        "danger_min": DANGER_MIN_BY_HORIZON.get(horizon_h, 0.35),
    }


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _metric(metrics: dict, *path: str) -> float | int | None:
    cur = metrics
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def evaluate_readiness(metrics: dict, horizon_h: int) -> ReadinessResult:
    """Evaluate readiness status and deterministic failure reasons."""
    t = thresholds_for_horizon(horizon_h)
    skill = _metric(metrics, "baselines", "skill_score")
    if skill is None:
        skill = metrics.get("skill_score")
    mae = _metric(metrics, "regression", "mae")
    danger_recall = _metric(metrics, "safety", "danger_42", "recall")
    danger_support = _metric(metrics, "safety", "danger_42", "support")
    pi_available = bool(_metric(metrics, "prediction_interval", "available"))
    pi_coverage = _metric(metrics, "prediction_interval", "coverage_90") if pi_available else None

    reasons: list[str] = []

    if skill is not None and float(skill) <= float(t["skill_min"]):
        reasons.append(f"skill<={_fmt_num(float(t['skill_min']))}")
    if mae is not None and float(mae) >= float(t["mae_max"]):
        reasons.append(f"mae>={_fmt_num(float(t['mae_max']))}")
    if pi_coverage is not None and not (t["pi_coverage_floor"] <= float(pi_coverage) <= t["pi_coverage_ceiling"]):
        reasons.append(
            f"pi_cov_not_in[{_fmt_num(float(t['pi_coverage_floor']))},{_fmt_num(float(t['pi_coverage_ceiling']))}]"
        )

    danger_waived = False
    if danger_support is not None and int(danger_support) < DANGER_SUPPORT_WAIVER:
        danger_waived = True
    if danger_recall is not None and not danger_waived and float(danger_recall) < float(t["danger_min"]):
        reasons.append(f"danger<{float(t['danger_min']):.0%}")

    status = "ready"
    if skill is not None and float(skill) < float(t["not_ready_skill_floor"]):
        status = "not_ready"
    elif reasons:
        status = "candidate"

    return ReadinessResult(
        status=status,
        failed_reasons=tuple(reasons),
        skill_score=float(skill) if skill is not None else None,
        mae=float(mae) if mae is not None else None,
        danger_recall_42=float(danger_recall) if danger_recall is not None else None,
        danger_42_support=int(danger_support) if danger_support is not None else None,
        pi_coverage_90=float(pi_coverage) if pi_coverage is not None else None,
    )
