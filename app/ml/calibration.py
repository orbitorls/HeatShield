"""
Bias Calibration Module — แก้อคติ (bias) ของโมเดลพยากรณ์ Heat Index

Fits station/horizon/hour-bucket bias corrections on validation residuals,
then applies them deterministically at inference time. ML-free per CLAUDE.md
(`app/core/`): only numpy + sklearn.IsotonicRegression are used (isotonic
fitting is deterministic monotone regression, not statistical learning).

4 levels of granularity:
    L1 — mean shift per station_id
    L2 — mean shift per (station_id, horizon_h)
    L3 — mean shift per (station_id, horizon_h, hour_bucket)
         hour_buckets: 0-5 → 0, 6-11 → 1, 12-17 → 2, 18-23 → 3
    L4 — isotonic regression on raw → actual, fit per (station_id, horizon_h)

A `select_calibration` helper picks the level with lowest val MAE.

ระดับการ calibrate ทั้ง 4 จะถูกเลือกอัตโนมัติจาก validation MAE,
และผลที่ได้สามารถ serialize ลง JSON เพื่อบันทึกร่วมกับ model registry ได้
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

CalibrationLevel = Literal["L1", "L2", "L3", "L4"]
_VALID_LEVELS: tuple[CalibrationLevel, ...] = ("L1", "L2", "L3", "L4")
_MIN_GROUP_FOR_ISOTONIC = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hour_bucket(hour: int) -> int:
    """แบ่งชั่วโมง 0–23 เป็น 4 ช่วง (bucket 0–3)."""
    h = int(hour)
    if 0 <= h <= 5:
        return 0
    if 6 <= h <= 11:
        return 1
    if 12 <= h <= 17:
        return 2
    return 3


def _as_arrays(
    y_true,
    y_pred,
    station_ids,
    horizons,
    hours,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    st = np.asarray(station_ids).reshape(-1).astype(str)
    ho = np.asarray(horizons).reshape(-1).astype(int)
    hr = np.asarray(hours).reshape(-1).astype(int)
    n = yt.shape[0]
    if not (yp.shape[0] == st.shape[0] == ho.shape[0] == hr.shape[0] == n):
        raise ValueError(
            "All input arrays must have the same length: got "
            f"y_true={yt.shape}, y_pred={yp.shape}, station_ids={st.shape}, "
            f"horizons={ho.shape}, hours={hr.shape}"
        )
    if n == 0:
        raise ValueError("Empty input arrays — cannot fit calibration on 0 rows")
    return yt, yp, st, ho, hr


def _apply_arrays(
    y_pred,
    station_ids,
    horizons,
    hours,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    st = np.asarray(station_ids).reshape(-1).astype(str)
    ho = np.asarray(horizons).reshape(-1).astype(int)
    hr = np.asarray(hours).reshape(-1).astype(int)
    n = yp.shape[0]
    if not (st.shape[0] == ho.shape[0] == hr.shape[0] == n):
        raise ValueError(
            "apply() arrays must have matching length; got "
            f"y_pred={yp.shape}, station_ids={st.shape}, "
            f"horizons={ho.shape}, hours={hr.shape}"
        )
    return yp, st, ho, hr, n


# ---------------------------------------------------------------------------
# Calibration data class
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """ผลลัพธ์ของการ calibrate — เก็บ shift / isotonic params พร้อม metadata.

    Use `apply` to correct new predictions; use `to_json`/`from_json` to
    persist alongside the model registry.
    """

    level: CalibrationLevel
    # L1: { station_id: shift }
    # L2: { f"{station}|{horizon}": shift }
    # L3: { f"{station}|{horizon}|{bucket}": shift }
    shifts: dict[str, float] = field(default_factory=dict)
    # L4: { f"{station}|{horizon}": IsotonicRegression OR None (identity) }
    isotonics: dict[str, IsotonicRegression | None] = field(default_factory=dict)

    # ------------------------------------------------------------------ apply
    def apply(
        self,
        y_pred,
        station_ids,
        horizons,
        hours,
    ) -> np.ndarray:
        """แก้ค่าทำนายตามระดับ calibration ที่ fit ไว้ — คืนค่า np.ndarray.

        Unseen groups pass through unchanged so inference never crashes
        on a station/horizon combo that wasn't in the validation set.
        """
        yp, st, ho, hr, n = _apply_arrays(y_pred, station_ids, horizons, hours)
        out = yp.copy()

        if self.level == "L1":
            for i in range(n):
                key = st[i]
                if key in self.shifts:
                    out[i] = yp[i] + self.shifts[key]
            return out

        if self.level == "L2":
            for i in range(n):
                key = f"{st[i]}|{int(ho[i])}"
                if key in self.shifts:
                    out[i] = yp[i] + self.shifts[key]
            return out

        if self.level == "L3":
            for i in range(n):
                b = _hour_bucket(int(hr[i]))
                key = f"{st[i]}|{int(ho[i])}|{b}"
                if key in self.shifts:
                    out[i] = yp[i] + self.shifts[key]
            return out

        if self.level == "L4":
            # group rows by (station, horizon) and call isotonic per group
            keys = np.array([f"{st[i]}|{int(ho[i])}" for i in range(n)])
            for key in np.unique(keys):
                mask = keys == key
                iso = self.isotonics.get(key)
                if iso is None:
                    # identity / unseen group → pass through
                    continue
                out[mask] = iso.predict(yp[mask])
            return out

        raise ValueError(f"Unknown calibration level: {self.level!r}")

    # ------------------------------------------------------------- serialize
    def to_json(self) -> dict[str, Any]:
        """แปลง Calibration เป็น dict ที่ JSON-serializable."""
        payload: dict[str, Any] = {
            "level": self.level,
            "shifts": {k: float(v) for k, v in self.shifts.items()},
            "isotonics": {},
        }
        for key, iso in self.isotonics.items():
            if iso is None:
                payload["isotonics"][key] = None
            else:
                payload["isotonics"][key] = {
                    "X_thresholds": [float(x) for x in np.asarray(iso.X_thresholds_)],
                    "y_thresholds": [float(y) for y in np.asarray(iso.y_thresholds_)],
                    "X_min": float(iso.X_min_),
                    "X_max": float(iso.X_max_),
                    "increasing": bool(iso.increasing_),
                    "out_of_bounds": getattr(iso, "out_of_bounds", "clip"),
                }
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Calibration":
        """โหลดกลับมาเป็น Calibration object — round-trip safe."""
        level = payload["level"]
        if level not in _VALID_LEVELS:
            raise ValueError(f"Invalid level in payload: {level!r}")
        shifts = {k: float(v) for k, v in payload.get("shifts", {}).items()}
        isotonics: dict[str, IsotonicRegression | None] = {}
        for key, iso_payload in payload.get("isotonics", {}).items():
            if iso_payload is None:
                isotonics[key] = None
                continue
            iso = IsotonicRegression(
                increasing=iso_payload.get("increasing", True),
                out_of_bounds=iso_payload.get("out_of_bounds", "clip"),
            )
            iso.X_thresholds_ = np.asarray(iso_payload["X_thresholds"], dtype=float)
            iso.y_thresholds_ = np.asarray(iso_payload["y_thresholds"], dtype=float)
            iso.X_min_ = float(iso_payload["X_min"])
            iso.X_max_ = float(iso_payload["X_max"])
            iso.increasing_ = bool(iso_payload.get("increasing", True))
            # Build interpolator the same way sklearn does internally
            from scipy.interpolate import interp1d  # type: ignore

            iso.f_ = interp1d(
                iso.X_thresholds_,
                iso.y_thresholds_,
                kind="linear",
                bounds_error=False,
                fill_value=(iso.y_thresholds_[0], iso.y_thresholds_[-1]),
            )
            isotonics[key] = iso
        return cls(level=level, shifts=shifts, isotonics=isotonics)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_calibration(
    level: CalibrationLevel,
    y_true,
    y_pred,
    station_ids,
    horizons,
    hours,
) -> Calibration:
    """Fit a bias-correction at one of 4 granularities — ฟิตค่าชดเชย bias.

    Args:
        level: "L1" | "L2" | "L3" | "L4" — ระดับความละเอียดของ calibration
        y_true, y_pred: arrays อันเดียวกันความยาว n (target และ prediction)
        station_ids: array ของ station id (string)
        horizons: array ของ horizon (ชั่วโมง, int)
        hours: array ของชั่วโมงในวัน (0–23)

    Returns:
        Calibration object พร้อม `.apply(...)`, `.to_json()`, `.from_json(...)`.

    Raises:
        ValueError: ถ้า level ไม่ถูกต้อง หรือ input ว่าง / ขนาดไม่ตรงกัน.
    """
    if level not in _VALID_LEVELS:
        raise ValueError(
            f"level must be one of {_VALID_LEVELS}, got {level!r}"
        )
    yt, yp, st, ho, hr = _as_arrays(y_true, y_pred, station_ids, horizons, hours)
    residual = yt - yp  # shift to add: yp + shift ≈ yt

    if level == "L1":
        shifts: dict[str, float] = {}
        for s in np.unique(st):
            mask = st == s
            shifts[str(s)] = float(residual[mask].mean())
        return Calibration(level="L1", shifts=shifts)

    if level == "L2":
        shifts = {}
        keys = np.array([f"{st[i]}|{int(ho[i])}" for i in range(st.shape[0])])
        for key in np.unique(keys):
            mask = keys == key
            shifts[str(key)] = float(residual[mask].mean())
        return Calibration(level="L2", shifts=shifts)

    if level == "L3":
        shifts = {}
        buckets = np.array([_hour_bucket(int(h)) for h in hr])
        keys = np.array([
            f"{st[i]}|{int(ho[i])}|{int(buckets[i])}"
            for i in range(st.shape[0])
        ])
        for key in np.unique(keys):
            mask = keys == key
            shifts[str(key)] = float(residual[mask].mean())
        return Calibration(level="L3", shifts=shifts)

    # L4 — isotonic regression per (station, horizon)
    isotonics: dict[str, IsotonicRegression | None] = {}
    keys = np.array([f"{st[i]}|{int(ho[i])}" for i in range(st.shape[0])])
    for key in np.unique(keys):
        mask = keys == key
        x_g = yp[mask]
        y_g = yt[mask]
        if x_g.size < _MIN_GROUP_FOR_ISOTONIC:
            isotonics[str(key)] = None  # identity fallback
            continue
        # Degenerate input (all-equal raw predictions) → identity
        if np.all(x_g == x_g[0]):
            isotonics[str(key)] = None
            continue
        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )
        iso.fit(x_g, y_g)
        isotonics[str(key)] = iso
    return Calibration(level="L4", isotonics=isotonics)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

def select_calibration(
    *,
    y_true_train,
    y_pred_train,
    station_ids_train,
    horizons_train,
    hours_train,
    y_true_val,
    y_pred_val,
    station_ids_val,
    horizons_val,
    hours_val,
    levels: Sequence[CalibrationLevel] = _VALID_LEVELS,
) -> Calibration:
    """เลือก level ที่ให้ val MAE ต่ำสุด แล้วคืน Calibration ที่ฟิตด้วย level นั้น.

    Args:
        y_true_train / y_pred_train / station_ids_train / horizons_train / hours_train:
            ใช้สำหรับ FIT calibration ในแต่ละ level
        y_true_val / y_pred_val / station_ids_val / horizons_val / hours_val:
            ใช้สำหรับเลือก level ที่ MAE ต่ำที่สุด

    Returns:
        Calibration object ของ level ที่ดีที่สุด (เลือกเล็กก่อนเมื่อเสมอกัน
        ตามลำดับ argument `levels`).
    """
    if not levels:
        raise ValueError("levels must contain at least one entry")

    yt_v = np.asarray(y_true_val, dtype=float).reshape(-1)
    yp_v = np.asarray(y_pred_val, dtype=float).reshape(-1)

    best: Calibration | None = None
    best_mae = float("inf")
    for lvl in levels:
        if lvl not in _VALID_LEVELS:
            raise ValueError(
                f"Invalid level {lvl!r} in levels={levels} "
                f"(allowed: {_VALID_LEVELS})"
            )
        calib = fit_calibration(
            level=lvl,
            y_true=y_true_train,
            y_pred=y_pred_train,
            station_ids=station_ids_train,
            horizons=horizons_train,
            hours=hours_train,
        )
        corrected = calib.apply(yp_v, station_ids_val, horizons_val, hours_val)
        mae = float(np.mean(np.abs(corrected - yt_v)))
        if mae < best_mae:
            best_mae = mae
            best = calib
    assert best is not None  # at least one level was tried
    return best
