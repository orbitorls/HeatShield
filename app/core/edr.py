"""Event Data Recorder (EDR) — structured audit trail for HeatShield AI.

Records training, prediction, and system events as append-only JSON lines
for post-hoc debugging, compliance, and model lineage tracking.

Usage:
    from app.core.edr import edr

    with edr.training_run(station="BKK_01", horizon=6, backend="lightgbm") as run:
        run.log_hyperparams({"num_leaves": 128})
        run.log_metric("mae", 1.12)
        run.log_champion_result(won=True, mae_delta=-0.08)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_EDR_DIR = Path(__file__).parents[2] / "logs" / "edr"
_EDR_DIR.mkdir(parents=True, exist_ok=True)


class EventType(str, Enum):
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAIL = "training_fail"
    PREDICTION = "prediction"
    CHAMPION_CHALLENGER = "champion_challenger"
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"


class EDREvent(BaseModel):
    """Single EDR record — immutable after creation."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: EventType
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str | None = None
    station_id: str | None = None
    horizon_h: int | None = None
    backend: str | None = None
    version: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class _RunContext:
    """Mutable accumulator for a single training/prediction run."""

    def __init__(
        self,
        event_type: EventType,
        run_id: str,
        station_id: str | None,
        horizon_h: int | None,
        backend: str | None,
        version: str | None,
    ):
        self.event_type = event_type
        self.run_id = run_id
        self.station_id = station_id
        self.horizon_h = horizon_h
        self.backend = backend
        self.version = version
        self._records: list[EDREvent] = []
        self._start_ns = time.perf_counter_ns()

    def _emit(self, subtype: str, payload: dict[str, Any], tags: list[str] | None = None) -> None:
        event = EDREvent(
            event_type=self.event_type,
            run_id=self.run_id,
            station_id=self.station_id,
            horizon_h=self.horizon_h,
            backend=self.backend,
            version=self.version,
            payload={"subtype": subtype, **payload},
            tags=tags or [],
        )
        self._records.append(event)
        _append_to_file(event)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self._emit("hyperparams", {"params": params})

    def log_metric(self, name: str, value: float, extra: dict[str, Any] | None = None) -> None:
        payload = {"name": name, "value": float(value)}
        if extra:
            payload["extra"] = extra
        self._emit("metric", payload)

    def log_champion_result(self, won: bool, mae_delta: float, reason: str = "") -> None:
        self._emit(
            "champion_result",
            {"won": won, "mae_delta": float(mae_delta), "reason": reason},
            tags=["champion" if won else "challenger"],
        )

    def log_model_artifact(self, path: str, file_size: int | None = None) -> None:
        self._emit("artifact", {"path": path, "file_size": file_size})

    def log_data_quality(self, rows: int, gaps: int, outliers: int) -> None:
        self._emit("data_quality", {"rows": rows, "gaps": gaps, "outliers": outliers})

    def log_exception(self, exc: BaseException) -> None:
        self._emit(
            "exception",
            {
                "exc_type": type(exc).__name__,
                "exc_msg": str(exc),
            },
            tags=["error"],
        )

    def elapsed_ms(self) -> int:
        return int((time.perf_counter_ns() - self._start_ns) / 1_000_000)

    def finalize(self, success: bool = True) -> None:
        self._emit(
            "summary",
            {
                "success": success,
                "elapsed_ms": self.elapsed_ms(),
                "record_count": len(self._records),
            },
        )


class EventDataRecorder:
    """Global singleton for structured event recording."""

    def __init__(self) -> None:
        self._run_counter = 0

    @contextmanager
    def training_run(
        self,
        station: str,
        horizon: int,
        backend: str,
        version: str = "v3",
    ) -> Generator[_RunContext, None, None]:
        self._run_counter += 1
        run_id = f"tr-{self._run_counter:04d}-{uuid.uuid4().hex[:6]}"
        ctx = _RunContext(
            event_type=EventType.TRAINING_START,
            run_id=run_id,
            station_id=station,
            horizon_h=horizon,
            backend=backend,
            version=version,
        )
        ctx._emit("start", {"message": f"training started for {station} h{horizon}"})
        try:
            yield ctx
        except Exception as exc:
            ctx.log_exception(exc)
            ctx.finalize(success=False)
            raise
        else:
            ctx.finalize(success=True)

    @contextmanager
    def prediction_request(
        self,
        station: str,
        version: str = "v3",
    ) -> Generator[_RunContext, None, None]:
        self._run_counter += 1
        run_id = f"pred-{self._run_counter:04d}-{uuid.uuid4().hex[:6]}"
        ctx = _RunContext(
            event_type=EventType.PREDICTION,
            run_id=run_id,
            station_id=station,
            horizon_h=None,
            backend=None,
            version=version,
        )
        ctx._emit("request", {"station": station})
        try:
            yield ctx
        except Exception as exc:
            ctx.log_exception(exc)
            ctx.finalize(success=False)
            raise
        else:
            ctx.finalize(success=True)

    def log_system(self, message: str, level: str = "info", extra: dict[str, Any] | None = None) -> None:
        event = EDREvent(
            event_type=EventType.SYSTEM,
            payload={"message": message, "level": level, **(extra or {})},
        )
        _append_to_file(event)

    def log_data_quality_issue(
        self,
        station: str,
        issue: str,
        severity: str = "warning",
        details: dict[str, Any] | None = None,
    ) -> None:
        event = EDREvent(
            event_type=EventType.DATA_QUALITY,
            station_id=station,
            payload={"issue": issue, "severity": severity, **(details or {})},
            tags=[severity, "data_quality"],
        )
        _append_to_file(event)

    def query(self, *, station: str | None = None, event_type: EventType | None = None, limit: int = 100) -> list[EDREvent]:
        """Simple in-memory query over today's EDR file (for ad-hoc debugging)."""
        path = _daily_path()
        if not path.exists():
            return []
        results: list[EDREvent] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(results) >= limit:
                    break
                try:
                    data = json.loads(line)
                    if station and data.get("station_id") != station:
                        continue
                    if event_type and data.get("event_type") != event_type.value:
                        continue
                    results.append(EDREvent.model_validate(data))
                except Exception:
                    continue
        return results


# Global singleton — import once and reuse
edr = EventDataRecorder()


def _daily_path() -> Path:
    return _EDR_DIR / f"edr_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"


def _append_to_file(event: EDREvent) -> None:
    path = _daily_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(event.to_json() + "\n")
