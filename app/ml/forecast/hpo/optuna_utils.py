"""Small shared helpers for forecast training runtime state."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def forecast_version() -> str:
    """Return the active forecast artifact version suffix, defaulting to v3."""
    value = os.environ.get("HEATSHIELD_FORECAST_VERSION", "v3").strip() or "v3"
    return value[1:] if value.startswith("forecast_") else value


def forecast_model_root() -> Path:
    return Path(__file__).resolve().parents[4] / "app" / "models" / f"forecast_{forecast_version()}"


def feature_signature(columns: list[str] | tuple[str, ...], *, extra: str = "") -> str:
    """Generate a hash signature for feature columns and extra metadata."""
    payload = "|".join(sorted(map(str, columns))) + f"|{extra}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def get_study_version() -> str:
    """Get the current study version based on model version and feature config."""
    from app.ml.forecast.features import _load_feature_config
    
    # Get model version
    model_ver = forecast_version()
    
    # Get feature config version
    try:
        config = _load_feature_config()
        feature_ver = config.get("version", "unknown")
    except Exception:
        feature_ver = "unknown"
    
    return f"{model_ver}_{feature_ver}"


def heartbeat_path(run_id: str | None) -> Path | None:
    if not run_id:
        return None
    return Path("logs") / "train" / f"{run_id}.heartbeat"


def write_heartbeat(
    run_id: str | None,
    *,
    current_slot: str = "",
    phase: str = "",
    trials_done: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write heartbeat JSON locally and to Drive when configured."""
    path = heartbeat_path(run_id)
    if path is None:
        return
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "current_slot": current_slot,
        "phase": phase,
    }
    if trials_done is not None:
        payload["trials_done"] = int(trials_done)
    if extra:
        payload.update(extra)

    paths = [path]
    drive_root = os.environ.get("HEATSHIELD_DRIVE_ROOT")
    if drive_root:
        paths.append(Path(drive_root) / "logs" / f"{run_id}.heartbeat")

    for target in paths:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
        except Exception:
            # Heartbeat is best-effort and must not fail training.
            pass
