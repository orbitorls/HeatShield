"""Model artifact registry — save/load XGBoost models with metadata sidecars.

Uses XGBoost native binary JSON (.ubj) format.
This avoids arbitrary-code-execution risks and ensures version portability.
We intentionally do NOT use Python object serialization formats that allow
arbitrary code execution. The .ubj format is XGBoost's own safe binary format.

v1 layout (single booster):
    app/models/forecast_v{n}.ubj
    app/models/forecast_v{n}.json

v2 layout (multi-role ensemble, 3 seeds × 3 roles = 9 boosters):
    app/models/forecast_v{n}/
    ├── mean_s42.ubj
    ├── mean_s123.ubj
    ├── mean_s7.ubj
    ├── q05_s42.ubj
    ├── q05_s123.ubj
    ├── q05_s7.ubj
    ├── q95_s42.ubj
    ├── q95_s123.ubj
    ├── q95_s7.ubj
    └── metadata.json

v2 multi-horizon layout (per-horizon subdirectories):
    app/models/forecast_v{n}/
    ├── h6/
    │   ├── mean_s42.ubj  mean_s123.ubj  mean_s7.ubj
    │   ├── q05_s42.ubj   q05_s123.ubj   q05_s7.ubj
    │   └── q95_s42.ubj   q95_s123.ubj   q95_s7.ubj
    ├── h12/  (same structure)
    ├── h24/  (same structure)
    ├── h48/  (same structure)
    ├── h72/  (same structure)
    └── metadata.json
"""
from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import logging

import xgboost as xgb

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parents[2] / "app" / "models"

_SEEDS = [42, 123, 7]
_ROLES = ["mean", "q05", "q95"]


def _next_version(kind: str) -> int:
    """Find the next available version number for a model kind.

    Detects both v1 (file: {kind}_v{n}.ubj) and v2 (dir: {kind}_v{n}/) formats.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    versions: list[int] = []

    # v1 files
    for p in _MODELS_DIR.glob(f"{kind}_v*.ubj"):
        m = re.search(r"_v(\d+)\.ubj$", p.name)
        if m:
            versions.append(int(m.group(1)))

    # v2 directories
    for p in _MODELS_DIR.iterdir():
        if p.is_dir():
            m = re.match(rf"^{re.escape(kind)}_v(\d+)$", p.name)
            if m:
                versions.append(int(m.group(1)))

    return max(versions) + 1 if versions else 1


def _save_role_boosters(
    dir_path: Path,
    all_boosters: dict[str, list[xgb.Booster]],
) -> None:
    """Save 9 .ubj files (3 roles × 3 seeds) into dir_path."""
    for role in _ROLES:
        boosters = all_boosters[role]
        for booster, seed in zip(boosters, _SEEDS):
            ubj_path = dir_path / f"{role}_s{seed}.ubj"
            booster.save_model(str(ubj_path))


def save_model(
    booster_or_ensemble,
    metadata: dict,
    kind: Literal["forecast"],
) -> Path:
    """Save model and metadata sidecar.

    Accepts three input forms:

    v1 (single booster):
        booster_or_ensemble: xgb.Booster
        Writes:
            app/models/{kind}_v{n}.ubj
            app/models/{kind}_v{n}.json

    v2 single-horizon (ensemble dict with all_boosters):
        booster_or_ensemble: dict with key "all_boosters" →
            {"mean": [b42, b123, b7], "q05": [...], "q95": [...]}
        Writes:
            app/models/{kind}_v{n}/mean_s42.ubj  (and 8 more .ubj files)
            app/models/{kind}_v{n}/metadata.json

    v2 multi-horizon (ensemble dict with horizons key):
        booster_or_ensemble: dict with key "horizons" →
            {6: {"all_boosters": {...}}, 12: {...}, 24: {...}, 48: {...}, 72: {...}}
        Writes:
            app/models/{kind}_v{n}/h{horizon}/mean_s42.ubj  (9 files per horizon)
            app/models/{kind}_v{n}/metadata.json

    Returns the saved path (file for v1, directory for v2).
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = _next_version(kind)

    # --- v2 multi-horizon: dict with "horizons" key ---
    if isinstance(booster_or_ensemble, dict) and "horizons" in booster_or_ensemble:
        horizons_data: dict[int, dict] = booster_or_ensemble["horizons"]
        dir_path = _MODELS_DIR / f"{kind}_v{version}"
        dir_path.mkdir(parents=True, exist_ok=True)

        for horizon_h, horizon_payload in horizons_data.items():
            h_dir = dir_path / f"h{horizon_h}"
            h_dir.mkdir(parents=True, exist_ok=True)
            _save_role_boosters(h_dir, horizon_payload["all_boosters"])

        meta = {
            "kind": kind,
            "version": version,
            "format": "v2_multi_horizon",
            "roles": _ROLES,
            "seeds": _SEEDS,
            "horizons": sorted(horizons_data.keys()),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        meta_path = dir_path / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        return dir_path

    # --- v2 single-horizon: ensemble dict with all_boosters ---
    if isinstance(booster_or_ensemble, dict) and "all_boosters" in booster_or_ensemble:
        all_boosters: dict[str, list[xgb.Booster]] = booster_or_ensemble["all_boosters"]
        dir_path = _MODELS_DIR / f"{kind}_v{version}"
        dir_path.mkdir(parents=True, exist_ok=True)

        _save_role_boosters(dir_path, all_boosters)

        meta = {
            "kind": kind,
            "version": version,
            "format": "v2_ensemble",
            "roles": _ROLES,
            "seeds": _SEEDS,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        meta_path = dir_path / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        return dir_path

    # --- v1: single booster (backward compat) ---
    booster: xgb.Booster = booster_or_ensemble
    model_path = _MODELS_DIR / f"{kind}_v{version}.ubj"
    meta_path = _MODELS_DIR / f"{kind}_v{version}.json"

    booster.save_model(str(model_path))

    meta = {
        "kind": kind,
        "version": version,
        "format": "v1_single",
        "model_file": model_path.name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return model_path


def _load_role_boosters(dir_path: Path) -> dict[str, list[xgb.Booster]]:
    """Load 9 .ubj files (3 roles × 3 seeds) from dir_path."""
    all_boosters: dict[str, list[xgb.Booster]] = {role: [] for role in _ROLES}
    for role in _ROLES:
        for seed in _SEEDS:
            ubj_path = dir_path / f"{role}_s{seed}.ubj"
            b = xgb.Booster()
            b.load_model(str(ubj_path))
            all_boosters[role].append(b)
    return all_boosters


def load_latest(kind: Literal["forecast"], horizon: int | None = None) -> dict:
    """Load the highest-versioned model and its metadata sidecar.

    Args:
        kind: Model kind (e.g. "forecast").
        horizon: Optional horizon in hours. Only used when loading a multi-horizon
            model directory. If specified, only that horizon's boosters are loaded.
            If None and the model is multi-horizon, all horizons are loaded.

    Returns a dict:

    v1 (single booster):
        {
            "booster": xgb.Booster,
            "metadata": {...},
        }

    v2 single-horizon (ensemble):
        {
            "mean": [booster_s42, booster_s123, booster_s7],
            "q05": [...],
            "q95": [...],
            "metadata": {...},
            "booster": booster_s42_mean,  # backward-compat shim
        }

    v2 multi-horizon, horizon=None (all horizons):
        {
            "horizons": {
                6:  {"mean": [...], "q05": [...], "q95": [...]},
                12: {...},
                ...
            },
            "metadata": {...},
            "booster": <mean_s42 from h24 or first available horizon>,
        }

    v2 multi-horizon, horizon=<N> (single horizon):
        {
            "mean": [booster_s42, booster_s123, booster_s7],
            "q05": [...],
            "q95": [...],
            "metadata": {...},
            "booster": booster_s42_mean,  # backward-compat shim
        }

    Raises FileNotFoundError if no model of that kind exists.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all candidate versions from both v1 files and v2 directories
    candidates: list[tuple[int, Path]] = []

    for p in _MODELS_DIR.glob(f"{kind}_v*.ubj"):
        m = re.search(r"_v(\d+)\.ubj$", p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    for p in _MODELS_DIR.iterdir():
        if p.is_dir():
            m = re.match(rf"^{re.escape(kind)}_v(\d+)$", p.name)
            if m:
                candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(
            f"No saved model of kind '{kind}' found in {_MODELS_DIR}. "
            "Run the training script first."
        )

    best_version, best_path = max(candidates, key=lambda x: x[0])

    # --- v2: directory ---
    if best_path.is_dir():
        meta_path = best_path / "metadata.json"
        metadata: dict = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        # Detect multi-horizon layout: directory contains h{N}/ subdirectories
        horizon_dirs = sorted(
            [d for d in best_path.iterdir() if d.is_dir() and re.match(r"^h\d+$", d.name)],
            key=lambda d: int(d.name[1:]),
        )

        if horizon_dirs:
            # --- v2 multi-horizon ---
            available_horizons = [int(d.name[1:]) for d in horizon_dirs]

            if horizon is not None:
                # Load only the requested horizon
                h_dir = best_path / f"h{horizon}"
                if not h_dir.exists():
                    raise FileNotFoundError(
                        f"Horizon h{horizon} not found in {best_path}. "
                        f"Available: {available_horizons}"
                    )
                h_boosters = _load_role_boosters(h_dir)
                result: dict = {**h_boosters, "metadata": metadata}
                result["booster"] = h_boosters["mean"][0]
                return result

            # Load all horizons
            all_horizons: dict[int, dict[str, list[xgb.Booster]]] = {}
            for h_dir in horizon_dirs:
                h = int(h_dir.name[1:])
                all_horizons[h] = _load_role_boosters(h_dir)

            # Backward-compat booster shim: prefer h24, fall back to first available
            shim_horizon = 24 if 24 in all_horizons else min(all_horizons.keys())
            shim_booster = all_horizons[shim_horizon]["mean"][0]

            return {
                "horizons": all_horizons,
                "metadata": metadata,
                "booster": shim_booster,
            }

        # --- v2 single-horizon (old Phase C layout): boosters at dir root ---
        all_boosters = _load_role_boosters(best_path)
        result = {**all_boosters, "metadata": metadata}
        # Backward-compat shim
        result["booster"] = all_boosters["mean"][0]
        return result

    # --- v1: single .ubj file ---
    meta_path = best_path.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    booster = xgb.Booster()
    booster.load_model(str(best_path))

    return {"booster": booster, "metadata": metadata}


# ---------------------------------------------------------------
# v3 — model-agnostic registry (add after existing functions)
# ---------------------------------------------------------------

def _v3_root() -> Path:
    """Return active v3-compatible registry root.

    Defaults to forecast_v3. Training can set HEATSHIELD_FORECAST_VERSION=v4
    to write a parallel artifact tree without changing API runtime defaults.
    """
    version = os.environ.get("HEATSHIELD_FORECAST_VERSION", "v3").strip() or "v3"
    version = version.removeprefix("forecast_")
    return Path(__file__).parents[2] / "app" / "models" / f"forecast_{version}"


def save_model_v3(
    forecaster,          # BaseForecaster instance
    metadata: dict,
    station_id: str,
    horizon_h: int,
) -> Path:
    """Save a v3 backend forecaster to the per-(station,horizon) directory.

    Layout:
      app/models/forecast_v3/
        choice_matrix.json
        {station_id}/
          h{H}/
            bundle.json    ← metadata + backend_name
            <backend-specific files written by forecaster.save()>
    """
    root = _v3_root()
    slot_dir = root / station_id / f"h{horizon_h}"
    slot_dir.mkdir(parents=True, exist_ok=True)

    # Write backend artifacts
    forecaster.save(slot_dir)

    # Backend artifacts own bundle.json. Keep registry metadata in a separate
    # sidecar so save_model_v3() does not overwrite backend-specific fields
    # such as feature medians, calibrators, gates, or best params.
    registry_meta = {
        "backend_name": forecaster.backend_name,
        "feature_list": forecaster.feature_list,
        "target_kind": forecaster.target_kind,
        "station_id": station_id,
        "horizon_h": horizon_h,
        **metadata,
    }
    (slot_dir / "registry.json").write_text(json.dumps(registry_meta, indent=2, default=str))

    # Update choice_matrix.json
    matrix_path = root / "choice_matrix.json"
    matrix = {}
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text())
    matrix.setdefault(station_id, {})[str(horizon_h)] = forecaster.backend_name
    matrix_path.write_text(json.dumps(matrix, indent=2))

    logger.info("Saved v3 %s model: station=%s h=%d -> %s",
                forecaster.backend_name, station_id, horizon_h, slot_dir)
    return slot_dir


def load_latest_v3(station_id: str, horizon_h: int):
    """Load the v3 forecaster for (station_id, horizon_h).

    Reads choice_matrix.json to find backend_name, then imports and loads
    the correct backend class from app.ml.forecast.backends.

    Raises FileNotFoundError if no v3 model exists for this (station, horizon).
    """
    slot_dir = _v3_root() / station_id / f"h{horizon_h}"
    bundle_path = slot_dir / "bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"No v3 model found at {slot_dir}")

    bundle = json.loads(bundle_path.read_text())
    backend_name = bundle["backend_name"]

    if backend_name == "lightgbm_quantile":
        from app.ml.forecast.backends.lgbm_backend import LGBMForecaster
        return LGBMForecaster.load(slot_dir)
    elif backend_name == "lightgbm_hi_quantile":
        from app.ml.forecast.backends.lgbm_backend import LGBMDirectHIForecaster
        return LGBMDirectHIForecaster.load(slot_dir)
    elif backend_name == "xgboost":
        from app.ml.forecast.backends.xgb_backend import XGBForecaster
        return XGBForecaster.load(slot_dir)
    elif backend_name == "catboost_quantile":
        from app.ml.forecast.backends.catboost_backend import CatBoostForecaster
        return CatBoostForecaster.load(slot_dir)
    elif backend_name == "tabpfn":
        raise NotImplementedError(
            "TabPFN backend is not yet implemented. "
            "Re-train with --backend lightgbm to use an available backend."
        )
    else:
        raise ValueError(f"Unknown backend_name={backend_name!r} in {bundle_path}")


def list_v3_models() -> dict:
    """Return the choice_matrix.json content or empty dict if no v3 models exist."""
    matrix_path = _v3_root() / "choice_matrix.json"
    if not matrix_path.exists():
        return {}
    return json.loads(matrix_path.read_text())


class LazyModelLoader:
    """Lazy-loading wrapper for v2 multi-horizon ensembles.

    Loads individual horizon boosters on first access instead of pulling
    everything into memory at startup. Keeps a bounded LRU of loaded
    horizons to prevent unbounded memory growth.
    """

    def __init__(self, model_dir: Path, metadata: dict, max_cached_horizons: int = 3) -> None:
        self._model_dir = model_dir
        self.metadata = metadata
        self._max_cached = max_cached_horizons
        self._cache: dict[int, dict[str, list[xgb.Booster]]] = {}
        self._cache_order: list[int] = []
        self._available_horizons: list[int] | None = None

    @property
    def available_horizons(self) -> list[int]:
        if self._available_horizons is None:
            self._available_horizons = sorted(
                int(d.name[1:])
                for d in self._model_dir.iterdir()
                if d.is_dir() and re.match(r"^h\d+$", d.name)
            )
        return self._available_horizons

    def _load_horizon(self, horizon: int) -> dict[str, list[xgb.Booster]]:
        """Load (or return cached) boosters for a single horizon."""
        if horizon in self._cache:
            # Move to MRU
            self._cache_order.remove(horizon)
            self._cache_order.append(horizon)
            return self._cache[horizon]

        h_dir = self._model_dir / f"h{horizon}"
        if not h_dir.exists():
            raise FileNotFoundError(
                f"Horizon h{horizon} not found in {self._model_dir}. "
                f"Available: {self.available_horizons}"
            )

        boosters = _load_role_boosters(h_dir)
        self._cache[horizon] = boosters
        self._cache_order.append(horizon)

        while len(self._cache_order) > self._max_cached:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)

        return boosters

    def get_horizon(self, horizon: int) -> dict[str, list[xgb.Booster]]:
        """Return boosters for a specific horizon (loaded on demand)."""
        return self._load_horizon(horizon)

    def get_all_horizons(self) -> dict[int, dict[str, list[xgb.Booster]]]:
        """Load all horizons (use sparingly — defeats lazy loading)."""
        return {h: self._load_horizon(h) for h in self.available_horizons}

    @property
    def booster(self) -> xgb.Booster:
        """Backward-compat shim: prefer h24, fall back to first available."""
        h = 24 if 24 in self.available_horizons else self.available_horizons[0]
        return self._load_horizon(h)["mean"][0]
