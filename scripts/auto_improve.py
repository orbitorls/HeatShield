"""Autonomous champion-challenger improvement loop for HeatShield AI forecast models.

Runs multiple rounds of training across all stations and backends.
Each challenger is only saved if it beats the current champion MAE.

Usage:
    python scripts/auto_improve.py                        # all stations, default backends
    python scripts/auto_improve.py --stations BKK_01      # single station
    python scripts/auto_improve.py --backends lightgbm xgboost
    python scripts/auto_improve.py --rounds 3 --trials 50
    python scripts/auto_improve.py --start 2015-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATION_STARTS = {
    "BKK_01": "2015-01-01",
    "CNX_01": "2015-01-01",
    "KKN_01": "2018-01-01",
    "HYI_01": "2018-01-01",
    "RYG_01": "2018-01-01",
}
ALL_STATIONS = list(STATION_STARTS.keys())
ALL_BACKENDS = ["lightgbm", "xgboost", "lightgbm_hi"]
V3_ROOT = Path(__file__).resolve().parents[1] / "app" / "models" / "forecast_v3"


def _read_leaderboard() -> dict[str, dict]:
    """Return {slot: {mae, rmse, skill, backend}} from current registry.json files."""
    board = {}
    for p in sorted(V3_ROOT.rglob("registry.json")):
        if p.parent.parent.name == "forecast_v3":
            continue
        d = json.loads(p.read_text())
        ev = d.get("evaluation", {})
        slot = f"{p.parent.parent.name}/{p.parent.name}"
        board[slot] = {
            "mae": ev.get("regression", {}).get("mae"),
            "rmse": ev.get("regression", {}).get("rmse"),
            "skill": ev.get("baselines", {}).get("skill_score"),
            "backend": d.get("backend_name", "?"),
            "run_id": d.get("run_id", "?"),
        }
    return board


def _print_leaderboard(board: dict, title: str = "Leaderboard") -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    print(f"{'Slot':<20} {'Backend':<22} {'MAE':>7} {'Skill':>7}")
    print("-" * 60)
    for slot in sorted(board):
        b = board[slot]
        mae = f"{b['mae']:.3f}" if b["mae"] is not None else "?"
        skill = f"{b['skill']:.3f}" if b["skill"] is not None else "?"
        print(f"{slot:<20} {b['backend']:<22} {mae:>7} {skill:>7}")


def _run_training(
    station: str,
    backend: str,
    trials: int,
    start: str,
    force: bool,
    extra_args: list[str] | None = None,
) -> bool:
    """Run train_forecast.py for one station+backend. Returns True if successful."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_forecast.py"),
        "--station", station,
        "--backend", backend,
        "--trials", str(trials),
        "--start", start,
    ]
    if force:
        cmd.append("--force")
    if extra_args:
        cmd.extend(extra_args)

    log_dir = Path("logs") / "autorun"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{ts}_{station}_{backend}.log"

    logger.info("Training %s backend=%s trials=%d start=%s force=%s → %s",
                station, backend, trials, start, force, log_path.name)

    with open(log_path, "w", encoding="utf-8") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        logger.error("FAILED %s/%s — check %s", station, backend, log_path)
        return False
    logger.info("OK %s/%s", station, backend)
    return True


def _delta_summary(before: dict, after: dict) -> None:
    improved, regressed, unchanged = [], [], []
    for slot in sorted(before):
        b = before.get(slot, {})
        a = after.get(slot, {})
        if b.get("mae") is None or a.get("mae") is None:
            continue
        delta = b["mae"] - a["mae"]
        if delta > 0.005:
            improved.append((slot, b["mae"], a["mae"], delta, a["backend"]))
        elif delta < -0.005:
            regressed.append((slot, b["mae"], a["mae"], delta, a["backend"]))
        else:
            unchanged.append(slot)

    if improved:
        print(f"\n✓ IMPROVED ({len(improved)} slots):")
        for slot, old, new, d, be in improved:
            print(f"  {slot:<20} {old:.3f} → {new:.3f} ({d:+.3f}) [{be}]")
    if regressed:
        print(f"\n✗ REGRESSED ({len(regressed)} slots) — champion retained:")
        for slot, old, new, d, be in regressed:
            print(f"  {slot:<20} {old:.3f} → {new:.3f} ({d:+.3f}) [{be}]")
    if unchanged:
        print(f"\n= No change: {', '.join(unchanged)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous improvement loop for HeatShield AI forecasters")
    parser.add_argument("--stations", nargs="+", default=ALL_STATIONS, choices=ALL_STATIONS,
                        help="Stations to improve (default: all)")
    parser.add_argument("--backends", nargs="+", default=ALL_BACKENDS,
                        choices=ALL_BACKENDS, help="Backends to try (default: all)")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of improvement rounds (default: 2)")
    parser.add_argument("--trials", type=int, default=60,
                        help="Optuna trials per slot per round (default: 60)")
    parser.add_argument("--force-round1", action="store_true",
                        help="Force overwrite on round 1 (re-baselines from scratch)")
    parser.add_argument("--start", type=str, default=None,
                        help="Override data start date for all stations (default: per-station)")
    args = parser.parse_args()

    logger.info("AutoImprove: stations=%s backends=%s rounds=%d trials=%d",
                args.stations, args.backends, args.rounds, args.trials)

    before = _read_leaderboard()
    _print_leaderboard(before, "Baseline Leaderboard")

    for round_idx in range(1, args.rounds + 1):
        print(f"\n{'='*70}")
        print(f" ROUND {round_idx}/{args.rounds}")
        print(f"{'='*70}")

        for backend_idx, backend in enumerate(args.backends):
            print(f"\n--- Backend: {backend} ---")
            # Round 1, backend 0 → force if --force-round1; otherwise always challenger
            force = args.force_round1 and round_idx == 1 and backend_idx == 0

            for station in args.stations:
                start = args.start or STATION_STARTS[station]
                ok = _run_training(station, backend, args.trials, start, force=force)
                if not ok:
                    logger.warning("Skipping %s/%s due to training error", station, backend)

    after = _read_leaderboard()
    _print_leaderboard(after, "Final Leaderboard")
    _delta_summary(before, after)
    print("\nDone.")


if __name__ == "__main__":
    main()
