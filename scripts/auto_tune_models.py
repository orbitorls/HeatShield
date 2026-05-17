"""
auto_tune_models.py — Standalone auto-tune script
===================================================
รันเทรน/benchmark/retrain วนซ้ำจนกว่าโมเดลจะผ่าน threshold ทุกตัว

Usage:
    python scripts/auto_tune_models.py
    python scripts/auto_tune_models.py --max-rounds 10
    python scripts/auto_tune_models.py --stations NSW_01 SPB_01
    python scripts/auto_tune_models.py --reset

Thresholds (ปรับได้ที่ THRESHOLDS dict ด้านล่าง):
    MAE <= 2.0  (h6/h12), <= 2.4 (h24)
    skill >= -0.04
    pi_coverage in [0.80, 0.98]
    danger_recall >= 0.4 (ถ้ามี danger cases)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_tune")

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT / "scripts" / "train_forecast.py"
STATUS_SCRIPT = ROOT / "scripts" / "check_model_status.py"
STATE_FILE = ROOT / ".auto_tune_state.json"
STATUS_JSON = ROOT / "logs" / "model_status.json"

# ---- Thresholds -------------------------------------------------------
THRESHOLDS = {
    "mae": {6: 2.0, 12: 2.2, 24: 2.4, 48: 2.8, 72: 3.0},
    "skill": -0.04,          # min skill score
    "pi_cov_min": 0.80,
    "pi_cov_max": 0.98,
    "danger_recall_min": 0.40,  # only checked when danger cases exist
}

# ---- Trial schedule per round (more trials each retry) ----------------
TRIAL_SCHEDULE = [10, 15, 20, 30, 40]   # round 0,1,2,3,4+

# ---- Data window per round (longer data = more accurate but slower) ---
DATA_YEARS_SCHEDULE = [2, 3, 4, 5, 5]   # round 0,1,2,3,4+


# =======================================================================
# Helpers
# =======================================================================

def run_status() -> dict:
    """Run check_model_status.py and return parsed JSON."""
    subprocess.run(
        [sys.executable, str(STATUS_SCRIPT)],
        cwd=ROOT, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with open(STATUS_JSON) as f:
        return json.load(f)


def get_failed(status: dict, target_stations: list[str] | None = None) -> list[tuple[str, int]]:
    """Return list of (station, horizon_h) that fail our THRESHOLDS.

    status JSON keys (from check_model_status.py):
      mae, skill_score, pi_coverage_90, danger_recall_42, danger_42_support
    """
    failed = []
    # support both "all" (full list) and "failed" (pre-filtered list)
    model_list = status.get("all", status.get("failed", []))
    for m in model_list:
        sid = m.get("station", "")
        if target_stations and sid not in target_stations:
            continue
        hz = m.get("horizon", "h6")
        h = int(str(hz).replace("h", ""))

        mae = m.get("mae")
        skill = m.get("skill_score")
        pi_cov = m.get("pi_coverage_90")
        danger_recall = m.get("danger_recall_42")   # None means no danger cases
        danger_support = m.get("danger_42_support", 0)

        reasons = []
        if mae is not None and mae > THRESHOLDS["mae"].get(h, 2.5):
            reasons.append(f"mae={mae:.3f}>{THRESHOLDS['mae'].get(h,2.5)}")
        if skill is not None and skill < THRESHOLDS["skill"]:
            reasons.append(f"skill={skill:.3f}<{THRESHOLDS['skill']}")
        if pi_cov is not None:
            if pi_cov < THRESHOLDS["pi_cov_min"] or pi_cov > THRESHOLDS["pi_cov_max"]:
                reasons.append(f"pi_cov={pi_cov:.3f}")
        if danger_support and danger_recall is not None and danger_recall < THRESHOLDS["danger_recall_min"]:
            reasons.append(f"danger_recall={danger_recall:.3f}")

        if reasons:
            failed.append((sid, h))
            log.info("  FAIL %s h%d: %s", sid, h, ", ".join(reasons))

    return failed


def train_one(station: str, horizon: int, trials: int, years: int) -> bool:
    """Call train_forecast.py for one (station, horizon) and return success."""
    start_date = (date.today() - timedelta(days=1) - timedelta(days=365 * years)).isoformat()
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--station", station,
        "--horizons", str(horizon),
        "--trials", str(trials),
        "--quick-mode",        # CV=1, early_stop=10, single seed
        "--device", "auto",    # GPU if available, else CPU
        "--backend", "lightgbm",
        "--force",
        "--start", start_date,
    ]
    log.info("  CMD: %s", " ".join(cmd))
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        elapsed = time.time() - t0
        log.info("  OK  %s h%d in %.0fs (%.1f min)", station, horizon, elapsed, elapsed / 60)
        return True
    except subprocess.CalledProcessError as e:
        log.error("  FAIL %s h%d: exit=%s", station, horizon, e.returncode)
        return False


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}   # {"{sid}_{h}": rounds_done}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# =======================================================================
# Main loop
# =======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune models until all pass thresholds")
    parser.add_argument("--max-rounds", type=int, default=6,
                        help="Max retry rounds per model (default 6)")
    parser.add_argument("--stations", nargs="*", default=None,
                        help="Limit to specific station IDs (default: all failed)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset retry counter state")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("State reset.")

    state = load_state()
    max_rounds = args.max_rounds

    log.info("=" * 65)
    log.info("AUTO-TUNE LOOP  (max %d rounds per model)", max_rounds)
    log.info("=" * 65)

    global_round = 0
    while True:
        global_round += 1
        log.info("\n--- Global round %d ---", global_round)

        # 1. Get current status
        try:
            status = run_status()
        except Exception as e:
            log.error("check_model_status.py failed: %s", e)
            sys.exit(1)

        # 2. Find all models that still fail
        failed = get_failed(status, args.stations)
        if not failed:
            log.info("\nAll models pass thresholds! Done.")
            break

        log.info("%d model(s) still failing:", len(failed))
        for sid, h in failed:
            key = f"{sid}_{h}"
            done = state.get(key, 0)
            log.info("  %s h%d (retried %d times so far)", sid, h, done)

        # 3. Train each failing model
        any_trained = False
        for sid, h in failed:
            key = f"{sid}_{h}"
            done = state.get(key, 0)

            if done >= max_rounds:
                log.warning("  SKIP %s h%d — reached max rounds (%d)", sid, h, max_rounds)
                continue

            round_idx = min(done, len(TRIAL_SCHEDULE) - 1)
            trials = TRIAL_SCHEDULE[round_idx]
            years = DATA_YEARS_SCHEDULE[round_idx]

            log.info("\nTraining %s h%d  [round %d/%d, trials=%d, years=%d]",
                     sid, h, done + 1, max_rounds, trials, years)

            success = train_one(sid, h, trials, years)
            state[key] = done + 1
            save_state(state)
            any_trained = True

            if not success:
                log.error("  Training failed for %s h%d, continuing...", sid, h)

        if not any_trained:
            log.warning("All remaining failed models have hit max rounds. Stopping.")
            break

    # Final status report
    log.info("\n" + "=" * 65)
    log.info("FINAL STATUS")
    log.info("=" * 65)
    subprocess.run([sys.executable, str(STATUS_SCRIPT)], cwd=ROOT)


if __name__ == "__main__":
    main()
