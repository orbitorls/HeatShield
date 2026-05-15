"""Wait for ERA5 ingest to finish, then auto-start full 25-slot retrain.

Usage:
    python scripts/auto_retrain_after_ingest.py --ingest-pid 31348
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parents[1]


def pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback: try os.kill(pid, 0) on Unix; on Windows use tasklist
        import subprocess
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True
        )
        return str(pid) in result.stdout


def wait_for_ingest(ingest_pid: int, poll_seconds: int = 60) -> None:
    logger.info("Waiting for ERA5 ingest PID=%d to finish…", ingest_pid)
    while pid_alive(ingest_pid):
        time.sleep(poll_seconds)
        logger.info("ERA5 ingest PID=%d still running…", ingest_pid)
    logger.info("ERA5 ingest PID=%d finished.", ingest_pid)


def run_retrain() -> int:
    log_path = BASE_DIR / "logs" / "retrain_v48_era5.log"
    log_path.parent.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(BASE_DIR / "scripts" / "train_forecast.py"),
        "--model-version", "v3",
        "--backend", "lightgbm",
        "--horizons", "6,12,24,48,72",
        "--trials", "30",
        "--force",
        "--workers", "4",
    ]
    logger.info("Starting retrain: %s", " ".join(cmd))
    logger.info("Retrain log: %s", log_path)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=lf, stderr=lf)
        pid_file = BASE_DIR / "logs" / "retrain_v48_era5.pid"
        pid_file.write_text(str(proc.pid))
        logger.info("Retrain started PID=%d", proc.pid)
        proc.wait()
    rc = proc.returncode
    if rc == 0:
        logger.info("Retrain completed successfully.")
    else:
        logger.error("Retrain exited with code %d — check %s", rc, log_path)
    return rc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-pid", type=int, required=True,
                        help="PID of the running ERA5 ingest process")
    parser.add_argument("--poll", type=int, default=120,
                        help="Poll interval in seconds (default: 120)")
    parser.add_argument("--retrain-only", action="store_true",
                        help="Skip waiting, start retrain immediately")
    args = parser.parse_args()

    if not args.retrain_only:
        wait_for_ingest(args.ingest_pid, poll_seconds=args.poll)
    else:
        logger.info("--retrain-only: skipping wait, starting retrain now")

    rc = run_retrain()
    sys.exit(rc)


if __name__ == "__main__":
    main()
