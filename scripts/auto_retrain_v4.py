"""
Watcher: monitors ERA5 ingest PID, then runs full retrain.
Usage: python scripts/auto_retrain_v4.py --ingest-pid <PID>
"""
import argparse
import subprocess
import sys
import time
import os
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")


def pid_alive(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=10,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def run(cmd: list[str], log_file: Path) -> int:
    log.info("Running: %s", " ".join(cmd))
    with open(log_file, "a") as f:
        f.write(f"\n\n=== {datetime.now().isoformat()} ===\n")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        proc.wait()
    log.info("Finished with exit code %d", proc.returncode)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest-pid", type=int, required=True)
    ap.add_argument("--poll-interval", type=int, default=120)
    args = ap.parse_args()

    log.info("Watcher started. Monitoring ingest PID=%d", args.ingest_pid)

    # Wait for ingest to finish
    while pid_alive(args.ingest_pid):
        log.info("ERA5 ingest PID=%d still running — sleeping %ds", args.ingest_pid, args.poll_interval)
        time.sleep(args.poll_interval)

    log.info("ERA5 ingest PID=%d finished. Starting retrain.", args.ingest_pid)

    # Check ingest log for completion
    ingest_log = ROOT / "logs" / "era5_yearly_v1.log"
    if ingest_log.exists():
        tail = ingest_log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
        log.info("Ingest log tail:\n%s", "\n".join(tail))

    # Full retrain
    retrain_log = ROOT / "logs" / "retrain_v4.log"
    rc = run([
        PYTHON, str(ROOT / "scripts" / "train_forecast.py"),
        "--model-version", "v3",
        "--backend", "lightgbm",
        "--horizons", "6,12,24,48,72",
        "--trials", "30",
        "--force",
        "--workers", "4",
    ], retrain_log)

    if rc != 0:
        log.error("Retrain failed (exit %d). See %s", rc, retrain_log)
        sys.exit(rc)

    log.info("Retrain complete. Running evaluation...")
    eval_log = ROOT / "logs" / "eval_v4.log"
    run([PYTHON, str(ROOT / "scripts" / "evaluate_model.py")], eval_log)

    log.info("Done. Check logs/retrain_v4.log and logs/eval_v4.log for results.")


if __name__ == "__main__":
    main()
