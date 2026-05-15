#!/usr/bin/env bash
# One-click Colab training pipeline (ingest -> validate -> train).
# Usage:
#   bash scripts/colab_train_real.sh
#   TRIALS=120 DEVICE=gpu GATE_BACKEND=brf bash scripts/colab_train_real.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/Heat-wave-backend}"
START_DATE="${START_DATE:-2021-01-01}"
END_DATE="${END_DATE:-$(date -u +%Y-%m-%d)}"
TRIALS="${TRIALS:-50}"
DEVICE="${DEVICE:-gpu}"
GATE_BACKEND="${GATE_BACKEND:-brf}"
MIN_ROWS="${MIN_ROWS:-500}"
RUN_INGEST="${RUN_INGEST:-1}"
WORKERS="${WORKERS:-1}"
LGBM_PARALLEL_BOOSTERS="${LGBM_PARALLEL_BOOSTERS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/colab_train_real.sh [options]

Options:
  --start YYYY-MM-DD       Ingest/training window start (default: 2021-01-01)
  --end YYYY-MM-DD         Ingest/training window end (default: today UTC)
  --trials N               Optuna trials (default: 120)
  --device MODE            gpu|cuda|cpu|auto (default: gpu)
  --gate-backend NAME      brf|lightgbm (default: brf)
  --min-rows N             Minimum rows per station before training (default: 500)
  --workers N              Parallel station workers (default: 1)
  --parallel-boosters N    CPU-only LightGBM booster jobs (default: 1)
  --skip-ingest            Skip ingest step and train from existing data
  --help                   Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --start) START_DATE="$2"; shift 2 ;;
    --end) END_DATE="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --gate-backend) GATE_BACKEND="$2"; shift 2 ;;
    --min-rows) MIN_ROWS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --parallel-boosters) LGBM_PARALLEL_BOOSTERS="$2"; shift 2 ;;
    --skip-ingest) RUN_INGEST=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "[colab-train-real] ERROR: unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ ! -d "${REPO_DIR}" ]; then
  echo "[colab-train-real] ERROR: repo dir not found: ${REPO_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"

echo "[colab-train-real] repo      : ${REPO_DIR}"
echo "[colab-train-real] start/end : ${START_DATE} -> ${END_DATE}"
echo "[colab-train-real] trials    : ${TRIALS}"
echo "[colab-train-real] device    : ${DEVICE}"
echo "[colab-train-real] gate      : ${GATE_BACKEND}"
echo "[colab-train-real] min_rows  : ${MIN_ROWS}"

export LGBM_PARALLEL_BOOSTERS
if [ "${DEVICE}" = "gpu" ] || [ "${DEVICE}" = "cuda" ]; then
  WORKERS=1
  LGBM_PARALLEL_BOOSTERS=1
  export LGBM_PARALLEL_BOOSTERS
fi
echo "[colab-train-real] workers   : ${WORKERS}"
echo "[colab-train-real] lgbm parallel boosters: ${LGBM_PARALLEL_BOOSTERS}"

if [ ! -d "/content/drive/MyDrive" ]; then
  echo "[colab-train-real] WARN: Drive not mounted. If you need persistence, mount Drive in a Python cell first."
fi

echo "[colab-train-real] Installing training dependencies..."
pip install -r requirements-train.txt

if [ "${RUN_INGEST}" = "1" ]; then
  echo "[colab-train-real] Ingesting NASA POWER data..."
  python scripts/ingest_nasa_power.py --start "${START_DATE}" --end "${END_DATE}"
else
  echo "[colab-train-real] Skipping ingest step (--skip-ingest)."
fi

echo "[colab-train-real] Validating row counts..."
python - "${START_DATE}" "${END_DATE}" "${MIN_ROWS}" <<'PY'
from datetime import date
import sys
from app.data.stations import STATIONS
from app.data.loaders import read_observations

start = date.fromisoformat(sys.argv[1])
end = date.fromisoformat(sys.argv[2])
min_rows = int(sys.argv[3])

bad = []
for sid in STATIONS:
    n = len(read_observations(sid, start, end))
    print(f"{sid}: {n}")
    if n < min_rows:
        bad.append((sid, n))

if bad:
    raise SystemExit(f"Not enough rows for training: {bad}")

print("Data readiness check: OK")
PY

echo "[colab-train-real] Detecting train_forecast flags..."
HELP="$(python scripts/train_forecast.py --help 2>&1 || true)"
CMD=(python scripts/train_forecast.py --trials "${TRIALS}" --start "${START_DATE}" --end "${END_DATE}" --workers "${WORKERS}")

if echo "${HELP}" | grep -q -- "--device"; then
  CMD+=(--device "${DEVICE}")
fi
if echo "${HELP}" | grep -q -- "--gate-backend"; then
  CMD+=(--gate-backend "${GATE_BACKEND}")
fi

echo "[colab-train-real] Training command: ${CMD[*]}"
"${CMD[@]}"

echo "[colab-train-real] Done."
echo "[colab-train-real] Latest eval runs:"
ls -lt logs/eval/runs 2>/dev/null | head -n 5 || true
