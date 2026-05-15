#!/usr/bin/env bash
# High-quality Colab training launcher (GPU + full evaluation artifacts).
# Safe to re-run: existing (station,horizon) slots are skipped by default.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/Heat-wave-backend}"
if [ ! -d "${REPO_DIR}" ]; then
  echo "[colab-train] ERROR: repo not found at ${REPO_DIR}" >&2
  exit 1
fi
cd "${REPO_DIR}"

if [ ! -f "scripts/train_forecast.py" ]; then
  echo "[colab-train] ERROR: scripts/train_forecast.py not found" >&2
  exit 1
fi

DEFAULT_END="$(python -c "from datetime import date,timedelta; print((date.today()-timedelta(days=1)).isoformat())")"
DEFAULT_START="$(python -c "from datetime import date,timedelta; print((date.today()-timedelta(days=365*5)).isoformat())")"
DEFAULT_RUN_ID="colab_full_$(date -u +%Y%m%dT%H%M%SZ)"

START_DATE="${START_DATE:-${DEFAULT_START}}"
END_DATE="${END_DATE:-${DEFAULT_END}}"
TRIALS="${TRIALS:-50}"
HORIZONS="${HORIZONS:-6,12,24,48,72}"
DEVICE="${DEVICE:-gpu}"
GATE_BACKEND="${GATE_BACKEND:-brf}"
WORKERS="${WORKERS:-1}"
LGBM_PARALLEL_BOOSTERS="${LGBM_PARALLEL_BOOSTERS:-1}"
RUN_ID="${RUN_ID:-${DEFAULT_RUN_ID}}"
STATION="${STATION:-}"
FORCE_RETRAIN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/colab_train_full.sh [options]

Options:
  --start YYYY-MM-DD       Data start date (default: 5 years ago)
  --end YYYY-MM-DD         Data end date (default: yesterday)
  --trials N               Optuna trials per slot (default: 120)
  --horizons CSV           Horizon list (default: 6,12,24,48,72)
  --station ID             Single station id (default: all stations)
  --device MODE            auto|cpu|gpu|cuda (default: gpu)
  --gate-backend NAME      lightgbm|brf (default: brf)
  --workers N              Parallel station workers (default: 1)
  --parallel-boosters N    CPU-only LightGBM booster jobs (default: 1)
  --run-id ID              Run directory id under logs/eval/runs/
  --force                  Retrain slots even if bundle already exists
  --help                   Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --start) START_DATE="$2"; shift 2 ;;
    --end) END_DATE="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --horizons) HORIZONS="$2"; shift 2 ;;
    --station) STATION="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --gate-backend) GATE_BACKEND="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --parallel-boosters) LGBM_PARALLEL_BOOSTERS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --force) FORCE_RETRAIN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "[colab-train] ERROR: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p logs/train
LOG_PATH="logs/train/${RUN_ID}.log"

export LGBM_PARALLEL_BOOSTERS
if [ "${DEVICE}" = "gpu" ] || [ "${DEVICE}" = "cuda" ]; then
  WORKERS=1
  LGBM_PARALLEL_BOOSTERS=1
  export LGBM_PARALLEL_BOOSTERS
fi

cmd=(
  python scripts/train_forecast.py
  --start "${START_DATE}"
  --end "${END_DATE}"
  --trials "${TRIALS}"
  --horizons "${HORIZONS}"
  --device "${DEVICE}"
  --gate-backend "${GATE_BACKEND}"
  --workers "${WORKERS}"
  --run-id "${RUN_ID}"
)

if [ -n "${STATION}" ]; then
  cmd+=(--station "${STATION}")
fi
if [ "${FORCE_RETRAIN}" -eq 1 ]; then
  cmd+=(--force)
fi

echo "[colab-train] repo      : ${REPO_DIR}"
echo "[colab-train] run_id    : ${RUN_ID}"
echo "[colab-train] date range: ${START_DATE} -> ${END_DATE}"
echo "[colab-train] trials    : ${TRIALS}"
echo "[colab-train] horizons  : ${HORIZONS}"
echo "[colab-train] device    : ${DEVICE}"
echo "[colab-train] gate      : ${GATE_BACKEND}"
echo "[colab-train] workers   : ${WORKERS}"
echo "[colab-train] lgbm parallel boosters: ${LGBM_PARALLEL_BOOSTERS}"
echo "[colab-train] log       : ${LOG_PATH}"
echo "[colab-train] command   : ${cmd[*]}"

"${cmd[@]}" 2>&1 | tee "${LOG_PATH}"

echo "[colab-train] done."
echo "[colab-train] artifacts: logs/eval/runs/${RUN_ID}"
echo "[colab-train] leaderboard: logs/eval/runs/${RUN_ID}/leaderboard.json"
