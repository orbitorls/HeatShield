"""Check status of all trained models using shared readiness gates."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.ml.forecast.readiness import evaluate_readiness

PROJECT_ROOT = Path(__file__).parent.parent
base = PROJECT_ROOT / "app" / "models" / "forecast_v3"
results = []

for station_dir in sorted(base.iterdir()):
    if not station_dir.is_dir():
        continue
    for horizon_dir in sorted(station_dir.iterdir()):
        if not horizon_dir.is_dir() or not horizon_dir.name.startswith("h"):
            continue
        reg_file = horizon_dir / "registry.json"
        if reg_file.exists():
            try:
                with open(reg_file) as f:
                    data = json.load(f)
                ev = data.get("evaluation", {})
                baselines = ev.get("baselines", {})
                regression = ev.get("regression", {})
                safety = ev.get("safety", {})
                danger_42 = safety.get("danger_42", {})
                pi = ev.get("prediction_interval", {})
                horizon_h = int(horizon_dir.name[1:]) if horizon_dir.name.startswith("h") else 6
                readiness = evaluate_readiness(ev, horizon_h)
                results.append({
                    "station": station_dir.name,
                    "horizon": horizon_dir.name,
                    "mae": regression.get("mae"),
                    "skill_score": baselines.get("skill_score"),
                    "danger_recall_42": danger_42.get("recall"),
                    "danger_42_support": danger_42.get("support"),
                    "pi_coverage_90": pi.get("coverage_90"),
                    "n_train_rows": data.get("n_train_rows"),
                    "status": data.get("status", "unknown"),  # backward-compatible field
                    "registry_status": data.get("status", "unknown"),
                    "computed_status": readiness.status,
                    "failed_reasons": list(readiness.failed_reasons),
                })
            except Exception as e:
                print(f"Error reading {reg_file}: {e}")

results.sort(key=lambda x: (x["station"], x["horizon"]))

print(f"Total models: {len(results)}")
print(f"Stations: {len(set(r['station'] for r in results))}")
print()
print("Station      | Horizon | MAE   | Skill | Danger Recall | PI Cov | Rows   | Registry | Computed")
print("-" * 106)
for r in results:
    mae = f"{r['mae']:5.2f}" if r["mae"] is not None else "  N/A"
    skill = f"{r['skill_score']:5.2f}" if r["skill_score"] is not None else "  N/A"
    danger = f"{r['danger_recall_42']:13.2%}" if r["danger_recall_42"] is not None else "          N/A"
    pi_cov = f"{r['pi_coverage_90']:6.1%}" if r["pi_coverage_90"] is not None else "   N/A"
    rows = f"{r['n_train_rows']:6}" if r["n_train_rows"] else "   N/A"
    print(
        f"{r['station']:<12} | {r['horizon']:<7} | {mae} | {skill} | {danger} | {pi_cov} | {rows} | "
        f"{r['registry_status']:<8} | {r['computed_status']}"
    )

failed = []
for r in results:
    if r["failed_reasons"]:
        failed.append({**r, "reasons": ", ".join(r["failed_reasons"])})

print()
print(f"FAILED THRESHOLDS: {len(failed)}/{len(results)}")
print("Station      | Horizon | Reasons")
print("-" * 45)
for f in failed:
    print(f"{f['station']:<12} | {f['horizon']:<7} | {f['reasons']}")

# Categorize failures
reason_counter = Counter()
station_counter = Counter()
for f in failed:
    for reason in f["reasons"].split(", "):
        reason_counter[reason.strip()] += 1
    station_counter[f["station"]] += 1

print()
print("FAILURE BREAKDOWN:")
for reason, count in reason_counter.most_common():
    print(f"  {reason}: {count} models")

print()
print("STATIONS WITH MOST FAILURES:")
for station, count in station_counter.most_common():
    print(f"  {station}: {count} horizons failed")

# Save for later
out_dir = PROJECT_ROOT / "logs"
out_dir.mkdir(exist_ok=True)
with open(out_dir / "model_status.json", "w") as f:
    json.dump({"all": results, "failed": failed}, f, indent=2)
print()
print("Saved to logs/model_status.json")
