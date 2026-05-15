#!/usr/bin/env python3
"""Training script with real-time monitoring."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_training(station: str, horizon: int, trials: int, force: bool = False):
    """Run training for a single station/horizon with monitoring."""
    cmd = [
        sys.executable, "scripts/train_forecast.py",
        "--station", station,
        "--horizons", str(horizon),
        "--trials", str(trials),
    ]
    if force:
        cmd.append("--force")
    
    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"[START] {datetime.now().strftime('%H:%M:%S')}")
    print(f"[STATION] {station:8} | [HORIZON] h{horizon:2} | [TRIALS] {trials}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    # Parse results
    if result.returncode == 0:
        print(f"[SUCCESS] ⏱ {minutes}m {seconds}s")
        # Try to extract metrics from registry
        registry_path = Path(f"app/models/forecast_v3/{station}/h{horizon}/registry.json")
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                eval_data = data.get("evaluation", {})
                skill = eval_data.get("baselines", {}).get("skill_score")
                mae = eval_data.get("regression", {}).get("mae")
                status = data.get("status", "unknown")
                danger_recall = eval_data.get("safety", {}).get("danger_42", {}).get("recall")
                pi_width = eval_data.get("prediction_interval", {}).get("mean_width")
                
                skill_str = f"{skill:+.3f}" if skill is not None else "N/A"
                mae_str = f"{mae:.2f}°C" if mae is not None else "N/A"
                recall_str = f"{danger_recall:.1%}" if danger_recall is not None else "N/A"
                width_str = f"{pi_width:.1f}°C" if pi_width is not None else "N/A"
                
                icon = "✅" if status == "ready" else "⚠️" if status == "candidate" else "❌"
                print(f"[METRICS] MAE:{mae_str:10} Skill:{skill_str:10} Recall:{recall_str:8} PI:{width_str:10} | {icon} {status}")
            except Exception as e:
                print(f"[WARN] Could not read metrics: {e}")
        return True, elapsed
    else:
        print(f"[FAILED] Exit code: {result.returncode}")
        if result.stderr:
            print(result.stderr[-800:])  # Last 800 chars
        return False, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", nargs="+", default=["BKK_01", "CNX_01", "KKN_01", "HYI_01", "RYG_01"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[6, 12, 24])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# HeatShield AI - Model Training with Monitoring")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Stations: {', '.join(args.stations)}")
    print(f"# Horizons: {', '.join(map(str, args.horizons))}h")
    print(f"# Trials: {args.trials}")
    print(f"{'#'*70}")
    
    results = []
    total_start = time.time()
    
    for station in args.stations:
        for horizon in args.horizons:
            success, elapsed = run_training(station, horizon, args.trials, args.force)
            results.append({
                "station": station,
                "horizon": horizon,
                "success": success,
                "elapsed": elapsed,
            })
            time.sleep(1)  # Cooldown
    
    # Summary
    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    
    print(f"\n{'#'*70}")
    print(f"# TRAINING COMPLETE")
    print(f"# Total time: {total_min}m {int(total_elapsed % 60)}s")
    print(f"{'#'*70}")
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\n[SUMMARY] Total: {total} | ✅ Passed: {passed} | ❌ Failed: {total - passed}")
    
    # Show all results
    print(f"\n{'='*70}")
    print("All Results:")
    for r in results:
        icon = "✅" if r["success"] else "❌"
        elapsed_min = int(r["elapsed"] // 60)
        print(f"  {icon} {r['station']:8} h{r['horizon']:2} - {elapsed_min}m {int(r['elapsed'] % 60)}s")


if __name__ == "__main__":
    main()
