"""Extract metrics from model registry to create leaderboard without accessing logs directory."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def extract_metrics_from_slot(slot_dir: Path) -> dict:
    """Extract metrics from a single model slot's registry.json."""
    registry_path = slot_dir / "registry.json"
    if not registry_path.exists():
        return None
    
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        evaluation = data.get("evaluation", {})
        regression = evaluation.get("regression", {})
        baselines = evaluation.get("baselines", {})
        safety = evaluation.get("safety", {})
        danger_42 = safety.get("danger_42", {})
        
        return {
            "backend": data.get("backend_name", "unknown"),
            "station": data.get("station_id", "unknown"),
            "horizon_h": data.get("horizon_h", 0),
            "mae": regression.get("mae", float("nan")),
            "skill_score": baselines.get("skill_score", float("nan")),
            "danger_recall_42": danger_42.get("recall", float("nan")),
            "status": data.get("status", "unknown"),
            "n_train_rows": data.get("n_train_rows", 0),
        }
    except Exception as e:
        logger.warning(f"Failed to read {registry_path}: {e}")
        return None

def main():
    model_root = Path("app/models/forecast_v3")
    if not model_root.exists():
        logger.error(f"Model root not found: {model_root}")
        return
    
    stations = [d for d in model_root.iterdir() if d.is_dir()]
    all_metrics: List[dict] = []
    
    for station_dir in stations:
        station_id = station_dir.name
        horizon_dirs = [d for d in station_dir.iterdir() if d.is_dir() and d.name.startswith("h")]
        
        for horizon_dir in horizon_dirs:
            try:
                horizon_h = int(horizon_dir.name[1:])
            except ValueError:
                continue
            
            metrics = extract_metrics_from_slot(horizon_dir)
            if metrics:
                metrics["station"] = station_id
                metrics["horizon_h"] = horizon_h
                all_metrics.append(metrics)
    
    # Sort by station, then horizon
    all_metrics.sort(key=lambda x: (x["station"], x["horizon_h"]))
    
    # Print leaderboard
    print("\n" + "=" * 80)
    print("MODEL LEADERBOARD (from registry.json)")
    print("=" * 80)
    print(f"{'Backend':<12} {'Station':<8} {'Horizon':<8} {'MAE':<8} {'Skill':<8} {'Danger Recall':<12} {'Status':<12}")
    print("-" * 80)
    
    for m in all_metrics:
        mae_val = m['mae'] if m['mae'] is not None else float('nan')
        skill_val = m['skill_score'] if m['skill_score'] is not None else float('nan')
        danger_val = m['danger_recall_42'] if m['danger_recall_42'] is not None else float('nan')
        
        mae_str = f"{mae_val:.3f}" if mae_val == mae_val else "nan"
        skill_str = f"{skill_val:.3f}" if skill_val == skill_val else "nan"
        danger_str = f"{danger_val:.3f}" if danger_val == danger_val else "nan"
        
        print(
            f"{m['backend']:<12} {m['station']:<8} h{m['horizon_h']:<7} "
            f"{mae_str:<8} {skill_str:<8} "
            f"{danger_str:<12} {m['status']:<12}"
        )
    
    print("=" * 80)
    print(f"Total slots: {len(all_metrics)}")
    
    # Calculate averages
    valid_mae = [m['mae'] for m in all_metrics if m['mae'] is not None]
    valid_skill = [m['skill_score'] for m in all_metrics if m['skill_score'] is not None]
    valid_danger = [m['danger_recall_42'] for m in all_metrics if m['danger_recall_42'] is not None]
    
    if valid_mae:
        print(f"Average MAE: {sum(valid_mae)/len(valid_mae):.3f}")
    if valid_skill:
        print(f"Average Skill Score: {sum(valid_skill)/len(valid_skill):.3f}")
    if valid_danger:
        print(f"Average Danger Recall: {sum(valid_danger)/len(valid_danger):.3f}")
    
    # Save to JSON
    output_path = Path("model_metrics.json")
    output_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    main()
