"""Scan all v3 model slots and report readiness status.

Usage:
    python scripts/scan_readiness.py
    python scripts/scan_readiness.py --detail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.ml.forecast.readiness import evaluate_readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan v3 model readiness")
    parser.add_argument("--detail", action="store_true", help="Show per-slot details")
    parser.add_argument("--root", type=str, default="app/models/forecast_v3",
                        help="Root directory for v3 models")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root directory not found: {root}")
        sys.exit(1)

    slots: list[dict] = []
    for station_dir in sorted(root.iterdir()):
        if not station_dir.is_dir():
            continue
        sid = station_dir.name
        for h_dir in sorted(station_dir.iterdir()):
            if not h_dir.is_dir() or not h_dir.name.startswith("h"):
                continue
            horizon_h = int(h_dir.name[1:])
            reg_path = h_dir / "registry.json"
            if not reg_path.exists():
                slots.append({
                    "station": sid,
                    "horizon": horizon_h,
                    "status": "missing",
                    "reason": "registry.json not found",
                })
                continue
            try:
                data = json.loads(reg_path.read_text(encoding="utf-8"))
            except Exception as exc:
                slots.append({
                    "station": sid,
                    "horizon": horizon_h,
                    "status": "error",
                    "reason": str(exc),
                })
                continue

            eval_data = data.get("evaluation", {})
            readiness = evaluate_readiness(eval_data, horizon_h)
            slots.append({
                "station": sid,
                "horizon": horizon_h,
                "status": readiness.status,
                "reasons": ", ".join(readiness.failed_reasons) if readiness.failed_reasons else "",
                "mae": readiness.mae,
                "skill": readiness.skill_score,
                "danger_recall": readiness.danger_recall_42,
                "pi_coverage": readiness.pi_coverage_90,
                "backend": data.get("backend_name", "unknown"),
            })

    # Summary
    total = len(slots)
    ready = sum(1 for s in slots if s["status"] == "ready")
    candidate = sum(1 for s in slots if s["status"] == "candidate")
    not_ready = sum(1 for s in slots if s["status"] == "not_ready")
    missing = sum(1 for s in slots if s["status"] == "missing")

    print("=" * 80)
    print("HEATSHIELD AI v3 Model Readiness Scan")
    print("=" * 80)
    print(f"Total slots:   {total}")
    print(f"Ready:         {ready}  ({ready/total*100:.1f}%)")
    print(f"Candidate:     {candidate}  ({candidate/total*100:.1f}%)")
    print(f"Not Ready:     {not_ready}  ({not_ready/total*100:.1f}%)")
    print(f"Missing:       {missing}  ({missing/total*100:.1f}%)")
    print("=" * 80)

    if args.detail:
        print("\nPer-slot details:")
        print("-" * 80)
        print(f"{'Station':<12} {'H':<4} {'Status':<12} {'MAE':<8} {'Skill':<8} {'DangerR':<8} {'PI':<8} {'Backend':<20} {'Reasons'}")
        print("-" * 80)
        for s in slots:
            mae_str = f"{s['mae']:.3f}" if s.get('mae') is not None else "N/A"
            skill_str = f"{s['skill']:.3f}" if s.get('skill') is not None else "N/A"
            danger_str = f"{s['danger_recall']:.3f}" if s.get('danger_recall') is not None else "N/A"
            pi_str = f"{s['pi_coverage']:.3f}" if s.get('pi_coverage') is not None else "N/A"
            print(
                f"{s['station']:<12} {s['horizon']:<4} {s['status']:<12} "
                f"{mae_str:<8} {skill_str:<8} {danger_str:<8} {pi_str:<8} "
                f"{s.get('backend', 'unknown'):<20} {s.get('reasons', s.get('reason', ''))}"
            )
        print("-" * 80)

    # List not-ready and candidate slots
    problematic = [s for s in slots if s["status"] in ("candidate", "not_ready", "missing")]
    if problematic:
        print("\nProblematic slots (needs retraining or investigation):")
        print("-" * 80)
        for s in problematic:
            reason = s.get("reasons", s.get("reason", ""))
            print(f"  {s['station']} h{s['horizon']}: {s['status']} — {reason}")
        print("-" * 80)
    else:
        print("\nAll slots are READY!")


if __name__ == "__main__":
    main()
