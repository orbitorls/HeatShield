# HeatShield AI Model Accuracy & Training Speed Optimization

## Overview

**Goal:** Improve heat index forecast model accuracy and training speed across all 25 models (5 stations × 5 horizons) using GPU CUDA + balanced approach.

## Current Problems

### Accuracy Issues
1. **Data Leakage:** `_oversample_danger_rows()` runs before `split_xy_4way()` causing synthetic rows to leak into test split
2. **Negative Skill Scores:** Multiple models (BKK, CNX, HYI) worse than climatology baseline
3. **Insufficient Optuna Trials:** 20 trials for 10+ hyperparameters is too few
4. **Feature Noise:** 110+ features with constant/redundant columns
5. **Danger Recall Drop:** Sharp decline from 40°C to 42°C threshold

### Speed Issues
1. **Sequential GPU Training:** 14 boosters trained one-at-a-time in triple-nested loop
2. **O(n²) KDE:** gaussian_kde called every Optuna trial
3. **Multiple pd.concat:** 8+ sequential concat calls in feature building
4. **Excessive gc.collect():** 3-4 calls adding unnecessary overhead

## Design

### Phase 1: Fix Data Leakage
Move `_oversample_danger_rows()` inside `LGBMForecaster.fit()` AFTER `split_xy_4way()`. Only apply oversampling to train split.

### Phase 2: GPU-Accelerated Parallel Training
Replace sequential GPU loop with ThreadPoolExecutor (max 4 concurrent jobs). Force CUDA device in Optuna trial params.

### Phase 3: Optuna Tuning Enhancement
Increase trials to 40 (h6) / 35 (h24) / 25 (h48+). Reduce startup trials to 5. Replace Hyperband with MedianPruner. Increase tuning rounds to 100.

### Phase 4: Feature & Speed Optimization
- Batch concat in features.py (8+ → 2-3 concats)
- Replace gaussian_kde with histogram density estimation (O(n) vs O(n²))
- Remove 2 of 3 gc.collect() calls
- Feature importance pruning after first station training

### Phase 5: Pilot Training Enhancement
Increase pilot max rounds from 300 to 500. Add optional multi-target (temp_c + rh) pilot.

### Phase 6: Post-Fix Re-evaluation
Run full training for all stations, update quality gate thresholds based on honest metrics.

## Expected Outcomes

| Metric | Before | Expected After |
|--------|--------|----------------|
| Models achieving "ready" status | 1/25 | 3-5/25 |
| Avg MAE | 1.5-2.4°C | 1.3-2.0°C |
| Training time per slot | 160-280s | 60-120s |
| Total training time (25 slots) | ~1.5-2h | ~25-50min |
