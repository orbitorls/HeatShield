# Model Accuracy & Training Speed Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix data leakage, accelerate training via GPU, improve hyperparameter tuning quality, and optimize feature engineering.

**Architecture:** Six-phase approach modifying lgbm_backend.py, train_forecast.py, and features.py.

**Tech Stack:** Python 3.10+, LightGBM 4.5+, Optuna 3.6+, CUDA, pandas, NumPy

---

## File Structure

| File | Changes |
|------|---------|
| `app/ml/forecast/backends/lgbm_backend.py` | Fix leakage, GPU parallel, Optuna params, pilot rounds, KDE, GC |
| `scripts/train_forecast.py` | Remove oversampling before fit(), feature pruning, device override |
| `app/ml/forecast/features.py` | Batch concat optimization |

---

### Task 1: Fix Data Leakage — Move Oversampling Inside fit()

**Files:**
- Modify: `app/ml/forecast/backends/lgbm_backend.py:496-600`
- Modify: `scripts/train_forecast.py:698-703`

- [ ] **Step 1: Add `_maybe_oversample_train()` method to LGBMForecaster**
- [ ] **Step 2: Call oversampling after split_xy_4way() in fit()**
- [ ] **Step 3: Remove oversampling call from scripts/train_forecast.py**
- [ ] **Step 4: Run existing tests to verify no regression**

### Task 2: GPU Parallel Refit

**Files:**
- Modify: `app/ml/forecast/backends/lgbm_backend.py:614-663`

- [ ] **Step 1: Replace sequential GPU loop with ThreadPoolExecutor**
- [ ] **Step 2: Verify GPU parallel works (falls back to CPU gracefully)**

### Task 3: Optuna Tuning Enhancement

**Files:**
- Modify: `app/ml/forecast/backends/lgbm_backend.py:731-883`

- [ ] **Step 1: Update default n_trials to 40**
- [ ] **Step 2: Update _adaptive_trials() table**
- [ ] **Step 3: Replace pruner, reduce startup trials**
- [ ] **Step 4: Verify Optuna changes**

### Task 4: Feature & Speed Optimization

**Files:**
- Modify: `app/ml/forecast/features.py:278-528`
- Modify: `app/ml/forecast/backends/lgbm_backend.py:260-283,572-603`
- Modify: `scripts/train_forecast.py:610-640`

- [ ] **Step 1: Batch concat in features.py**
- [ ] **Step 2: Replace gaussian_kde with histogram**
- [ ] **Step 3: Reduce gc.collect() calls**
- [ ] **Step 4: Add feature pruning in train_forecast.py**
- [ ] **Step 5: Run tests**

### Task 5: Pilot Training Enhancement

**Files:**
- Modify: `app/ml/forecast/backends/lgbm_backend.py:584-599`

- [ ] **Step 1: Increase pilot max rounds from 300 to 500**
- [ ] **Step 2: Add optional multi-target pilot (temp_c + rh)**

### Task 6: Post-Fix Re-evaluation

- [ ] **Step 1: Run full training for all stations**
- [ ] **Step 2: Check new quality statuses**
- [ ] **Step 3: Update quality gate thresholds**

### Task 7: Smoke Test Verification

- [ ] **Step 1: Run smoke test**
- [ ] **Step 2: Run full test suite**
