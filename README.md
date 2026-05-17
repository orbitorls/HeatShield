# HeatShield AI — Backend

> Adaptive Heat-Health Risk Intelligence for School Safety and Outdoor Workers
>
> *"We do not forecast the weather. We convert heat into safer decisions."*

HeatShield AI เป็นระบบ AI ที่แปลง `อุณหภูมิ + ความชื้น + พื้นที่ + กิจกรรม` ให้กลายเป็น **คะแนนความเสี่ยงสุขภาพ** และ **คำแนะนำเชิงปฏิบัติ** สำหรับโรงเรียนและคนทำงานกลางแจ้ง — ไม่ยึดนิยาม heatwave แบบตัวเลขเดียว เพราะแต่ละพื้นที่มีภูมิอากาศและความเปราะบางไม่เหมือนกัน

ที่มาแนวคิด: ดูเอกสาร `README.md` และ `AGENTS.md`

---

## ระบบ 3 ชั้น

| ชั้น | คำถามที่ตอบ | วิธีคิด |
| ------ | ------------- | --------- |

| 1) Heatwave Event Detection | ช่วงนี้เป็นเหตุการณ์ร้อนผิดปกติเมื่อเทียบพื้นที่นั้นหรือไม่? | local percentile (90/95th) + consecutive days/nights |
| 2) Heat-Health Risk Scoring | วันนี้เสี่ยงต่อสุขภาพระดับไหนสำหรับกลุ่มนี้? | heat index + humidity + time of day + exposure duration + activity intensity + vulnerability |
| 3) Decision Support | ควรทำอะไรต่อ? | action card: เลื่อน/งด/พัก/แจ้งครู-หัวหน้างาน |

## สถาปัตยกรรม 5 Layers

```text
Data Layer       → TMD API, ERA5, NASA POWER, MODIS, OpenStreetMap
Model Layer      → LightGBM Quantile Regression + rule-based safety threshold
Risk Layer       → Scoring engine + local percentile + vulnerability weight
Decision Layer   → Rule engine + templates + risk fusion
Product Layer    → React/Next.js dashboard + FastAPI + PostgreSQL/PostGIS

```

## โครงสร้างโปรเจค

```text
app/
├── api/                   ← FastAPI route modules
│   ├── forecast.py        ← Heat index forecasting endpoints
│   ├── risk.py            ← Risk assessment and classification
│   ├── events.py          ← Heatwave event detection
│   ├── events_auto.py    ← Automatic event detection
│   ├── whatif.py          ← What-if scenario simulation
│   ├── action_card.py    ← Action card generation
│   ├── heat_index.py     ← Heat index calculation
│   ├── stations.py       ← Station metadata
│   └── profiles.py        ← Vulnerability profiles
├── core/                  ← Domain logic (business rules)
│   ├── heat_index.py      ← Rothfusz equation
│   ├── risk_scoring.py    ← Risk category classification
│   ├── whatif.py          ← What-if scenario simulation
│   ├── adaptive_definition.py  ← Local percentile + consecutive days
│   ├── vulnerability.py   ← Profile catalog (student, worker, ...)
│   ├── action_card.py     ← Action card generator
│   ├── calibration.py     ← Bias calibration
│   ├── risk_fusion.py     ← Multi-source risk fusion
│   ├── edr.py             ← Error detection and response
│   ├── monitoring.py      ← System monitoring and health checks
│   └── config.py          ← Configuration management
├── ml/                    ← Machine learning pipeline
│   ├── forecast/          ← Heat index forecasting models
│   │   ├── backends/      ← Model implementations (LightGBM Quantile)
│   │   ├── training.py    ← Model training pipeline
│   │   ├── evaluation.py  ← Model evaluation and metrics
│   │   ├── prediction.py  ← Model inference
│   │   ├── features.py    ← Feature engineering
│   │   ├── splitting.py   ← Data splitting strategies
│   │   ├── conformal.py   ← Conformal prediction
│   │   └── danger_gate.py ← Danger threshold classification
│   ├── registry.py       ← Model versioning and metadata
│   ├── report.py         ← Model evaluation reporting
│   └── viz.py             ← Model visualization
├── data/                  ← Data access layer
│   ├── clients/           ← External API clients (TMD, ERA5, NASA POWER, MODIS)
│   ├── schemas/           ← Pydantic data models
│   ├── loaders/           ← Data ingestion and preprocessing
│   ├── quality.py         ← Data quality validation
│   ├── circuit_breaker.py ← API circuit breaker
│   └── cross_source_validation.py ← Multi-source validation
├── models/                ← Trained model artifacts
│   └── forecast_v3/      ← Current model version (LightGBM Quantile)
│       └── <station_id>/h<horizon>/  ← Per-station, per-horizon models
docs/                      ← Documentation and design specs
tests/                     ← Test suite (pytest)
scripts/                   ← Operational scripts
configs/                   ← Configuration files
frontend/                  ← React/Next.js frontend application

```

## สถานีตรวจวัดอากาศ (16 สถานีทั่วประเทศไทย)

### ภาคกลาง (3 สถานี)

- **BKK_01** - กรุงเทพมหานคร (Don Mueang)
- **NSW_01** - นครสวรรค์
- **SPB_01** - สุพรรณบุรี

### ภาคเหนือ (3 สถานี)

- **CNX_01** - เชียงใหม่
- **CEI_01** - เชียงราย
- **LPT_01** - ลำปาง

### ภาคตะวันออกเฉียงเหนือ/อีสาน (4 สถานี)

- **KKN_01** - ขอนแก่น
- **UDN_01** - อุดรธานี
- **NMA_01** - นครราชสีมา (โคราช)
- **UBN_01** - อุบลราชธานี

### ภาคตะวันออก (3 สถานี)

- **RYG_01** - ระยอง
- **JTI_01** - จันทบุรี
- **CBI_01** - ชลบุรี

### ภาคใต้ (3 สถานี)

- **HYI_01** - หาดใหญ่ (สงขลา)
- **HKT_01** - ภูเก็ต
- **NST_01** - นครศรีธรรมราช

## การติดตั้ง (development)

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# รัน API
uvicorn app.main:app --reload

# รันเทส
pytest

# รันเทสที่ช้า (ต้องมีข้อมูลจริง)
pytest -m slow

# ดู API documentation
# เปิด http://localhost:8000/docs

```

## Scripts หลัก

### Data Ingestion

```bash
python scripts/ingest_all.py           # Ingest data from all sources
python scripts/ingest_tmd.py           # TMD observations
python scripts/ingest_era5.py          # ERA5 reanalysis
python scripts/ingest_nasa_power.py    # NASA POWER satellite data
python scripts/ingest_modis.py         # MODIS satellite data

```

### Model Training & Evaluation

```bash
python scripts/train_forecast.py       # Train LightGBM quantile models
python scripts/evaluate_model.py        # Evaluate trained models
python scripts/generate_model_report_latex.py  # Generate comprehensive PDF report

```

### Auto-retraining & Monitoring

```bash
python scripts/auto_retrain_pipeline.py  # Automatic retraining pipeline
python scripts/auto_retrain_after_ingest.py  # Retrain after data ingestion
python scripts/train_with_monitor.py     # Training with monitoring

```

### Utilities

```bash
python scripts/seed_demo_data.py         # Seed demo data
python scripts/check_coverage.py         # Check data coverage
python scripts/check_tmd_health.py      # Check TMD API health

```

## Endpoints หลัก

| Method | Path | หน้าที่ |
| -------- | ------ | --------- |
| GET | `/health` | Health check |
| GET | `/health/detailed` | Detailed health with monitoring metrics |
| POST | `/forecast/heat-index` | Heat index forecasting with ML models |
| POST | `/heat-index` | คำนวณ heat index จาก temp/humidity |
| POST | `/events/detect` | ตรวจจับ heatwave event ด้วย local percentile |
| POST | `/events/auto` | Automatic heatwave event detection |
| POST | `/risk/score` | คำนวณ Heat-Health Risk Score |
| POST | `/whatif/simulate` | จำลองผลของการเลื่อนเวลา/เพิ่มพักน้ำ/ย้ายสถานที่ |
| POST | `/action-card` | สร้างคำแนะนำเชิงปฏิบัติ |
| GET | `/stations` | List all weather stations |
| GET | `/profiles` | List vulnerability profiles |

## Model Architecture

### Forecasting Model (v3 - LightGBM Quantile Regression)
- **Backend**: LightGBM Quantile Regression with three quantile heads (5th, 50th, 95th percentiles)

- **Target**: Two-head (temperature_c, relative_humidity) at forecast horizon t+h
- **Composition**: Heat index computed via Rothfusz equation at inference time

- **Features**: 108 engineered features including temporal, geographic, meteorological, lag, and rolling statistics
- **Forecast Horizons**: 6h, 12h, 24h, 48h, 72h

- **Training Period**: 2021-2025 data from TMD observations and ERA5 reanalysis
- **Data Split**: 80% training, 10% validation, 10% test (chronological)

### Model Status

- **Ready for Deployment**: BKK_01 (all horizons), CNX_01 (all horizons), KKN_01 (all horizons), RYG_01 (12h, 24h, 48h, 72h)
- **Candidate for Deployment**: HYI_01 (all horizons), RYG_01 (6h)

### Key Performance Metrics

- **Best Overall**: Khon Kaen 6h (skill score: 0.18, accuracy: 90.8%)
- **Lowest MAE**: Nong Khai 24h (1.42°C)

- **Best Safety Performance**: Bangkok 24h (danger recall at 42°C: 93.6%)
- **Prediction Interval Coverage**: All models achieve 90.7-96.9% for 90% intervals

## Risk Categories

Heat index risk categories based on WHO/Thai-adapted thresholds:

- **Caution**: 27-32°C — Normal conditions, standard precautions
- **Extreme Caution**: 32-40°C — Increased risk, avoid prolonged exposure

- **Danger**: 40-54°C — High risk, avoid outdoor activities
- **Extreme Danger**: >54°C — Extreme risk, emergency conditions

## System Features

### Monitoring & Observability

- **Health Checks**: `/health` and `/health/detailed` endpoints
- **Error Detection & Response (EDR)**: Automatic error detection and circuit breaking

- **Circuit Breaker**: TMD API circuit breaker for resilience
- **Prediction Monitoring**: Track prediction latency, confidence, and data source

### Data Quality & Validation

- **Cross-source Validation**: Validate data across multiple sources (TMD, ERA5, NASA POWER)
- **Data Quality Checks**: Comprehensive data quality validation and cleaning

- **Bias Calibration**: Model bias calibration for improved accuracy

### Advanced Features

- **Conformal Prediction**: Uncertainty quantification with prediction intervals
- **Danger Gate**: Safety threshold classification for high-risk conditions

- **Risk Fusion**: Multi-source risk fusion for robust decision making
- **What-if Simulator**: Scenario simulation for planning and decision support

## Impact Metrics (ตาม proposal §9)

- **Forecast accuracy** — MAE/RMSE เทียบ baseline
- **Alert lead time** — แจ้งล่วงหน้า 24–72 ชม.

- **High-risk exposure reduction** — ลด 30–60% ใน scenario
- **Decision improvement** — คะแนน decision quality

- **Action card usability** — ≥ 4/5 จากกลุ่มทดลอง
- **False reassurance control** — uncertainty สูง → conservative

## Documentation

- **Architecture**: `docs/architecture.md`
- **Data Dictionary**: `docs/data_dictionary.md`

- **Colab Training**: `docs/colab-training.md`
- **Colab Quick Guide (Thai)**: `docs/COLAB_TRAINING_GUIDE.md`

- **Model Reports**: `comprehensive_model_report.pdf`, `model_evaluation_report.pdf`

## License & Reference

อ้างอิงตามเอกสารใน proposal §19: WHO, WMO, กรมควบคุมโรค, TMD, UNDRR
