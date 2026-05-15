# Repository Guidelines

## Project Structure & Module Organization

This is a Python FastAPI backend for HeatShield AI, a heat index forecasting and risk assessment system for Thailand. The project provides real-time heat index predictions with uncertainty quantification and risk category classification to support public health decision-making.

### Directory Structure

- **`app/`**: Core application code
  - **`app/api/`**: FastAPI route modules for HTTP endpoints
    - `forecast.py`: Heat index forecasting endpoints
    - `risk.py`: Risk assessment and classification endpoints
    - `stations.py`: Station metadata and configuration endpoints
  - **`app/core/`**: Domain logic and business rules
    - `heat_index.py`: Heat index calculation using Rothfusz equation
    - `risk_scoring.py`: Risk category classification and threshold logic
    - `what_if.py`: What-if scenario simulation for planning
  - **`app/data/`**: Data access layer
    - `clients/`: External API clients (TMD, ERA5, NASA POWER, MODIS)
    - `schemas/`: Pydantic data models for API requests/responses
    - `loaders/`: Data ingestion and preprocessing utilities
    - `quality.py`: Data quality validation and cleaning
  - **`app/ml/`**: Machine learning pipeline
    - `forecast/`: Heat index forecasting models
      - `backends/`: Model backend implementations (LightGBM Quantile)
      - `training.py`: Model training pipeline
      - `evaluation.py`: Model evaluation and metrics
      - `prediction.py`: Model inference and prediction
      - `registry.py`: Model versioning and metadata management
  - **`app/models/`**: Trained model artifacts
    - `forecast_v3/`: Current model version (LightGBM Quantile)
      - `<station_id>/h<horizon>/`: Per-station, per-horizon models
        - `model.txt`: Trained LightGBM model
        - `registry.json`: Model metadata and evaluation results
        - `calibration.json`: Bias calibration parameters
- **`scripts/`**: Operational scripts
  - `ingest_all.py`: Data ingestion for all stations
  - `train_forecast.py`: Model training pipeline
  - `evaluate_model.py`: Model evaluation and reporting
  - `seed_demo_data.py`: Demo data seeding for development
- **`tests/`**: Test suite
  - `test_*.py`: Unit and integration tests
- **`docs/`**: Documentation
- **`data/`**: Raw and cached datasets
  - `raw/`: Original downloaded data
  - `cache/`: Processed and cached data
- **`logs/`**: Generated metrics, plots, and logs

### Module Organization Principles

- **Route modules** (`app/api/`): Handle HTTP requests/responses only. Delegate business logic to `app/core/` or `app/ml/`.
- **Domain logic** (`app/core/`): Pure functions implementing business rules. No I/O or database operations.
- **ML logic** (`app/ml/`): Training, evaluation, and prediction logic. Keep backend-agnostic where possible.
- **Data access** (`app/data/`): External API interactions, data loading, and quality checks.

## Model Registry & Evaluation

### Model Architecture

The project uses LightGBM Quantile Regression for probabilistic heat index forecasting:

- **Backend**: LightGBM Quantile Regression with three quantile heads (5th, 50th, 95th percentiles)
- **Target**: Two-head (temperature_c, relative_humidity) at forecast horizon t+h
- **Composition**: Heat index computed via Rothfusz equation at inference time
- **Features**: 108 engineered features including:
  - Temporal features: hour of day, day of year, month
  - Geographic features: station latitude, longitude, elevation
  - Meteorological features: temperature, humidity, wind speed, solar radiation
  - Lag features: Historical values at 1h, 3h, 6h, 12h, 24h lags
  - Rolling statistics: Mean, std, min, max over 3h, 6h, 12h, 24h windows

### Trained Models

The project currently has 15 LightGBM quantile regression models trained for heat index forecasting across 5 Thai meteorological stations:

- **Stations**:
  - Bangkok (BKK_01): Central Thailand — 13.76°N 100.50°E
  - Chiang Mai (CNX_01): Northern Thailand — 18.79°N 98.98°E
  - Khon Kaen (KKN_01): Northeast Thailand — 16.44°N 102.84°E
  - Nong Khai (HYI_01): Northeast Thailand — 17.88°N 102.74°E
  - Rayong (RYG_01): Eastern Thailand — 12.68°N 101.27°E
- **Forecast Horizons**: 6 hours, 12 hours, 24 hours
- **Model Location**: `app/models/forecast_v3/<station_id>/h<horizon>/`
- **Training Period**: 2021-2025 data from TMD observations and ERA5 reanalysis
- **Data Split**: 80% training, 10% validation, 10% test (chronological)

### Model Status

- **Ready for Deployment**: BKK_01 (all horizons), CNX_01 (all horizons), KKN_01 (all horizons), RYG_01 (12h, 24h)
- **Candidate for Deployment**: HYI_01 (all horizons), RYG_01 (6h)

### Key Performance Metrics

- **Best Overall**: Khon Kaen 6h (skill score: 0.18, accuracy: 90.8%)
- **Lowest MAE**: Nong Khai 24h (1.42°C)
- **Best Safety Performance**: Bangkok 24h (danger recall at 42°C: 93.6%)
- **Prediction Interval Coverage**: All models achieve 90.7-96.9% for 90% intervals

### Evaluation Metrics

- **Regression Metrics**: MAE, RMSE, Bias, Correlation
- **Baseline Comparisons**: Persistence, Climatology, Skill Score
- **Classification Metrics**: Precision, Recall, F1 for risk categories
- **Safety Metrics**: Danger recall at 40°C and 42°C thresholds
- **Prediction Interval Metrics**: Coverage, Mean Width, Pinball Loss
- **Runtime Metrics**: Training time, inference time
- **Data Split Statistics**: Row counts, date ranges per split

### Evaluation Reports

- **Comprehensive model report**: `comprehensive_model_report.pdf` (19 pages)
  - Executive summary with key findings
  - Model architecture and feature engineering details
  - Regression metrics with visualizations
  - Baseline comparisons
  - Classification metrics
  - Safety metrics
  - Prediction interval metrics
  - Runtime and data statistics
  - Summary tables
  - Model status and metadata
- **Data visualizations**: `model_visualizations.pdf`

### Generating Model Reports

To generate a comprehensive LaTeX PDF report of model evaluation results:

1. Ensure LaTeX (pdflatex) is installed on your system
2. The report is generated from model registry data in `app/models/forecast_v3/*/h*/registry.json`
3. The comprehensive report includes:
   - Executive summary with key findings
   - Model architecture and feature engineering details
   - Regression metrics (MAE, RMSE, Bias, Correlation) with visualizations
   - Baseline comparisons (Persistence, Climatology, Skill Score)
   - Classification metrics for risk categories
   - Safety metrics (Danger recall at 40°C and 42°C)
   - Prediction interval metrics (Coverage, Width, Pinball loss)
   - Runtime and data split statistics
   - Summary tables for all metrics
   - Model status and metadata

The LaTeX document uses PGFPlots/TikZ for all visualizations, ensuring self-contained report generation without external plotting dependencies.

### Risk Categories

Heat index risk categories based on WHO/Thai-adapted thresholds:

- **Caution**: 27-32°C — Normal conditions, standard precautions
- **Extreme Caution**: 32-40°C — Increased risk, avoid prolonged exposure
- **Danger**: 40-54°C — High risk, avoid outdoor activities
- **Extreme Danger**: >54°C — Extreme risk, emergency conditions

## Build, Test, and Development Commands

### Environment Setup

Set up a local development environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running the API

Start the FastAPI development server:

```powershell
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Running Tests

Run the default test suite (excludes slow tests):

```powershell
pytest
```

Run slow, data/model-dependent tests:

```powershell
pytest -m slow
```

Run specific test files:

```powershell
pytest tests/test_forecast.py
```

Run with coverage:

```powershell
pytest --cov=app --cov-report=html
```

### Common Scripts

- **Data ingestion**: `python scripts/ingest_all.py`
  - Fetches data from TMD, ERA5, NASA POWER, MODIS
  - Processes and caches data for all stations
  - Updates data quality reports

- **Model training**: `python scripts/train_forecast.py`
  - Trains LightGBM quantile models for all stations and horizons
  - Performs hyperparameter optimization with Optuna
  - Saves trained models to `app/models/forecast_v3/`

- **Model evaluation**: `python scripts/evaluate_model.py`
  - Evaluates trained models on test set
  - Computes regression, classification, and safety metrics
  - Updates model registry files

- **Demo data seeding**: `python scripts/seed_demo_data.py`
  - Seeds database with sample data for development
  - Creates demo stations and forecasts

## Coding Style & Naming Conventions

### Python Version

Use Python 3.10+ for type hints and modern language features.

### Code Style

- Follow PEP 8 with 4-space indentation
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for all public functions and classes

### Naming Conventions

- **Modules and functions**: `snake_case`
  - Example: `calculate_heat_index`, `train_model`
- **Classes**: `PascalCase`
  - Example: `HeatIndexCalculator`, `ModelTrainer`
- **Constants**: `UPPER_SNAKE_CASE`
  - Example: `MAX_FORECAST_HORIZON`, `DEFAULT_STATION`
- **Private functions**: `_snake_case`
  - Example: `_internal_helper`, `_validate_input`

### Architecture Patterns

- **Route modules**: Keep focused on HTTP request/response handling. Delegate business logic to `app/core/` or `app/ml/`.
- **Domain logic**: Implement as small pure functions in `app/core/` for reusability and testability.
- **API schemas**: Use Pydantic models for request/response validation in `app/data/schemas/`.
- **ML logic**: Place training, evaluation, and prediction logic in `app/ml/` with backend-agnostic interfaces.

### Example Code Structure

```python
# app/core/heat_index.py
def calculate_heat_index(temperature_c: float, relative_humidity: float) -> float:
    """Calculate heat index using Rothfusz equation.
    
    Args:
        temperature_c: Temperature in Celsius
        relative_humidity: Relative humidity as percentage (0-100)
        
    Returns:
        Heat index in Celsius
    """
    # Implementation
    pass

# app/api/forecast.py
from app.core.heat_index import calculate_heat_index

@router.post("/forecast")
async def create_forecast(request: ForecastRequest) -> ForecastResponse:
    # Route module handles HTTP, delegates to core/ml
    heat_index = calculate_heat_index(request.temp, request.humidity)
    return ForecastResponse(heat_index=heat_index)
```

## Testing Guidelines

### Test Framework

Tests use `pytest` and `pytest-asyncio` for synchronous and asynchronous testing.

### Test Organization

- Place test files beside the modules they test: `tests/test_<feature>.py`
- Name test functions descriptively: `test_<behavior>`
- Group related tests in test classes when appropriate

### Test Types

- **Unit tests**: Test individual functions and classes in isolation
  - Cover deterministic domain logic directly
  - Mock external dependencies (API clients, databases)
  - Fast execution, no external dependencies

- **Integration tests**: Test module interactions
  - Test API endpoints with test database
  - Test data pipelines with sample data
  - May use test fixtures for setup

- **Slow tests**: Tests requiring real data or trained models
  - Mark with `@pytest.mark.slow`
  - Require ingested data or trained models
  - Run separately with `pytest -m slow`

### Example Test Structure

```python
# tests/test_heat_index.py
import pytest
from app.core.heat_index import calculate_heat_index

class TestHeatIndexCalculation:
    def test_basic_calculation(self):
        result = calculate_heat_index(30.0, 70.0)
        assert result > 30.0  # Heat index should be higher than temp
        
    def test_extreme_conditions(self):
        result = calculate_heat_index(40.0, 90.0)
        assert result > 50.0  # Should be in danger range
        
@pytest.mark.slow
def test_with_real_data():
    # Test with actual ingested data
    pass
```

### Mocking External Services

Mock external weather/data services for unit tests:

```python
from unittest.mock import Mock, patch
from app.data.clients.tmd import TMDClient

@patch('app.data.clients.tmd.TMDClient.fetch_observation')
def test_forecast_with_mock_tmd(mock_fetch):
    mock_fetch.return_value = {"temp": 30.0, "humidity": 70.0}
    # Test logic
```

## Commit & Pull Request Guidelines

### Commit Messages

This workspace does not include Git history, so follow a simple imperative convention:

- Use present tense: "Add feature" not "Added feature"
- Keep commits scoped to one concern
- Examples:
  - "Add forecast regression test"
  - "Fix TMD retry handling"
  - "Update model evaluation metrics"
  - "Refactor heat index calculation"

### Pull Request Description

Pull requests should include:

1. **Change description**: Brief summary of what changed and why
2. **Validation commands**: List of commands run to validate the change
   - `pytest` for tests
   - `pytest -m slow` for data/model-dependent tests
   - `python scripts/train_forecast.py` if models were retrained
3. **Data/model artifact changes**: Call out any changes to:
   - Trained models in `app/models/`
   - Data files in `data/`
   - Evaluation reports in `logs/`
4. **Related issues**: Link to related issues or requirements
5. **Screenshots/metrics**: Include screenshots or metric plots when modifying generated visual outputs

### Example PR Template

```
## Summary
Add new feature for X to improve Y

## Changes
- Modified file A to do X
- Updated file B to support X

## Validation
- `pytest` - all tests pass
- `pytest -m slow` - slow tests pass
- Manual testing confirmed X works as expected

## Artifacts
- Retrained models for stations X, Y, Z
- Updated evaluation metrics in logs/

## Related
- Issue #123
```

## Security & Configuration Tips

### Environment Variables

- Do not commit `.env` files containing secrets
- Use `.env.example` for configuration documentation
- Required environment variables:
  - `DATABASE_URL`: PostgreSQL connection string
  - `TMD_API_KEY`: Thai Meteorological Department API key
  - `ERA5_API_KEY`: ECMWF CDS API key
  - `NASA_POWER_API_KEY`: NASA POWER API key

### API Keys

- Never commit API keys to the repository
- Use environment variables or secret management
- Rotate keys if compromised
- Document required keys in `.env.example`

### Data Artifacts

Treat these paths as artifact-heavy (update only when necessary):

- **`data/`**: Raw and cached datasets
  - Update only when ingesting new data
  - Use `.gitignore` to exclude large files
  - Document data sources and update procedures

- **`logs/`**: Generated metrics and plots
  - Update only when regenerating reports
  - Include generation scripts for reproducibility
  - Use versioned filenames for historical tracking

- **`app/models/`**: Trained model artifacts
  - Update only when retraining models
  - Include model metadata in registry files
  - Document training data and hyperparameters

### File Size Guidelines

- Keep `.git` repository size manageable
- Use Git LFS for large binary files if needed
- Compress large text files before committing
- Consider external storage for datasets >100MB

## Development Workflow

### Feature Development

1. Create a new branch from main
2. Implement changes following coding guidelines
3. Write tests for new functionality
4. Run tests: `pytest` and `pytest -m slow`
5. Update documentation as needed
6. Submit pull request with validation commands

### Bug Fixing

1. Create a new branch from main
2. Reproduce the bug with a test if possible
3. Fix the issue
4. Verify the fix with tests
5. Run full test suite
6. Submit pull request with bug description

### Model Updates

1. Retrain models: `python scripts/train_forecast.py`
2. Evaluate models: `python scripts/evaluate_model.py`
3. Generate reports: Create LaTeX PDF from registry data
4. Verify model performance meets thresholds
5. Update model artifacts in `app/models/`
6. Update documentation with new metrics
7. Submit pull request with validation results

## Troubleshooting

### Common Issues

**Data ingestion fails**
- Check API keys are valid and not expired
- Verify internet connectivity
- Check rate limits on external APIs
- Review logs in `logs/ingestion/`

**Model training fails**
- Verify data is ingested and cached
- Check for sufficient memory/disk space
- Review hyperparameter configuration
- Check Optuna logs for optimization errors

**Tests fail**
- Ensure virtual environment is activated
- Verify all dependencies are installed
- Check for test data availability
- Review test logs for specific errors

**API errors**
- Check FastAPI server is running
- Verify database connection
- Check environment variables are set
- Review API logs for specific errors

### Getting Help

- Check existing issues in the repository
- Review documentation in `docs/`
- Consult AGENTS.md for project guidelines
- Contact maintainers for persistent issues
