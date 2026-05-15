"""Configuration validation and management for HeatShield AI."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config() -> None:
    """Validate all required environment variables and settings.
    
    Raises:
        ConfigValidationError: If required config is missing or invalid
    """
    errors = []
    
    # Validate TMD configuration
    tmd_api_key = os.environ.get("TMD_API_KEY")
    if not tmd_api_key or tmd_api_key == "your_tmd_api_key_here":
        errors.append("TMD_API_KEY is not set or still has placeholder value")
    
    tmd_base_url = os.environ.get("TMD_BASE_URL")
    if not tmd_base_url:
        errors.append("TMD_BASE_URL is not set")
    
    # Validate ERA5 configuration (optional but warn if missing)
    cdsapi_key = os.environ.get("CDSAPI_KEY")
    if not cdsapi_key:
        logger.warning("CDSAPI_KEY is not set - ERA5 ingestion will not work")
    
    # Validate app environment
    env = os.environ.get("HEATSHIELD_ENV", "dev")
    if env not in ("dev", "staging", "production"):
        errors.append(f"HEATSHIELD_ENV must be one of: dev, staging, production (got: {env})")
    
    # Validate log level
    log_level = os.environ.get("HEATSHIELD_LOG_LEVEL", "INFO")
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if log_level not in valid_levels:
        errors.append(f"HEATSHIELD_LOG_LEVEL must be one of: {valid_levels} (got: {log_level})")
    
    # Validate GPU settings
    force_cpu = os.environ.get("HEATSHIELD_FORCE_CPU", "0")
    if force_cpu not in ("0", "1"):
        errors.append(f"HEATSHIELD_FORCE_CPU must be 0 or 1 (got: {force_cpu})")
    
    lgbm_device = os.environ.get("LGBM_DEVICE", "").lower()
    if lgbm_device and lgbm_device not in ("", "cpu", "gpu", "cuda"):
        errors.append(f"LGBM_DEVICE must be one of: cpu, gpu, cuda (got: {lgbm_device})")
    
    # Validate data directory exists
    data_dir = Path("data/raw")
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
    
    # Validate models directory exists
    models_dir = Path("app/models")
    if not models_dir.exists():
        errors.append(f"Models directory does not exist: {models_dir}")
    
    if errors:
        raise ConfigValidationError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    logger.info("Configuration validation passed")


def get_config() -> dict:
    """Get validated configuration as a dictionary.
    
    Returns:
        Dictionary with all configuration values
    """
    return {
        "tmd_api_key": os.environ.get("TMD_API_KEY", ""),
        "tmd_base_url": os.environ.get("TMD_BASE_URL", ""),
        "cdsapi_key": os.environ.get("CDSAPI_KEY", ""),
        "cdsapi_url": os.environ.get("CDSAPI_URL", ""),
        "env": os.environ.get("HEATSHIELD_ENV", "dev"),
        "log_level": os.environ.get("HEATSHIELD_LOG_LEVEL", "INFO"),
        "force_cpu": os.environ.get("HEATSHIELD_FORCE_CPU", "0") == "1",
        "lgbm_device": os.environ.get("LGBM_DEVICE", ""),
    }
