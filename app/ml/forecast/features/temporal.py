"""Enhanced temporal embeddings for time series forecasting.

Implements multi-frequency sinusoidal embeddings and learnable temporal features
based on recent research (iTransformer, PatchTST, Time2Vec).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def enhanced_temporal_embeddings(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Create enhanced multi-frequency temporal embeddings.

    Adds sinusoidal embeddings at multiple frequencies to capture complex
    seasonal patterns (daily, weekly, monthly, annual) beyond basic sin/cos.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with enhanced temporal features added
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        timestamps = df[timestamp_col]
    else:
        # Assume index is timestamp
        timestamps = pd.to_datetime(df.index)
    
    # Extract time components
    hour = timestamps.dt.hour
    day_of_year = timestamps.dt.dayofyear
    month = timestamps.dt.month
    day_of_week = timestamps.dt.dayofweek
    
    # Base frequency embeddings (existing)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    # Enhanced: Higher harmonics for complex diurnal patterns
    df["hour_sin_2"] = np.sin(4 * np.pi * hour / 24)  # 2nd harmonic
    df["hour_cos_2"] = np.cos(4 * np.pi * hour / 24)
    df["hour_sin_3"] = np.sin(6 * np.pi * hour / 24)  # 3rd harmonic
    df["hour_cos_3"] = np.cos(6 * np.pi * hour / 24)
    
    # Enhanced: Multi-frequency day-of-year embeddings
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    df["doy_sin_2"] = np.sin(4 * np.pi * day_of_year / 365)  # 2nd harmonic
    df["doy_cos_2"] = np.cos(4 * np.pi * day_of_year / 365)
    
    # Enhanced: Day of week patterns (weekly seasonality)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Enhanced: Time of day bucket (categorical encoding)
    df["time_bucket"] = pd.cut(
        hour,
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        include_lowest=True,
    )
    
    return df


def time2vec_embedding(
    timestamps: pd.Series,
    embedding_dim: int = 16,
) -> np.ndarray:
    """Time2Vec-style learnable temporal embedding.

    Combines linear and periodic components for flexible temporal representation.
    In practice, this would be implemented as a neural network layer during training.
    Here we provide a fixed version using multiple frequencies.

    Args:
        timestamps: Series of timestamps
        embedding_dim: Dimension of embedding vector

    Returns:
        Array of shape (n_samples, embedding_dim)
    """
    timestamps = pd.to_datetime(timestamps)
    hour = timestamps.dt.hour
    day_of_year = timestamps.dt.dayofyear
    
    embeddings = []
    
    # Linear component (time progression)
    normalized_hour = hour / 24.0
    normalized_doy = day_of_year / 365.0
    embeddings.append(normalized_hour)
    embeddings.append(normalized_doy)
    
    # Periodic components with different frequencies
    remaining_dim = embedding_dim - 2
    for i in range(remaining_dim // 2):
        freq = (i + 1) * 2
        embeddings.append(np.sin(freq * 2 * np.pi * normalized_hour))
        embeddings.append(np.cos(freq * 2 * np.pi * normalized_hour))
    
    return np.column_stack(embeddings)


def add_seasonal_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add seasonal features for meteorological forecasting.

    Thai seasons:
    - Hot: March-May (3-5)
    - Rainy: June-October (6-10)
    - Cool: November-February (11-2)

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with seasonal features added
    """
    df = df.copy()
    
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        month = df[timestamp_col].dt.month
    else:
        month = pd.to_datetime(df.index).month
    
    # Thai season encoding
    def get_thai_season(m: int) -> str:
        if 3 <= m <= 5:
            return "hot"
        elif 6 <= m <= 10:
            return "rainy"
        else:
            return "cool"
    
    df["season"] = month.apply(get_thai_season)
    
    # One-hot encode seasons
    season_dummies = pd.get_dummies(df["season"], prefix="season")
    df = pd.concat([df, season_dummies], axis=1)
    
    # Month cyclical encoding (already in main features, but enhanced here)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    return df


def add_hour_of_day_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add detailed hour-of-day features.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with hour features added
    """
    df = df.copy()
    
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        hour = df[timestamp_col].dt.hour
    else:
        hour = pd.to_datetime(df.index).hour
    
    # Hour bucket (categorical)
    df["hour_bucket"] = pd.cut(
        hour,
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        include_lowest=True,
    )
    
    # Is daytime (6:00-18:00)
    df["is_daytime"] = ((hour >= 6) & (hour < 18)).astype(int)
    
    # Is peak heat time (11:00-15:00)
    df["is_peak_heat"] = ((hour >= 11) & (hour < 15)).astype(int)
    
    return df
