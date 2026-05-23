"""Feature selection for time series forecasting models.

Implements SHAP-based feature selection and Boruta algorithm to identify
truly important features and reduce overfitting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False


def shap_feature_selection(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    threshold: float = 0.01,
    background_samples: int = 100,
) -> List[str]:
    """Select features based on SHAP importance values.

    Args:
        model: Trained model with predict method
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        threshold: Minimum mean absolute SHAP value threshold
        background_samples: Number of background samples for SHAP

    Returns:
        List of selected feature names

    Raises:
        ImportError: If shap package is not installed
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "shap package is required for SHAP feature selection. "
            "Install with: pip install shap"
        )

    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train_np = X_train.values
        X_val_np = X_val.values
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_np = X_train
        X_val_np = X_val

    # Create SHAP explainer
    try:
        # Try TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fall back to KernelExplainer for other models
        background_idx = np.random.choice(
            len(X_train_np), min(background_samples, len(X_train_np)), replace=False
        )
        background = X_train_np[background_idx]
        explainer = shap.KernelExplainer(model.predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_val_np)

    # Calculate mean absolute SHAP value per feature
    if isinstance(shap_values, list):
        # For multi-output models, use first output
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Select features above threshold
    selected_mask = mean_abs_shap > threshold
    selected_features = [
        feature_names[i] for i, selected in enumerate(selected_mask) if selected
    ]

    return selected_features


def boruta_feature_selection(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_estimators: int = 100,
    max_iter: int = 100,
    random_state: int = 42,
) -> List[str]:
    """Select features using Boruta algorithm.

    Boruta compares feature importance with shadow features to identify
    truly important features.

    Args:
        model: Model with fit/predict methods (e.g., RandomForest)
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of trees for Boruta
        max_iter: Maximum iterations for Boruta
        random_state: Random seed

    Returns:
        List of selected feature names

    Raises:
        ImportError: If boruta package is not installed
    """
    if not BORUTA_AVAILABLE:
        raise ImportError(
            "boruta package is required for Boruta feature selection. "
            "Install with: pip install boruta"
        )

    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train_np = X_train.values
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_np = X_train

    # Convert y to numpy
    if isinstance(y_train, pd.Series):
        y_train_np = y_train.values
    else:
        y_train_np = y_train

    # Initialize Boruta
    boruta = BorutaPy(
        model,
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )

    # Fit Boruta
    boruta.fit(X_train_np, y_train_np)

    # Get selected features
    selected_features = [feature_names[i] for i, support in enumerate(boruta.support_) if support]

    return selected_features


def recursive_feature_elimination(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    n_features_to_select: Optional[int] = None,
    step: int = 1,
) -> List[str]:
    """Recursive feature elimination based on model performance.

    Args:
        model: Model with fit/predict methods
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_features_to_select: Number of features to select (default: sqrt(n_features))
        step: Number of features to remove per iteration

    Returns:
        List of selected feature names
    """
    from sklearn.metrics import mean_absolute_error

    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train_df = X_train.copy()
        X_val_df = X_val.copy()
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_val_df = pd.DataFrame(X_val, columns=feature_names)

    if n_features_to_select is None:
        n_features_to_select = int(np.sqrt(len(feature_names)))

    current_features = feature_names.copy()
    best_mae = float("inf")

    while len(current_features) > n_features_to_select:
        # Train model with current features
        model.fit(X_train_df[current_features], y_train)
        y_pred = model.predict(X_val_df[current_features])
        mae = mean_absolute_error(y_val, y_pred)

        # Track best performance
        if mae < best_mae:
            best_mae = mae
            best_features = current_features.copy()

        # Remove least important features
        # (simplified: remove last feature, could use feature importance instead)
        features_to_remove = min(step, len(current_features) - n_features_to_select)
        current_features = current_features[:-features_to_remove]

    return best_features


def correlation_filter(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> List[str]:
    """Remove highly correlated features to reduce redundancy.

    Args:
        X: Feature DataFrame
        threshold: Correlation threshold for removal

    Returns:
        List of selected feature names
    """
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()

    # Find highly correlated feature pairs
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Identify features to remove
    to_drop = [
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > threshold)
    ]

    # Keep features not in to_drop
    selected_features = [col for col in X.columns if col not in to_drop]

    return selected_features


def variance_filter(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> List[str]:
    """Remove features with very low variance.

    Args:
        X: Feature DataFrame
        threshold: Minimum variance threshold

    Returns:
        List of selected feature names
    """
    # Calculate variance for each feature
    variances = X.var()

    # Select features above threshold
    selected_features = variances[variances > threshold].index.tolist()

    return selected_features


def combined_feature_selection(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    method: str = "shap",
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    shap_threshold: float = 0.01,
) -> List[str]:
    """Combined feature selection pipeline.

    Applies variance filter, correlation filter, and model-based selection.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        method: Model-based selection method ("shap", "boruta", "rfe")
        variance_threshold: Minimum variance threshold
        correlation_threshold: Maximum correlation threshold
        shap_threshold: SHAP importance threshold

    Returns:
        List of selected feature names
    """
    # Step 1: Variance filter
    selected = variance_filter(X_train, variance_threshold)
    X_train_filtered = X_train[selected]
    X_val_filtered = X_val[selected]

    # Step 2: Correlation filter
    selected = correlation_filter(X_train_filtered, correlation_threshold)
    X_train_filtered = X_train_filtered[selected]
    X_val_filtered = X_val_filtered[selected]

    # Step 3: Model-based selection
    if method == "shap":
        selected = shap_feature_selection(
            model, X_train_filtered, y_train, X_val_filtered, y_val, shap_threshold
        )
    elif method == "boruta":
        selected = boruta_feature_selection(model, X_train_filtered, y_train)
    elif method == "rfe":
        selected = recursive_feature_elimination(
            model, X_train_filtered, y_train, X_val_filtered, y_val
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return selected
