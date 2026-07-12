"""
Model I/O utilities with error handling for loading and saving ML models.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


def save_model(obj: Any, path: Path, description: str = "model") -> bool:
    """
    Safely save a model or preprocessor artifact using joblib.

    Args:
        obj: Object to serialize (model, scaler, encoder, etc.).
        path: Destination file path.
        description: Human-readable name for logging.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        logger.info(f"Saved {description} → {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {description}: {e}")
        return False


def load_model(path: Path, description: str = "model") -> Optional[Any]:
    """
    Safely load a model or preprocessor artifact.

    Args:
        path: File path to load from.
        description: Human-readable name for logging.

    Returns:
        Loaded object, or None if loading fails.
    """
    if not path.exists():
        logger.warning(f"{description} not found at {path}")
        return None
    try:
        obj = joblib.load(path)
        logger.info(f"Loaded {description} ← {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load {description}: {e}")
        return None


def safe_predict(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Run model.predict() with error catching.

    Args:
        model: Trained sklearn-compatible model.
        X: Feature array.

    Returns:
        Predictions array or None if prediction fails.
    """
    try:
        return model.predict(X)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def safe_predict_proba(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Run model.predict_proba() with error catching.

    Args:
        model: Trained sklearn-compatible model.
        X: Feature array.

    Returns:
        Probability array or None if not supported/fails.
    """
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        logger.warning("Model does not support predict_proba")
        return None
    except Exception as e:
        logger.error(f"predict_proba failed: {e}")
        return None


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Prediction probabilities (for ROC-AUC).
        class_names: Class label names.

    Returns:
        Dict of metrics: accuracy, f1, precision, recall, roc_auc.
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }

    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
            metrics["roc_auc"] = round(roc, 4)
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")

    return metrics


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dict with RMSE, MAE, R², MAPE.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE only when target values are non-zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else None

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 2) if mape is not None else None,
    }


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.

    Returns:
        DataFrame sorted by importance descending.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def check_models_exist() -> Dict[str, bool]:
    """
    Check whether all required model files exist.

    Returns:
        Dict mapping model name to boolean (True = file exists).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        CITY_TYPE_MODEL_PATH, CITY_TYPE_SCALER_PATH, CITY_TYPE_ENCODER_PATH,
        HEALTH_IMPACT_MODEL_PATH, HEALTH_IMPACT_SCALER_PATH,
        AIR_QUALITY_MODEL_PATH, AIR_QUALITY_SCALER_PATH,
    )

    return {
        "city_type_model": CITY_TYPE_MODEL_PATH.exists(),
        "city_type_scaler": CITY_TYPE_SCALER_PATH.exists(),
        "city_type_encoder": CITY_TYPE_ENCODER_PATH.exists(),
        "health_impact_model": HEALTH_IMPACT_MODEL_PATH.exists(),
        "health_impact_scaler": HEALTH_IMPACT_SCALER_PATH.exists(),
        "air_quality_model": AIR_QUALITY_MODEL_PATH.exists(),
        "air_quality_scaler": AIR_QUALITY_SCALER_PATH.exists(),
    }
