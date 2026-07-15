"""
Unified Health Impact Prediction Pipeline.

Automatically detects model type (ML sklearn/XGBoost or DL PyTorch ANN)
and routes prediction through the correct path.

Usage:
    from src.predict_health_impact import predict_health_impact

    result = predict_health_impact(inputs_dict, model_type="best_ml")
    result = predict_health_impact(inputs_dict, model_type="ann")
"""

import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.config import (
    HEALTH_IMPACT_MODEL_PATH,
    HEALTH_IMPACT_SCALER_PATH,
    HEALTH_IMPACT_ENCODER_PATH,
    HEALTH_IMPACT_ANN_PATH,
    HEALTH_IMPACT_ANN_CONFIG_PATH,
    HEALTH_IMPACT_ANN_PREPROCESSOR_PATH,
)
from utils.model_utils import load_model, load_ann_model, predict_with_ann

logger = logging.getLogger(__name__)


# ─── Model Type Constants ─────────────────────────────────────────────────────
MODEL_TYPE_ML = "best_ml"       # Best sklearn/XGBoost model saved as .pkl
MODEL_TYPE_ANN = "ann"          # PyTorch ANN saved as .pt


class PredictionResult:
    """Structured result container for health impact predictions."""

    def __init__(
        self,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        class_names: List[str],
        model_type: str,
        model_name: str,
    ) -> None:
        self.prediction = prediction
        self.confidence = confidence
        self.probabilities = probabilities
        self.class_names = class_names
        self.model_type = model_type   # "ML" or "DL"
        self.model_name = model_name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to plain dict for JSON / session storage."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "class_names": self.class_names,
            "model_type": self.model_type,
            "model_name": self.model_name,
        }


# ─── ML Prediction Path ───────────────────────────────────────────────────────

def _load_ml_artifacts() -> Tuple[Any, Any, Any]:
    """
    Load the best ML model artifacts (model + scaler + encoder).

    Returns:
        Tuple of (model, scaler, label_encoder) — any may be None on failure.
    """
    model = load_model(HEALTH_IMPACT_MODEL_PATH, "health_impact_best_ml_model")
    scaler = load_model(HEALTH_IMPACT_SCALER_PATH, "health_impact_scaler")
    encoder = load_model(HEALTH_IMPACT_ENCODER_PATH, "health_impact_encoder")
    return model, scaler, encoder


def _predict_ml(inputs: Dict[str, float]) -> "PredictionResult":
    """
    Run prediction using the best sklearn/XGBoost ML model.

    Args:
        inputs: Feature dict mapping column names to values.

    Returns:
        PredictionResult with prediction, confidence and probabilities.

    Raises:
        RuntimeError: If any required artifact is missing.
    """
    model, scaler, encoder = _load_ml_artifacts()

    if model is None or scaler is None or encoder is None:
        raise RuntimeError(
            "ML model artifacts missing. "
            "Run: python src/health_impact_model.py"
        )

    X = pd.DataFrame([inputs])

    # Use the feature order the scaler was fitted on
    if hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)
    else:
        feature_cols = list(inputs.keys())

    X = X[feature_cols]
    X_scaled = scaler.transform(X)

    pred_enc = model.predict(X_scaled)[0]
    proba = (
        model.predict_proba(X_scaled)[0]
        if hasattr(model, "predict_proba") else None
    )
    pred_class = encoder.inverse_transform([pred_enc])[0]
    class_names = list(encoder.classes_)

    if proba is not None:
        confidence = float(np.max(proba))
        proba_dict = dict(zip(class_names, proba.tolist()))
    else:
        confidence = 1.0
        proba_dict = {pred_class: 1.0}

    model_name = type(model).__name__
    logger.info(f"ML prediction: {pred_class} ({confidence*100:.1f}%) via {model_name}")

    return PredictionResult(
        prediction=pred_class,
        confidence=confidence,
        probabilities=proba_dict,
        class_names=class_names,
        model_type="ML",
        model_name=model_name,
    )


# ─── ANN Prediction Path ──────────────────────────────────────────────────────

def _load_ann_artifacts() -> Tuple[Any, Any, Dict]:
    """
    Load ANN model, its dedicated preprocessor, and architecture config.

    Returns:
        Tuple of (ann_model, preprocessor, config_dict) — any may be None.
    """
    ann_model = load_ann_model(HEALTH_IMPACT_ANN_PATH, HEALTH_IMPACT_ANN_CONFIG_PATH)
    preprocessor = load_model(
        HEALTH_IMPACT_ANN_PREPROCESSOR_PATH, "health_impact_ann_preprocessor"
    )

    config: Dict = {}
    if HEALTH_IMPACT_ANN_CONFIG_PATH.exists():
        try:
            with open(HEALTH_IMPACT_ANN_CONFIG_PATH, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except Exception as exc:
            logger.warning(f"Could not read ANN config: {exc}")

    return ann_model, preprocessor, config


def _predict_ann(inputs: Dict[str, float]) -> "PredictionResult":
    """
    Run prediction using the PyTorch ANN model.

    Args:
        inputs: Feature dict mapping column names to values.

    Returns:
        PredictionResult with prediction, confidence and probabilities.

    Raises:
        RuntimeError: If any required artifact is missing.
    """
    ann_model, preprocessor, config = _load_ann_artifacts()

    if ann_model is None or preprocessor is None:
        raise RuntimeError(
            "ANN artifacts missing. "
            "Run: python src/health_impact_dl_model.py"
        )

    # Use feature order from preprocessor (fitted on training data)
    feature_names: List[str] = getattr(
        preprocessor, "feature_names", list(inputs.keys())
    )
    row = {f: inputs.get(f, 0.0) for f in feature_names}
    X = pd.DataFrame([row])
    X_scaled = preprocessor.transform(X)

    y_pred_arr, y_proba_arr = predict_with_ann(ann_model, X_scaled)

    if len(y_pred_arr) == 0:
        raise RuntimeError("ANN inference returned empty result.")

    pred_enc = int(y_pred_arr[0])
    proba = y_proba_arr[0]

    class_names: List[str] = config.get(
        "class_names", list(preprocessor.label_encoder.classes_)
    )
    pred_class = preprocessor.label_encoder.inverse_transform([pred_enc])[0]
    confidence = float(np.max(proba))
    proba_dict = dict(zip(class_names, proba.tolist()))

    logger.info(f"ANN prediction: {pred_class} ({confidence*100:.1f}%)")

    return PredictionResult(
        prediction=pred_class,
        confidence=confidence,
        probabilities=proba_dict,
        class_names=class_names,
        model_type="DL",
        model_name="ANN (PyTorch)",
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def predict_health_impact(
    inputs: Dict[str, float],
    model_type: str = MODEL_TYPE_ML,
) -> "PredictionResult":
    """
    Unified Health Impact prediction entry point.

    Automatically detects whether to use the best ML model or the ANN,
    loads the appropriate artifacts, and returns a structured result.

    Args:
        inputs: Dict of feature_name -> float value. Must include all
                features used during training (AQI, PM10, PM2_5, ...).
        model_type: One of ``"best_ml"`` (default) or ``"ann"``.

    Returns:
        PredictionResult with prediction, confidence, probabilities, etc.

    Raises:
        ValueError: For unknown model_type.
        RuntimeError: If required model files are missing.
    """
    if model_type == MODEL_TYPE_ML:
        return _predict_ml(inputs)
    elif model_type == MODEL_TYPE_ANN:
        return _predict_ann(inputs)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose '{MODEL_TYPE_ML}' or '{MODEL_TYPE_ANN}'."
        )


def get_ann_config() -> Dict:
    """
    Read and return the ANN architecture config JSON.

    Returns:
        Config dict with input_dim, num_classes, feature_names, class_names.
        Empty dict if config not found.
    """
    if not HEALTH_IMPACT_ANN_CONFIG_PATH.exists():
        return {}
    try:
        with open(HEALTH_IMPACT_ANN_CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def is_ann_ready() -> bool:
    """Return True if all ANN artifacts exist and can be loaded."""
    return (
        HEALTH_IMPACT_ANN_PATH.exists()
        and HEALTH_IMPACT_ANN_CONFIG_PATH.exists()
        and HEALTH_IMPACT_ANN_PREPROCESSOR_PATH.exists()
    )


def is_ml_ready() -> bool:
    """Return True if all ML artifacts exist."""
    return (
        HEALTH_IMPACT_MODEL_PATH.exists()
        and HEALTH_IMPACT_SCALER_PATH.exists()
        and HEALTH_IMPACT_ENCODER_PATH.exists()
    )


# ─── CLI quick test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = {
        "AQI": 200.0, "PM10": 120.0, "PM2_5": 85.0,
        "NO2": 80.0, "SO2": 35.0, "O3": 130.0,
        "Temperature": 22.0, "Humidity": 65.0, "WindSpeed": 4.5,
        "RespiratoryCases": 150.0, "CardiovascularCases": 75.0,
        "HospitalAdmissions": 30.0, "HealthImpactScore": 78.0,
    }

    print("\n=== ML Prediction ===")
    try:
        r = predict_health_impact(sample, model_type=MODEL_TYPE_ML)
        print(f"  Prediction : {r.prediction}")
        print(f"  Confidence : {r.confidence*100:.1f}%")
        print(f"  Model      : {r.model_name}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== ANN Prediction ===")
    try:
        r = predict_health_impact(sample, model_type=MODEL_TYPE_ANN)
        print(f"  Prediction : {r.prediction}")
        print(f"  Confidence : {r.confidence*100:.1f}%")
        print(f"  Model      : {r.model_name}")
    except Exception as e:
        print(f"  ERROR: {e}")
