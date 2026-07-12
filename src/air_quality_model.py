"""
Air Quality Prediction Model Training Script.

Algorithm: XGBoost Regressor (primary), Random Forest Regressor (fallback)
Target: CO(GT) concentration (time-series regression)
Split: Time-based 80/20 (no shuffle)
Output: models/air_quality/{prediction_model,scaler}.pkl
"""

import sys
import os
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_loader import load_air_quality_dataset
from src.preprocessing import AirQualityPreprocessor
from utils.config import (
    AIR_QUALITY_TARGET, AIR_QUALITY_TARGET_FALLBACK,
    AIR_QUALITY_MODEL_PATH, AIR_QUALITY_SCALER_PATH,
    RANDOM_STATE, LOGS_DIR,
)
from utils.model_utils import save_model, evaluate_regressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def select_target(df: pd.DataFrame) -> str:
    """
    Select prediction target based on data quality.
    Prefer CO(GT); fall back to NO2(GT) if CO has too many missing values.
    """
    for col in [AIR_QUALITY_TARGET, AIR_QUALITY_TARGET_FALLBACK]:
        if col in df.columns:
            valid_pct = df[col].notna().mean()
            logger.info(f"Target '{col}' valid data: {valid_pct*100:.1f}%")
            if valid_pct >= 0.7:
                return col
    # Fallback: first numeric column that isn't Date/Time
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError("No suitable target column found in air quality dataset.")


def train_air_quality_model():
    """Train and save the air quality regression model."""
    logger.info("=" * 60)
    logger.info("AIR QUALITY PREDICTION MODEL TRAINING")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    logger.info("Loading UCI Air Quality dataset...")
    df = load_air_quality_dataset()
    logger.info(f"Dataset shape: {df.shape}")
    if "DateTime" in df.columns:
        logger.info(f"Date range: {df['DateTime'].min()} → {df['DateTime'].max()}")

    # ── 2. Select target ──────────────────────────────────────────
    target_col = select_target(df)
    logger.info(f"Using target column: {target_col}")

    # ── 3. Feature engineering ────────────────────────────────────
    preprocessor = AirQualityPreprocessor(target_col=target_col)
    X_scaled, y, X_df = preprocessor.fit_transform_df(df)
    logger.info(f"After feature engineering: {X_scaled.shape} features, {len(y)} samples")

    # ── 4. Time-based train/test split (80/20, no shuffle) ───────
    split_idx = int(len(y) * 0.80)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 5. Train XGBoost Regressor ────────────────────────────────
    logger.info("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = evaluate_regressor(y_test, xgb_pred)
    logger.info(f"XGBoost → RMSE: {xgb_metrics['rmse']:.4f} | MAE: {xgb_metrics['mae']:.4f} | R²: {xgb_metrics['r2']:.4f}")

    # ── 6. Train Random Forest Regressor (comparison) ─────────────
    logger.info("Training Random Forest Regressor (comparison)...")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_split=5,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_regressor(y_test, rf_pred)
    logger.info(f"RandomForest → RMSE: {rf_metrics['rmse']:.4f} | MAE: {rf_metrics['mae']:.4f} | R²: {rf_metrics['r2']:.4f}")

    # ── 7. Select best model by R² ────────────────────────────────
    if xgb_metrics["r2"] >= rf_metrics["r2"]:
        best_model = xgb_model
        best_metrics = xgb_metrics
        best_name = "XGBoost"
    else:
        best_model = rf_model
        best_metrics = rf_metrics
        best_name = "Random Forest"
    logger.info(f"\n🏆 Best model: {best_name} (R²: {best_metrics['r2']:.4f})")

    # ── 8. Save artifacts ─────────────────────────────────────────
    save_model(best_model, AIR_QUALITY_MODEL_PATH, f"Air quality model ({best_name})")
    save_model(preprocessor.scaler, AIR_QUALITY_SCALER_PATH, "Air quality scaler")

    # Also save metadata for inference
    metadata = {
        "target_col": target_col,
        "feature_names": preprocessor.feature_names,
        "best_model_name": best_name,
        "metrics": best_metrics,
    }
    meta_path = AIR_QUALITY_MODEL_PATH.parent / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── 9. Save metrics log ───────────────────────────────────────
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "target_col": target_col,
        "best_model": best_name,
        "xgb_metrics": xgb_metrics,
        "rf_metrics": rf_metrics,
        "best_metrics": best_metrics,
        "n_features": len(preprocessor.feature_names),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "air_quality_model_metrics.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Metrics saved → {log_path}")

    logger.info("\n✅ Air quality model training complete!")
    return best_model, preprocessor, best_metrics, target_col


if __name__ == "__main__":
    train_air_quality_model()
