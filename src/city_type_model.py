"""
City Type Classification Model Training Script.

Algorithm: Random Forest with RandomizedSearchCV
Target: Industrial vs Residential (98%+ accuracy)
Output: models/city_type/{random_forest_model,scaler,label_encoder}.pkl
"""

import sys
import os
import logging
import json
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
)

from src.data_loader import load_city_types_dataset
from src.preprocessing import CityTypePreprocessor
from utils.config import (
    CITY_TYPE_FEATURES, CITY_TYPE_TARGET,
    CITY_TYPE_MODEL_PATH, CITY_TYPE_SCALER_PATH, CITY_TYPE_ENCODER_PATH,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, CITY_TYPE_PARAM_GRID, CITY_TYPE_N_ITER,
    LOGS_DIR,
)
from utils.model_utils import save_model, evaluate_classifier, get_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_city_type_model():
    """Train and save the Random Forest city type classifier."""
    logger.info("=" * 60)
    logger.info("CITY TYPE CLASSIFICATION MODEL TRAINING")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    logger.info("Loading city types dataset...")
    df = load_city_types_dataset()
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class distribution:\n{df[CITY_TYPE_TARGET].value_counts()}")

    # ── 2. Split features / target ────────────────────────────────
    X = df[CITY_TYPE_FEATURES].copy()
    y = df[CITY_TYPE_TARGET].copy()

    # ── 3. Stratified train/test split ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 4. Fit preprocessor ───────────────────────────────────────
    preprocessor = CityTypePreprocessor()
    X_train_scaled, y_train_enc = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)
    y_test_enc = preprocessor.transform_target(y_test)

    # ── 5. RandomizedSearchCV ────────────────────────────────────
    logger.info(f"Starting RandomizedSearchCV ({CITY_TYPE_N_ITER} iterations, {CV_FOLDS} folds)...")
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        rf_base,
        param_distributions=CITY_TYPE_PARAM_GRID,
        n_iter=CITY_TYPE_N_ITER,
        cv=cv_strategy,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train_scaled, y_train_enc)

    best_model = search.best_estimator_
    logger.info(f"Best hyperparameters: {search.best_params_}")
    logger.info(f"Best CV accuracy: {search.best_score_:.4f}")

    # ── 6. Evaluate on test set ───────────────────────────────────
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)
    metrics = evaluate_classifier(
        y_test_enc, y_pred, y_proba, class_names=list(preprocessor.label_encoder.classes_)
    )

    logger.info(f"\n{'─'*40}")
    logger.info(f"Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"F1 (weighted):  {metrics['f1_weighted']:.4f}")
    logger.info(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
    logger.info(f"{'─'*40}")
    logger.info(f"\nClassification Report:\n{metrics['classification_report']}")

    # ── 7. Cross-validation on full training set ─────────────────
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train_enc, cv=cv_strategy, scoring="accuracy")
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 8. Feature importance ─────────────────────────────────────
    importance_df = get_feature_importance(best_model, CITY_TYPE_FEATURES)
    logger.info(f"\nFeature Importance:\n{importance_df.to_string(index=False)}")

    # ── 9. Save artifacts ─────────────────────────────────────────
    save_model(best_model, CITY_TYPE_MODEL_PATH, "Random Forest model")
    save_model(preprocessor.scaler, CITY_TYPE_SCALER_PATH, "StandardScaler")
    save_model(preprocessor.label_encoder, CITY_TYPE_ENCODER_PATH, "LabelEncoder")

    # ── 10. Save metrics log ──────────────────────────────────────
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "RandomForestClassifier",
        "best_params": search.best_params_,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "test_accuracy": metrics["accuracy"],
        "test_f1_weighted": metrics["f1_weighted"],
        "test_roc_auc": metrics["roc_auc"],
        "feature_importance": importance_df.to_dict("records"),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "city_type_model_metrics.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Metrics saved → {log_path}")

    logger.info("\n✅ City type model training complete!")
    return best_model, preprocessor, metrics


if __name__ == "__main__":
    train_city_type_model()
