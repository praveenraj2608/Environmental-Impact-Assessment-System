"""
Health Impact Classification Model Training Script.

Trains 4 models, selects best by ROC-AUC → F1 → CV stability.
Algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost
Output: models/health_impact/{best_model,scaler,label_encoder}.pkl
"""

import sys
import os
import logging
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from xgboost import XGBClassifier

from src.data_loader import load_health_impact_dataset
from src.preprocessing import HealthImpactPreprocessor
from utils.config import (
    HEALTH_IMPACT_TARGET, HEALTH_RECORD_ID_COL,
    HEALTH_IMPACT_MODEL_PATH, HEALTH_IMPACT_SCALER_PATH, HEALTH_IMPACT_ENCODER_PATH,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, LOGS_DIR,
)
from utils.model_utils import save_model, evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}


def train_health_impact_model():
    """Train, compare, and save the best health impact classifier."""
    logger.info("=" * 60)
    logger.info("HEALTH IMPACT CLASSIFICATION MODEL TRAINING")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    logger.info("Loading health impact dataset...")
    df = load_health_impact_dataset()
    logger.info(f"Dataset shape: {df.shape}")

    # Inspect target distribution
    class_dist = df[HEALTH_IMPACT_TARGET].value_counts()
    logger.info(f"Class distribution:\n{class_dist}")
    n_classes = class_dist.nunique()
    logger.info(f"Number of classes: {len(class_dist)}")

    # Check missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"Missing values:\n{missing[missing > 0]}")

    # ── 2. Prepare X and y ────────────────────────────────────────
    drop_cols = [c for c in [HEALTH_RECORD_ID_COL] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols + [HEALTH_IMPACT_TARGET]
                    and df[c].dtype in [np.float64, np.int64, float, int, "float64", "int64"]]

    X = df[feature_cols].copy()
    y = df[HEALTH_IMPACT_TARGET].copy()

    # ── 3. Stratified split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 4. Fit preprocessor ───────────────────────────────────────
    preprocessor = HealthImpactPreprocessor()
    X_train_proc, y_train_enc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    y_test_enc = preprocessor.transform_target(y_test)
    class_names = list(preprocessor.label_encoder.classes_)
    logger.info(f"Class labels: {class_names}")

    # XGBoost needs integer labels starting from 0
    # LabelEncoder already produces 0-indexed integers, so use those directly
    xgb_label_map = {label: i for i, label in enumerate(class_names)}

    # ── 5. Train and evaluate all candidates ──────────────────────
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for model_name, model in CANDIDATE_MODELS.items():
        logger.info(f"\nTraining {model_name}...")

        # XGBoost needs integer labels from 0
        if "XGB" in type(model).__name__:
            y_tr = np.array([xgb_label_map[c] for c in y_train])
            y_te = np.array([xgb_label_map[c] for c in y_test])
        else:
            y_tr = y_train_enc
            y_te = y_test_enc

        t0_train = time.time()
        model.fit(X_train_proc, y_tr)
        train_time = round(time.time() - t0_train, 4)

        t0_pred = time.time()
        y_pred = model.predict(X_test_proc)
        predict_time = round(time.time() - t0_pred, 6)
        y_proba = model.predict_proba(X_test_proc) if hasattr(model, "predict_proba") else None

        metrics = evaluate_classifier(y_te, y_pred, y_proba, class_names=class_names)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_proc, y_tr, cv=cv_strategy, scoring="accuracy")

        result = {
            "model_name": model_name,
            "model_obj": model,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "roc_auc": metrics["roc_auc"] or 0.0,
            "train_time": train_time,
            "predict_time": predict_time,
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "is_xgb": "XGB" in type(model).__name__,
        }
        results.append(result)

        logger.info(
            f"  Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1_weighted']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc'] or 'N/A'} | "
            f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

    # ── 6. Model comparison table ─────────────────────────────────
    comparison_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["model_obj", "is_xgb"]}
        for r in results
    ])
    logger.info(f"\n{'─'*60}\nModel Comparison:\n{comparison_df.to_string(index=False)}\n{'─'*60}")

    # ── 7. Select best model (ROC-AUC → F1 → CV stability) ───────
    best = sorted(results, key=lambda r: (r["roc_auc"], r["f1_weighted"], -r["cv_std"]))[-1]
    best_model = best["model_obj"]
    logger.info(f"\n🏆 Best model: {best['model_name']} (ROC-AUC: {best['roc_auc']:.4f})")

    # ── 8. Save artifacts ─────────────────────────────────────────
    save_model(best_model, HEALTH_IMPACT_MODEL_PATH, f"Best model ({best['model_name']})")
    save_model(preprocessor.scaler, HEALTH_IMPACT_SCALER_PATH, "StandardScaler")
    save_model(preprocessor.label_encoder, HEALTH_IMPACT_ENCODER_PATH, "LabelEncoder")

    # ── 9. Save metrics log ───────────────────────────────────────
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best["model_name"],
        "class_labels": class_names,
        "comparison": [
            {
                k: v for k, v in r.items()
                if k not in ["model_obj", "is_xgb"]
            }
            for r in results
        ],
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "health_impact_model_metrics.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Metrics saved → {log_path}")

    logger.info("\n✅ Health impact model training complete!")
    return best_model, preprocessor, results


if __name__ == "__main__":
    train_health_impact_model()
