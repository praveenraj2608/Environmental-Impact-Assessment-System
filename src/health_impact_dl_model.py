"""
Health Impact Deep Learning Model Training Script.

Trains an Artificial Neural Network (ANN) classifier for Health Impact
prediction using PyTorch. Reuses existing HealthImpactPreprocessor and data loader;
produces metrics comparable to the 4 sklearn/XGBoost candidates in
health_impact_model.py.

Architecture:
    Input -> Dense(128, relu) -> BatchNorm -> Dropout(0.3)
          -> Dense(64,  relu) -> BatchNorm -> Dropout(0.2)
          -> Dense(32,  relu)
          -> Dense(num_classes)
"""

import sys
import os
import json
import logging
import copy
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from src.data_loader import load_health_impact_dataset
from src.preprocessing import HealthImpactPreprocessor
from utils.config import (
    HEALTH_IMPACT_TARGET, HEALTH_RECORD_ID_COL,
    HEALTH_IMPACT_ANN_PATH, HEALTH_IMPACT_ANN_HISTORY_PATH,
    HEALTH_IMPACT_ANN_PREPROCESSOR_PATH, HEALTH_IMPACT_ANN_CONFIG_PATH,
    RANDOM_STATE, TEST_SIZE, LOGS_DIR,
)
from utils.model_utils import save_model, evaluate_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    logger.info("Loading health impact dataset for ANN training ...")
    df = load_health_impact_dataset()
    logger.info(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame):
    drop_cols = [c for c in [HEALTH_RECORD_ID_COL] if c in df.columns]
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols + [HEALTH_IMPACT_TARGET]
        and df[c].dtype in [np.float64, np.int64, float, int, "float64", "int64"]
    ]

    X = df[feature_cols].copy()
    y = df[HEALTH_IMPACT_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = HealthImpactPreprocessor()
    X_train_proc, y_train_enc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    y_test_enc = preprocessor.transform_target(y_test)

    class_names = list(preprocessor.label_encoder.classes_)
    
    return (
        X_train_proc, X_test_proc,
        y_train_enc, y_test_enc,
        preprocessor, class_names,
    )

class HealthImpactANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def build_ann_model(input_dim: int, num_classes: int):
    model = HealthImpactANN(input_dim, num_classes)
    return model

def train_ann_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == targets.data)
            train_total += inputs.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct.double() / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == targets.data)
                val_total += inputs.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct.double() / val_total
        
        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc.item())
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc.item())
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
            
    model.load_state_dict(best_model_wts)
    
    class History:
        def __init__(self, h):
            self.history = h
    return History(history)

def evaluate_ann_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test).to(device)
        outputs = model(inputs)
        y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        
    y_pred = np.argmax(y_proba, axis=1)

    metrics = evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=class_names,
    )
    return metrics

def save_ann_artifacts(
    model,
    preprocessor: HealthImpactPreprocessor,
    history,
    input_dim: int = 0,
    num_classes: int = 0,
    train_time: float = 0.0,
    predict_time: float = 0.0,
) -> None:
    """Save ANN model weights, preprocessor, training history, and architecture config."""
    HEALTH_IMPACT_ANN_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Save PyTorch state dict ────────────────────────────────────
    try:
        torch.save(model.state_dict(), str(HEALTH_IMPACT_ANN_PATH))
        logger.info(f"ANN model saved -> {HEALTH_IMPACT_ANN_PATH}")
    except Exception as exc:
        raise IOError(f"Failed to save PyTorch model: {exc}") from exc

    # ── Save ANN architecture config (required for reload) ─────────
    config_data = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_names": preprocessor.feature_names,
        "class_names": list(preprocessor.label_encoder.classes_),
        "train_time_seconds": round(train_time, 4),
        "predict_time_seconds": round(predict_time, 6),
    }
    try:
        with open(HEALTH_IMPACT_ANN_CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh, indent=2)
        logger.info(f"ANN config saved -> {HEALTH_IMPACT_ANN_CONFIG_PATH}")
    except Exception as exc:
        raise IOError(f"Failed to save ANN config: {exc}") from exc

    # ── Save preprocessor ──────────────────────────────────────────
    success = save_model(
        preprocessor,
        HEALTH_IMPACT_ANN_PREPROCESSOR_PATH,
        "ANN HealthImpactPreprocessor",
    )
    if not success:
        raise IOError("Failed to save ANN preprocessor")

    # ── Save training history ──────────────────────────────────────
    history_data = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "epochs_run": len(history.history["loss"]),
        "train_time_seconds": round(train_time, 4),
    }
    try:
        with open(HEALTH_IMPACT_ANN_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(history_data, fh, indent=2)
    except Exception as exc:
        raise IOError(f"Failed to save training history: {exc}") from exc

def train_health_impact_ann():
    """Full ANN training pipeline with timing, artifact saving, and metrics logging."""
    import time
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    df = load_data()

    (
        X_train, X_test,
        y_train, y_test,
        preprocessor, class_names,
    ) = preprocess_data(df)

    input_dim = X_train.shape[1]
    num_classes = len(class_names)

    model = build_ann_model(input_dim, num_classes)

    # ── Train with timing ──────────────────────────────────────────
    train_start = time.time()
    history = train_ann_model(
        model, X_train, y_train, X_test, y_test,
        epochs=100,
        batch_size=32,
    )
    train_time = time.time() - train_start
    logger.info(f"ANN training completed in {train_time:.2f}s")

    # ── Evaluate with timing ───────────────────────────────────────
    pred_start = time.time()
    metrics = evaluate_ann_model(model, X_test, y_test, class_names)
    predict_time = time.time() - pred_start
    metrics["train_time"] = round(train_time, 4)
    metrics["predict_time"] = round(predict_time, 6)

    save_ann_artifacts(
        model, preprocessor, history,
        input_dim=input_dim,
        num_classes=num_classes,
        train_time=train_time,
        predict_time=predict_time,
    )

    _update_health_impact_metrics_log(metrics, class_names)

    return metrics, history.history

def _update_health_impact_metrics_log(metrics: dict, class_names: list) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "health_impact_model_metrics.json"

    log_data = {}
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                log_data = json.load(fh)
        except Exception:
            pass

    ann_entry = {
        "model_name": "ANN (Deep Learning)",
        "accuracy": metrics.get("accuracy"),
        "f1_weighted": metrics.get("f1_weighted"),
        "precision_weighted": metrics.get("precision_weighted"),
        "recall_weighted": metrics.get("recall_weighted"),
        "roc_auc": metrics.get("roc_auc") or 0.0,
        "train_time": metrics.get("train_time"),
        "predict_time": metrics.get("predict_time"),
        "cv_mean": None,
        "cv_std": None,
    }

    comparison = log_data.get("comparison", [])
    comparison = [e for e in comparison if e.get("model_name") != "ANN (Deep Learning)"]
    comparison.append(ann_entry)
    log_data["comparison"] = comparison
    log_data["class_labels"] = class_names
    log_data["ann_trained_at"] = datetime.now().isoformat()

    valid = [e for e in comparison if e.get("roc_auc") is not None and e["roc_auc"] > 0]
    if valid:
        best_overall = max(valid, key=lambda e: (e["roc_auc"], e["f1_weighted"]))
        log_data["best_model_overall"] = best_overall["model_name"]

    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log_data, fh, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    import time
    start = time.time()
    metrics, _ = train_health_impact_ann()
    elapsed = time.time() - start
    print(f"\\nANN training completed in {elapsed:.1f}s")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1       : {metrics['f1_weighted']:.4f}")
    print(f"  ROC-AUC  : {metrics.get('roc_auc') or 'N/A'}")
