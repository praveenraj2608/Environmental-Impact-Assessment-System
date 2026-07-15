"""
Air Quality LSTM Training Module (PyTorch).

Integrates a PyTorch LSTM model into the existing Air Quality prediction pipeline.
Replaces the TensorFlow implementation due to Python 3.14 compatibility.
"""

import sys
import json
import logging
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_air_quality_dataset
from src.preprocessing import AirQualityPreprocessor
from utils.config import (
    AIR_QUALITY_TARGET, AIR_QUALITY_TARGET_FALLBACK,
    AIR_QUALITY_LSTM_MODEL_PATH, AIR_QUALITY_LSTM_SCALER_PATH,
    AIR_QUALITY_LSTM_CONFIG_PATH, AIR_QUALITY_LSTM_HISTORY_PATH,
    LSTM_LOOK_BACK, LOGS_DIR, REPORTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Plotting style ───────────────────────────────────────────────────────────
DARK_BG  = "#0f172a"
CARD_BG  = "#1e293b"
PRIMARY  = "#00d4aa"
ACCENT   = "#f59e0b"
DANGER   = "#ef4444"
TEXT     = "#f8fafc"
MUTED    = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   MUTED,
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       "#334155",
    "grid.alpha":       0.5,
    "figure.dpi":       120,
})


# ─── Step 1: Target selection ─────────────────────────────────────────────────

def select_target(df: pd.DataFrame) -> str:
    for col in [AIR_QUALITY_TARGET, AIR_QUALITY_TARGET_FALLBACK]:
        if col in df.columns and df[col].notna().mean() >= 0.7:
            return col
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        return numeric[0]
    raise ValueError("No suitable target column found.")


# ─── Step 2: Sequence generation ─────────────────────────────────────────────

def create_sequences(X: np.ndarray, y: np.ndarray, look_back: int = LSTM_LOOK_BACK) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    n = len(X)
    for i in range(n - look_back):
        X_seq.append(X[i : i + look_back])
        y_seq.append(y[i + look_back])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ─── Step 3: PyTorch LSTM Model ───────────────────────────────────────────────

class AirQualityLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dense_dim: int = 16, dropout: float = 0.2):
        super(AirQualityLSTM, self).__init__()
        # PyTorch LSTM expects input of shape (batch, seq, feature) when batch_first=True
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ─── Step 4: Generate and save plots ─────────────────────────────────────────

def _save_loss_curve(history_data: Dict, out_dir: Path) -> None:
    epochs = list(range(1, len(history_data["loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history_data["loss"],     color=DANGER,  lw=2, label="Train Loss (MSE)")
    axes[0].plot(epochs, history_data["val_loss"], color=ACCENT,  lw=2, linestyle="--", label="Val Loss (MSE)")
    axes[0].set_title("LSTM Training & Validation Loss", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].legend(framealpha=0.3); axes[0].grid(True)

    axes[1].plot(epochs, history_data["mae"],     color=PRIMARY, lw=2, label="Train MAE")
    axes[1].plot(epochs, history_data["val_mae"], color=ACCENT,  lw=2, linestyle="--", label="Val MAE")
    axes[1].set_title("LSTM Training & Validation MAE", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
    axes[1].legend(framealpha=0.3); axes[1].grid(True)

    fig.tight_layout()
    out_path = out_dir / "aq_loss_curve.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, target_col: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, s=8, color=PRIMARY)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted — {target_col} (LSTM)", fontsize=13)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(framealpha=0.3); ax.grid(True)

    ax2 = axes[1]
    n = min(500, len(y_true))
    ax2.plot(range(n), y_true[:n], color=PRIMARY, lw=1.5, label="Actual")
    ax2.plot(range(n), y_pred[:n], color=ACCENT, lw=1.5, linestyle="--", label="LSTM Predicted")
    ax2.set_title(f"{target_col} — First {n} Test Samples", fontsize=13)
    ax2.set_xlabel("Sample Index"); ax2.set_ylabel(target_col)
    ax2.legend(framealpha=0.3); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / "aq_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)


def _save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.4, s=8, color=ACCENT)
    ax.axhline(0, color=DANGER, lw=1.5, linestyle="--")
    ax.set_title("Residual Plot (Actual - Predicted)", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.grid(True)

    ax2 = axes[1]
    ax2.hist(residuals, bins=60, color="#7c3aed", alpha=0.8, edgecolor="none")
    ax2.axvline(0, color=DANGER, lw=1.5, linestyle="--", label="Zero error")
    ax2.set_title("Prediction Error Distribution", fontsize=13)
    ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")
    ax2.legend(framealpha=0.3); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / "aq_residual_plot.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax3 = plt.subplots(figsize=(9, 5))
    ax3.hist(residuals, bins=60, color="#7c3aed", alpha=0.85, edgecolor="none")
    ax3.axvline(0, color=DANGER, lw=1.5, linestyle="--", label="Zero error")
    ax3.set_title("LSTM Prediction Error Distribution", fontsize=13)
    ax3.set_xlabel("Residual (Actual - Predicted)"); ax3.set_ylabel("Count")
    ax3.legend(framealpha=0.3); ax3.grid(True)
    fig2.tight_layout()
    fig2.savefig(out_dir / "aq_prediction_error_distribution.png", bbox_inches="tight")
    plt.close(fig2)


# ─── Step 5: Evaluate ────────────────────────────────────────────────────────

def evaluate_lstm(y_true: np.ndarray, y_pred: np.ndarray, predict_time: float, train_time: float) -> Dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")

    return {
        "rmse":         round(rmse, 6),
        "mae":          round(mae, 6),
        "mse":          round(mse, 6),
        "mape":         round(mape, 4),
        "r2":           round(r2, 6),
        "train_time":   round(train_time, 4),
        "predict_time": round(predict_time, 6),
    }


# ─── Step 6: Save training history CSV ───────────────────────────────────────

def _save_history_csv(history_data: Dict, out_dir: Path) -> None:
    epochs = list(range(1, len(history_data["loss"]) + 1))
    hist_df = pd.DataFrame({
        "epoch":        epochs,
        "train_loss":   history_data["loss"],
        "val_loss":     history_data["val_loss"],
        "train_mae":    history_data["mae"],
        "val_mae":      history_data["val_mae"],
    })
    hist_df.to_csv(out_dir / "lstm_training_history.csv", index=False)


# ─── Step 7: Update metrics log ──────────────────────────────────────────────

def _update_aq_metrics_log(metrics: Dict, target_col: str, feature_names: List[str], look_back: int) -> None:
    log_path = LOGS_DIR / "air_quality_model_metrics.json"
    log_data: Dict = {}
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                log_data = json.load(fh)
        except Exception:
            pass

    log_data["lstm_metrics"] = metrics
    log_data["lstm_target_col"] = target_col
    log_data["lstm_feature_names"] = feature_names
    log_data["lstm_look_back"] = look_back
    log_data["lstm_trained_at"] = datetime.now().isoformat()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log_data, fh, indent=2, default=str)


# ─── Main training pipeline ───────────────────────────────────────────────────

def train_air_quality_lstm(look_back: int = LSTM_LOOK_BACK) -> Dict:
    torch.manual_seed(42)
    np.random.seed(42)

    logger.info("=" * 65)
    logger.info("AIR QUALITY LSTM TRAINING PIPELINE (PyTorch)")
    logger.info(f"Look-back window: {look_back} hours")
    logger.info("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AIR_QUALITY_LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    df = load_air_quality_dataset()
    target_col = select_target(df)
    
    preprocessor = AirQualityPreprocessor(target_col=target_col)
    X_df, y = preprocessor.prepare_features(df)
    feature_names = list(X_df.columns)
    y_arr = y.values

    split_idx = int(len(y_arr) * 0.80)
    X_train_raw = X_df.values[:split_idx]
    X_test_raw  = X_df.values[split_idx:]
    y_train_raw = y_arr[:split_idx]
    y_test_raw  = y_arr[split_idx:]

    lstm_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = lstm_scaler.fit_transform(X_train_raw)
    X_test_scaled  = lstm_scaler.transform(X_test_raw)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_test_scaled  = target_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, look_back)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  look_back)

    # PyTorch DataLoaders
    # Validation split from training data (10%)
    val_split = int(len(X_train_seq) * 0.9)
    X_train_t = torch.tensor(X_train_seq[:val_split], dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq[:val_split], dtype=torch.float32).unsqueeze(-1)
    X_val_t   = torch.tensor(X_train_seq[val_split:], dtype=torch.float32)
    y_val_t   = torch.tensor(y_train_seq[val_split:], dtype=torch.float32).unsqueeze(-1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    n_features = X_train_seq.shape[2]
    model = AirQualityLSTM(input_dim=n_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    logger.info("Training LSTM (EarlyStopping + ReduceLROnPlateau)...")
    t0_train = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            train_mae += torch.abs(preds - batch_y).sum().item()
        
        train_loss /= len(X_train_t)
        train_mae /= len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_mae = torch.abs(val_preds - y_val_t).mean().item()

        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0_train
    epochs_run = len(history["loss"])
    
    if best_model_state:
        model.load_state_dict(best_model_state)

    logger.info("Evaluating on test sequences...")
    t0_pred = time.time()
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
        y_pred_scaled = model(X_test_t).numpy().ravel()
    predict_time = time.time() - t0_pred

    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()

    metrics = evaluate_lstm(y_true, y_pred, predict_time, train_time)
    
    logger.info("Saving artifacts...")
    torch.save(model.state_dict(), AIR_QUALITY_LSTM_MODEL_PATH)
    
    import joblib
    joblib.dump(lstm_scaler, AIR_QUALITY_LSTM_SCALER_PATH)

    config_data = {
        "look_back":         look_back,
        "n_features":        n_features,
        "feature_names":     feature_names,
        "target_col":        target_col,
        "epochs_trained":    epochs_run,
        "train_time_s":      round(train_time, 2),
        "predict_time_s":    round(predict_time, 6),
        "metrics":           metrics,
        "target_scaler_min": float(target_scaler.data_min_[0]),
        "target_scaler_max": float(target_scaler.data_max_[0]),
        "saved_at":          datetime.now().isoformat(),
    }
    with open(AIR_QUALITY_LSTM_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(config_data, fh, indent=2)

    history["epochs_run"] = epochs_run
    history["train_time_s"] = round(train_time, 2)
    with open(AIR_QUALITY_LSTM_HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    _save_loss_curve(history, REPORTS_DIR)
    _save_actual_vs_predicted(y_true, y_pred, target_col, REPORTS_DIR)
    _save_residual_plot(y_true, y_pred, REPORTS_DIR)
    _save_history_csv(history, REPORTS_DIR)
    _update_aq_metrics_log(metrics, target_col, feature_names, look_back)

    return {"metrics": metrics, "look_back": look_back}


# ─── Inference helpers ────────────────────────────────────────────────────────

def load_lstm_model_for_inference():
    if not AIR_QUALITY_LSTM_MODEL_PATH.exists() or not AIR_QUALITY_LSTM_CONFIG_PATH.exists():
        return None, {}
    try:
        with open(AIR_QUALITY_LSTM_CONFIG_PATH, "r") as fh:
            config = json.load(fh)
        model = AirQualityLSTM(input_dim=config["n_features"])
        model.load_state_dict(torch.load(AIR_QUALITY_LSTM_MODEL_PATH, weights_only=True))
        model.eval()
        return model, config
    except Exception as exc:
        logger.error(f"Failed to load LSTM: {exc}")
        return None, {}

def predict_with_lstm(X_seq: np.ndarray, config: Dict) -> np.ndarray:
    model, _ = load_lstm_model_for_inference()
    with torch.no_grad():
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_pred_scaled = model(X_t).numpy().ravel()
    t_min = config.get("target_scaler_min", 0.0)
    t_max = config.get("target_scaler_max", 1.0)
    return y_pred_scaled * (t_max - t_min) + t_min

def is_lstm_ready() -> bool:
    return AIR_QUALITY_LSTM_MODEL_PATH.exists() and AIR_QUALITY_LSTM_CONFIG_PATH.exists()

if __name__ == "__main__":
    train_air_quality_lstm()
