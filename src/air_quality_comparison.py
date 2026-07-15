"""
Air Quality Model Comparison Script.

Evaluates all 3 trained models (Random Forest, XGBoost, LSTM) on the same
aligned test window and auto-selects the best model by RMSE.

Key points:
  - ML models (RF, XGBoost) use their standard 2-D test set.
  - LSTM uses sequences built from the same data window (look_back offset corrected).
  - Best model is chosen strictly by RMSE — never forced to be LSTM.
  - Updates logs/air_quality_model_metrics.json with comparison results.
  - Saves comparison.csv and metric bar charts to reports/.

Usage:
    python src/air_quality_comparison.py
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_air_quality_dataset
from src.preprocessing import AirQualityPreprocessor
from src.air_quality_lstm_model import (
    select_target, create_sequences, evaluate_lstm,
    load_lstm_model_for_inference, is_lstm_ready,
)
from utils.config import (
    AIR_QUALITY_TARGET, AIR_QUALITY_TARGET_FALLBACK,
    AIR_QUALITY_MODEL_PATH, AIR_QUALITY_SCALER_PATH,
    AIR_QUALITY_LSTM_MODEL_PATH, AIR_QUALITY_LSTM_SCALER_PATH,
    AIR_QUALITY_LSTM_CONFIG_PATH, AIR_QUALITY_LSTM_HISTORY_PATH,
    LSTM_LOOK_BACK, LOGS_DIR, REPORTS_DIR,
)
from utils.model_utils import load_model, evaluate_regressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Plotting style ───────────────────────────────────────────────────────────
DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
PRIMARY = "#00d4aa"
ACCENT  = "#f59e0b"
DANGER  = "#ef4444"
TEXT    = "#f8fafc"
MUTED   = "#94a3b8"
PALETTE = [PRIMARY, "#7c3aed", ACCENT, "#22c55e", DANGER]

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


# ─── Data preparation ─────────────────────────────────────────────────────────

def _prepare_shared_data():
    """
    Load and preprocess the UCI Air Quality dataset consistently.

    Returns:
        Tuple of (X_scaled_2d, y, feature_names, target_col, preprocessor)
        using the existing ML StandardScaler pipeline.
    """
    df = load_air_quality_dataset()
    target_col = select_target(df)
    preprocessor = AirQualityPreprocessor(target_col=target_col)
    X_scaled, y, X_df = preprocessor.fit_transform_df(df)
    return X_scaled, y, list(X_df.columns), target_col, preprocessor


# ─── ML model evaluation (RF and XGBoost) ────────────────────────────────────

def _evaluate_ml_models(
    X_scaled: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """
    Load and evaluate the saved ML model (best of RF/XGBoost).

    Also reads the per-model metrics from the existing metrics log
    so we can show both RF and XGBoost in the comparison table.

    Returns:
        Dict with 'best_ml', 'rf', 'xgb' metric entries.
    """
    split_idx = int(len(y) * 0.80)
    X_test = X_scaled[split_idx:]
    y_test = y[split_idx:]

    # Load saved best ML model
    model = load_model(AIR_QUALITY_MODEL_PATH, "air_quality_best_ml")
    results = {}

    if model is not None:
        t0 = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - t0
        metrics = evaluate_regressor(y_test, y_pred)
        metrics["predict_time"] = round(predict_time, 6)

        # Read best model name from saved metadata
        meta_path = AIR_QUALITY_MODEL_PATH.parent / "model_metadata.json"
        best_name = "Best ML"
        if meta_path.exists():
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
            best_name = meta.get("best_model_name", "Best ML")
        metrics["model_name"] = best_name
        metrics["y_pred"] = y_pred
        metrics["y_true"] = y_test
        results["best_ml"] = metrics

    # Load individual RF+XGBoost metrics from existing log
    log_path = LOGS_DIR / "air_quality_model_metrics.json"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as fh:
            log_data = json.load(fh)
        for key, label in [("rf_metrics", "Random Forest"), ("xgb_metrics", "XGBoost")]:
            if key in log_data:
                entry = dict(log_data[key])
                entry["model_name"] = label
                results[key.replace("_metrics", "")] = entry

    return results


# ─── LSTM evaluation ──────────────────────────────────────────────────────────

def _evaluate_lstm_model(
    X_df_raw: pd.DataFrame,
    y: np.ndarray,
    target_col: str,
    look_back: int = LSTM_LOOK_BACK,
) -> Optional[Dict]:
    """
    Load and evaluate the trained LSTM on the test sequence window.

    Uses the LSTM-specific MinMaxScaler (lstm_scaler.pkl) and
    creates sequences from the 20% test portion.

    Returns:
        Dict with metrics and predictions, or None if LSTM not ready.
    """
    if not is_lstm_ready():
        logger.warning("LSTM artifacts missing — skipping LSTM evaluation.")
        return None

    try:
        import joblib
        import torch

        lstm_scaler = joblib.load(AIR_QUALITY_LSTM_SCALER_PATH)
        model, config = load_lstm_model_for_inference()
        if model is None:
            return None

        split_idx = int(len(y) * 0.80)
        X_raw = X_df_raw.values

        # Scale with LSTM scaler
        X_train_raw = X_raw[:split_idx]
        X_test_raw  = X_raw[split_idx:]
        # Scaler already fitted on train — only transform
        X_test_scaled = lstm_scaler.transform(X_test_raw)

        # For target inverse-transform
        t_min = config.get("target_scaler_min", 0.0)
        t_max = config.get("target_scaler_max", 1.0)

        # We need enough history to build the look-back window for the first test point.
        # Append the last `look_back` rows of training data as prefix.
        X_train_scaled = lstm_scaler.transform(X_train_raw)
        X_prefix = X_train_scaled[-look_back:]
        X_full_test = np.vstack([X_prefix, X_test_scaled])

        # Also build y with same prefix for alignment
        y_prefix = y[split_idx - look_back : split_idx]
        # Scale y prefix
        def scale_y(vals):
            if t_max == t_min:
                return vals
            return (vals - t_min) / (t_max - t_min)

        y_test_raw = y[split_idx:]
        y_full = np.concatenate([scale_y(y_prefix), scale_y(y_test_raw)])

        X_seq, y_seq = create_sequences(X_full_test, y_full, look_back)
        logger.info(f"LSTM test sequences: {X_seq.shape}")

        t0 = time.time()
        model.eval()
        with torch.no_grad():
            X_seq_t = torch.tensor(X_seq, dtype=torch.float32)
            y_pred_scaled = model(X_seq_t).numpy().ravel()
        predict_time = time.time() - t0

        # Inverse-transform
        y_pred = y_pred_scaled * (t_max - t_min) + t_min
        y_true = y_seq * (t_max - t_min) + t_min

        metrics = evaluate_lstm(y_true, y_pred, predict_time,
                                train_time=config.get("train_time_s", 0.0))
        metrics["model_name"] = "LSTM (Deep Learning)"
        metrics["y_pred"] = y_pred
        metrics["y_true"] = y_true
        logger.info(
            f"LSTM eval: RMSE={metrics['rmse']:.4f} | "
            f"MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}"
        )
        return metrics

    except Exception as exc:
        logger.error(f"LSTM evaluation failed: {exc}")
        return None


# ─── Build comparison DataFrame ───────────────────────────────────────────────

def _build_comparison_df(ml_results: Dict, lstm_metrics: Optional[Dict]) -> pd.DataFrame:
    """
    Merge RF, XGBoost, and LSTM metrics into a single comparison DataFrame.

    Args:
        ml_results:   Dict from _evaluate_ml_models (keys: rf, xgb, best_ml).
        lstm_metrics: Dict from _evaluate_lstm_model or None.

    Returns:
        Sorted DataFrame (ascending RMSE) with all models.
    """
    rows = []
    for key, label in [("rf", "Random Forest"), ("xgb", "XGBoost")]:
        entry = ml_results.get(key, {})
        if not entry:
            continue
        rows.append({
            "Model":           label,
            "Type":            "ML",
            "RMSE":            entry.get("rmse"),
            "MAE":             entry.get("mae"),
            "MSE":             entry.get("mse"),
            "MAPE (%)":        entry.get("mape"),
            "R²":              entry.get("r2"),
            "Train Time (s)":  entry.get("train_time"),
            "Predict Time (s)":entry.get("predict_time"),
        })

    if lstm_metrics:
        rows.append({
            "Model":           "LSTM (Deep Learning)",
            "Type":            "DL",
            "RMSE":            lstm_metrics.get("rmse"),
            "MAE":             lstm_metrics.get("mae"),
            "MSE":             lstm_metrics.get("mse"),
            "MAPE (%)":        lstm_metrics.get("mape"),
            "R²":              lstm_metrics.get("r2"),
            "Train Time (s)":  lstm_metrics.get("train_time"),
            "Predict Time (s)":lstm_metrics.get("predict_time"),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "RMSE" in df.columns:
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    return df


# ─── Best model selection ─────────────────────────────────────────────────────

def _select_best_model(comparison_df: pd.DataFrame) -> str:
    """
    Auto-select best model by RMSE (ascending) → R² (descending).
    Never forces LSTM as winner — strictly metric-based.

    Returns:
        Name of the best model.
    """
    df = comparison_df.dropna(subset=["RMSE", "R²"])
    if df.empty:
        return "Unknown"
    best = df.sort_values(["RMSE", "R²"], ascending=[True, False]).iloc[0]
    return str(best["Model"])


# ─── Plot generators ──────────────────────────────────────────────────────────

def _plot_metric_comparison(comparison_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate grouped bar chart comparing RMSE, MAE, R² across all models."""
    metrics_to_plot = [
        ("RMSE",  "lower is better"),
        ("MAE",   "lower is better"),
        ("R²",    "higher is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_col = "Model"

    for ax, (metric, hint) in zip(axes, metrics_to_plot):
        df_plot = comparison_df.dropna(subset=[metric])
        colors = [PALETTE[1] if t == "DL" else PRIMARY for t in df_plot["Type"]]
        bars = ax.bar(df_plot[model_col], df_plot[metric], color=colors, width=0.55, zorder=3)
        for bar, val in zip(bars, df_plot[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (df_plot[metric].max() * 0.01),
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT,
            )
        ax.set_title(f"{metric} ({hint})", fontsize=12)
        ax.set_ylabel(metric)
        ax.grid(axis="y", zorder=0)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PRIMARY, label="Machine Learning"),
        Patch(facecolor=PALETTE[1], label="Deep Learning (LSTM)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", framealpha=0.3, fontsize=9)
    fig.suptitle("Air Quality Model Comparison", fontsize=14, color=TEXT)
    fig.tight_layout()
    path = out_dir / "aq_model_accuracy_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved aq_model_accuracy_comparison.png -> {path}")


# ─── CSV report ───────────────────────────────────────────────────────────────

def _save_comparison_csv(comparison_df: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "aq_comparison.csv"
    comparison_df.drop(columns=["Type"], errors="ignore").to_csv(path, index=False)
    logger.info(f"Saved aq_comparison.csv -> {path}")


# ─── Update metrics log ───────────────────────────────────────────────────────

def _update_metrics_log(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_model_type: str,
) -> None:
    log_path = LOGS_DIR / "air_quality_model_metrics.json"
    log_data: Dict = {}
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as fh:
            log_data = json.load(fh)

    records = comparison_df.where(comparison_df.notna(), other=None).to_dict(orient="records")
    log_data["best_model_overall"] = best_model_name
    log_data["best_model_type"] = best_model_type
    log_data["full_comparison"] = records

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log_data, fh, indent=2, default=str)
    logger.info(f"Metrics log updated -> {log_path}")


# ─── Main entry point ────────────────────────────────────────────────────────

def generate_comparison() -> Dict:
    """
    Run full Air Quality model comparison (RF, XGBoost, LSTM).

    Returns:
        Dict with comparison_df, best_model, best_model_type.
    """
    logger.info("=" * 65)
    logger.info("AIR QUALITY MODEL COMPARISON")
    logger.info("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Shared feature engineering
    logger.info("Preparing shared feature-engineered data...")
    X_scaled, y, feature_names, target_col, preprocessor = _prepare_shared_data()
    logger.info(f"Features: {X_scaled.shape} | Target: {target_col}")

    # Re-build raw X for LSTM (needs unscaled values for its own scaler)
    df = load_air_quality_dataset()
    aq_prep_raw = AirQualityPreprocessor(target_col=target_col)
    X_df_raw, y_raw_series = aq_prep_raw.prepare_features(df)
    y_raw = y_raw_series.values

    # ── Evaluate ML models ────────────────────────────────────────
    logger.info("Evaluating ML models (RF + XGBoost)...")
    ml_results = _evaluate_ml_models(X_scaled, y)

    # ── Evaluate LSTM ─────────────────────────────────────────────
    logger.info("Evaluating LSTM model...")
    lstm_metrics = _evaluate_lstm_model(X_df_raw, y_raw, target_col, look_back=LSTM_LOOK_BACK)

    # ── Build comparison table ────────────────────────────────────
    comparison_df = _build_comparison_df(ml_results, lstm_metrics)
    logger.info(f"\n{'='*65}\nModel Comparison:\n{comparison_df.to_string(index=False)}\n{'='*65}")

    # ── Auto-select best ──────────────────────────────────────────
    best_name = _select_best_model(comparison_df)
    winner_row = comparison_df[comparison_df["Model"] == best_name]
    winner_type = winner_row["Type"].values[0] if not winner_row.empty else "?"
    logger.info(f"\nBest model: {best_name} ({winner_type}) — selected by RMSE")

    # ── Save reports ──────────────────────────────────────────────
    _save_comparison_csv(comparison_df, REPORTS_DIR)
    _plot_metric_comparison(comparison_df, REPORTS_DIR)
    _update_metrics_log(comparison_df, best_name, winner_type)

    logger.info(f"\nAll comparison reports saved to {REPORTS_DIR}")

    return {
        "comparison_df":    comparison_df,
        "best_model":       best_name,
        "best_model_type":  winner_type,
        "ml_results":       ml_results,
        "lstm_metrics":     lstm_metrics,
    }


# ─── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = generate_comparison()
    print(f"\nBest model: {result['best_model']} ({result['best_model_type']})")
    print(f"Reports saved to: {REPORTS_DIR}")
