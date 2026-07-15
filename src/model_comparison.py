"""
Model Comparison & Report Generation Script.

Loads all trained Health Impact models (4 ML + 1 ANN), re-evaluates them
on the same held-out test set, auto-selects the best model, and saves:

CSV Reports (reports/):
    - comparison.csv          -- all models, all metrics
    - metrics.csv             -- best model metrics only
    - classification_report.csv
    - training_history.csv    -- ANN epoch history

PNG Plots (reports/):
    - model_accuracy_comparison.png
    - model_precision_comparison.png
    - model_recall_comparison.png
    - model_f1_comparison.png
    - roc_curve.png
    - confusion_matrix.png
    - accuracy_curve.png
    - loss_curve.png
    - learning_curve.png

Usage:
    python src/model_comparison.py
    # or import and call:
    from src.model_comparison import generate_comparison
    generate_comparison()
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/script use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report as sk_classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

from src.data_loader import load_health_impact_dataset
from src.preprocessing import HealthImpactPreprocessor
from src.health_impact_dl_model import HealthImpactANN
from utils.config import (
    HEALTH_IMPACT_TARGET, HEALTH_RECORD_ID_COL,
    HEALTH_IMPACT_MODEL_PATH, HEALTH_IMPACT_SCALER_PATH, HEALTH_IMPACT_ENCODER_PATH,
    HEALTH_IMPACT_ANN_PATH, HEALTH_IMPACT_ANN_CONFIG_PATH, HEALTH_IMPACT_ANN_PREPROCESSOR_PATH,
    HEALTH_IMPACT_ANN_HISTORY_PATH,
    RANDOM_STATE, TEST_SIZE, LOGS_DIR, REPORTS_DIR,
)
from utils.model_utils import load_model, load_ann_model, predict_with_ann, evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Plotting style ───────────────────────────────────────────────────────────
DARK_BG  = "#0f172a"
CARD_BG  = "#1e293b"
PRIMARY  = "#00d4aa"
ACCENT   = "#f59e0b"
DANGER   = "#ef4444"
SUCCESS  = "#22c55e"
TEXT     = "#f8fafc"
MUTED    = "#94a3b8"
PALETTE  = [PRIMARY, "#7c3aed", ACCENT, SUCCESS, DANGER, "#60a5fa"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "figure.dpi": 120,
})


# ─── Data preparation ─────────────────────────────────────────────────────────

def _prepare_test_data() -> Tuple[np.ndarray, np.ndarray, HealthImpactPreprocessor, List[str]]:
    """
    Load dataset and reproduce the exact train/test split used during training.

    Returns:
        (X_test_proc, y_test_enc, preprocessor, class_names)
    """
    df = load_health_impact_dataset()

    drop_cols = [c for c in [HEALTH_RECORD_ID_COL] if c in df.columns]
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols + [HEALTH_IMPACT_TARGET]
        and df[c].dtype in [np.float64, np.int64, float, int, "float64", "int64"]
    ]

    X = df[feature_cols].copy()
    y = df[HEALTH_IMPACT_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = HealthImpactPreprocessor()
    X_train_proc, _ = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    y_test_enc = preprocessor.transform_target(y_test)
    class_names = list(preprocessor.label_encoder.classes_)

    return X_test_proc, y_test_enc, preprocessor, class_names


# ─── ML model evaluation ──────────────────────────────────────────────────────

def _evaluate_ml_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Optional[Dict]:
    """Evaluate the saved best ML model on the test set."""
    model = load_model(HEALTH_IMPACT_MODEL_PATH, "best_ml_model")
    if model is None:
        logger.warning("Best ML model not found — skipping ML evaluation")
        return None

    model_name = type(model).__name__

    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0

    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    metrics = evaluate_classifier(y_test, y_pred, y_proba, class_names)
    metrics["predict_time"] = round(predict_time, 6)
    metrics["train_time"] = None  # training time not recoverable post-hoc
    metrics["model_name"] = model_name
    metrics["y_pred"] = y_pred
    metrics["y_proba"] = y_proba
    return metrics


# ─── ANN evaluation ───────────────────────────────────────────────────────────

def _evaluate_ann_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Optional[Dict]:
    """Evaluate the saved ANN on the test set (uses ANN's own preprocessor)."""
    ann_model = load_ann_model(HEALTH_IMPACT_ANN_PATH, HEALTH_IMPACT_ANN_CONFIG_PATH)
    if ann_model is None:
        logger.warning("ANN model not found — skipping ANN evaluation")
        return None

    t0 = time.time()
    y_pred_arr, y_proba_arr = predict_with_ann(ann_model, X_test)
    predict_time = time.time() - t0

    if len(y_pred_arr) == 0:
        return None

    metrics = evaluate_classifier(y_test, y_pred_arr, y_proba_arr, class_names)

    # Read timing from config if available
    train_time = None
    if HEALTH_IMPACT_ANN_CONFIG_PATH.exists():
        try:
            with open(HEALTH_IMPACT_ANN_CONFIG_PATH, "r") as fh:
                cfg = json.load(fh)
            train_time = cfg.get("train_time_seconds")
        except Exception:
            pass

    metrics["predict_time"] = round(predict_time, 6)
    metrics["train_time"] = train_time
    metrics["model_name"] = "ANN (Deep Learning)"
    metrics["y_pred"] = y_pred_arr
    metrics["y_proba"] = y_proba_arr
    return metrics


# ─── Load individual ML candidate metrics from log ───────────────────────────

def _load_ml_candidate_metrics() -> List[Dict]:
    """
    Load stored metrics for all 4 ML candidates from the training log.

    Returns:
        List of metric dicts (one per ML model), or empty list.
    """
    log_path = LOGS_DIR / "health_impact_model_metrics.json"
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        ml_entries = [
            e for e in data.get("comparison", [])
            if e.get("model_name") != "ANN (Deep Learning)"
        ]
        return ml_entries
    except Exception as exc:
        logger.warning(f"Could not load ML metrics log: {exc}")
        return []


# ─── Build unified comparison DataFrame ──────────────────────────────────────

def _build_comparison_df(
    ml_candidates: List[Dict],
    best_ml_metrics: Optional[Dict],
    ann_metrics: Optional[Dict],
) -> pd.DataFrame:
    """
    Merge ML candidate log entries with live best-ML and ANN evaluations.

    Priority: ML log entries for all 4 candidates; live ANN metrics for ANN.
    """
    rows = []

    # 4 ML candidates from training log
    for entry in ml_candidates:
        rows.append({
            "Model": entry.get("model_name", "?"),
            "Type": "ML",
            "Accuracy": entry.get("accuracy"),
            "Precision": entry.get("precision_weighted"),
            "Recall": entry.get("recall_weighted"),
            "F1 (Weighted)": entry.get("f1_weighted"),
            "ROC AUC": entry.get("roc_auc"),
            "CV Mean": entry.get("cv_mean"),
            "CV Std": entry.get("cv_std"),
            "Train Time (s)": entry.get("train_time"),
            "Predict Time (s)": entry.get("predict_time"),
        })

    # ANN
    if ann_metrics:
        rows.append({
            "Model": "ANN (Deep Learning)",
            "Type": "DL",
            "Accuracy": ann_metrics.get("accuracy"),
            "Precision": ann_metrics.get("precision_weighted"),
            "Recall": ann_metrics.get("recall_weighted"),
            "F1 (Weighted)": ann_metrics.get("f1_weighted"),
            "ROC AUC": ann_metrics.get("roc_auc"),
            "CV Mean": None,
            "CV Std": None,
            "Train Time (s)": ann_metrics.get("train_time"),
            "Predict Time (s)": ann_metrics.get("predict_time"),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "ROC AUC" in df.columns:
        df = df.sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    return df


# ─── Best model selection ─────────────────────────────────────────────────────

def _select_best_model(comparison_df: pd.DataFrame) -> str:
    """
    Automatically select best model by ROC AUC -> F1 -> Accuracy.
    Does NOT favour ML or DL — picks whoever scores highest.

    Returns:
        Name of the best model.
    """
    df = comparison_df.dropna(subset=["ROC AUC", "F1 (Weighted)"])
    if df.empty:
        return "Unknown"
    best_row = df.sort_values(
        ["ROC AUC", "F1 (Weighted)", "Accuracy"],
        ascending=False
    ).iloc[0]
    return str(best_row["Model"])


# ─── CSV report writers ───────────────────────────────────────────────────────

def _save_comparison_csv(comparison_df: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "comparison.csv"
    comparison_df.to_csv(path, index=False)
    logger.info(f"Saved comparison.csv -> {path}")


def _save_metrics_csv(comparison_df: pd.DataFrame, best_model_name: str, out_dir: Path) -> None:
    path = out_dir / "metrics.csv"
    mask = comparison_df["Model"] == best_model_name
    if mask.any():
        comparison_df[mask].to_csv(path, index=False)
    else:
        comparison_df.head(1).to_csv(path, index=False)
    logger.info(f"Saved metrics.csv -> {path}")


def _save_classification_report_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    out_dir: Path,
) -> None:
    report_dict = sk_classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.insert(0, "model", model_name)
    path = out_dir / "classification_report.csv"
    report_df.to_csv(path)
    logger.info(f"Saved classification_report.csv -> {path}")


def _save_training_history_csv(out_dir: Path) -> None:
    if not HEALTH_IMPACT_ANN_HISTORY_PATH.exists():
        return
    try:
        with open(HEALTH_IMPACT_ANN_HISTORY_PATH, "r", encoding="utf-8") as fh:
            history = json.load(fh)
        epochs = list(range(1, history["epochs_run"] + 1))
        hist_df = pd.DataFrame({
            "epoch": epochs,
            "train_accuracy": history["accuracy"],
            "val_accuracy": history["val_accuracy"],
            "train_loss": history["loss"],
            "val_loss": history["val_loss"],
        })
        path = out_dir / "training_history.csv"
        hist_df.to_csv(path, index=False)
        logger.info(f"Saved training_history.csv -> {path}")
    except Exception as exc:
        logger.warning(f"Could not save training_history.csv: {exc}")


# ─── PNG plot generators ──────────────────────────────────────────────────────

def _bar_comparison_plot(
    comparison_df: pd.DataFrame,
    metric_col: str,
    title: str,
    filename: str,
    out_dir: Path,
) -> None:
    """Generic grouped bar chart for a single metric across all models."""
    df = comparison_df.dropna(subset=[metric_col]).copy()
    if df.empty:
        return

    colors = [PALETTE[1] if t == "DL" else PRIMARY for t in df["Type"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["Model"], df[metric_col], color=colors, width=0.55, zorder=3)

    for bar, val in zip(bars, df[metric_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, color=TEXT,
        )

    ax.set_title(title, fontsize=14, color=TEXT, pad=12)
    ax.set_ylabel(metric_col, color=TEXT)
    ax.set_ylim(max(0, df[metric_col].min() - 0.02), min(1.05, df[metric_col].max() + 0.04))
    ax.grid(axis="y", zorder=0)
    plt.xticks(rotation=15, ha="right")

    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PRIMARY, label="Machine Learning"),
        Patch(facecolor=PALETTE[1], label="Deep Learning (ANN)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.3)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {filename} -> {out_path}")


def _plot_roc_curve(
    comparison_df: pd.DataFrame,
    best_ml_metrics: Optional[Dict],
    ann_metrics: Optional[Dict],
    class_names: List[str],
    out_dir: Path,
) -> None:
    """Multi-class macro-average ROC curve for ML best model + ANN."""
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1)

    sources = []
    if best_ml_metrics and best_ml_metrics.get("y_proba") is not None:
        sources.append(("Best ML", best_ml_metrics, PRIMARY))
    if ann_metrics and ann_metrics.get("y_proba") is not None:
        sources.append(("ANN (DL)", ann_metrics, ACCENT))

    for label, metrics, color in sources:
        y_true = metrics.get("y_true_for_roc")
        y_proba = metrics.get("y_proba")
        if y_true is None or y_proba is None:
            continue
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        fpr_dict, tpr_dict = {}, {}
        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        # Macro average
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes
        roc_auc_val = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color=color, lw=2,
                label=f"{label} (AUC={roc_auc_val:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Macro Average (All Classes)", fontsize=13)
    ax.legend(loc="lower right", framealpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "roc_curve.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved roc_curve.png -> {out_path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    filename: str,
    out_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=13, color=TEXT, pad=10)
    # Fix label colours for dark background
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {filename} -> {out_path}")


def _plot_ann_history(out_dir: Path) -> None:
    """Plot ANN training/validation accuracy and loss curves."""
    if not HEALTH_IMPACT_ANN_HISTORY_PATH.exists():
        logger.warning("ANN history not found — skipping training curves")
        return

    with open(HEALTH_IMPACT_ANN_HISTORY_PATH, "r", encoding="utf-8") as fh:
        history = json.load(fh)

    epochs = list(range(1, history["epochs_run"] + 1))

    # ── Accuracy curve ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["accuracy"], color=PRIMARY, lw=2, label="Train Accuracy")
    ax.plot(epochs, history["val_accuracy"], color=ACCENT, lw=2,
            linestyle="--", label="Val Accuracy")
    ax.set_title("ANN — Training & Validation Accuracy", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(framealpha=0.3)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curve.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy_curve.png")

    # ── Loss curve ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["loss"], color=DANGER, lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], color=ACCENT, lw=2,
            linestyle="--", label="Val Loss")
    ax.set_title("ANN — Training & Validation Loss", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(framealpha=0.3)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved loss_curve.png")


def _plot_learning_curve(
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> None:
    """Approximate learning curve using the saved best ML model (subset sizes)."""
    from sklearn.ensemble import RandomForestClassifier
    # Use a lightweight proxy model for learning curve (not the saved 100-tree model)
    proxy = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    try:
        X_test_proc, y_test_enc, preprocessor, _ = _prepare_test_data()
        df = load_health_impact_dataset()
        drop_cols = [c for c in [HEALTH_RECORD_ID_COL] if c in df.columns]
        feature_cols = [
            c for c in df.columns
            if c not in drop_cols + [HEALTH_IMPACT_TARGET]
            and df[c].dtype in [np.float64, np.int64, float, int, "float64", "int64"]
        ]
        X = df[feature_cols].copy()
        y = df[HEALTH_IMPACT_TARGET].copy()
        X_train_full, _, y_train_full, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        X_tr_proc, y_tr_enc = preprocessor.fit_transform(X_train_full, y_train_full)

        train_sizes, train_scores, test_scores = learning_curve(
            proxy, X_tr_proc, y_tr_enc,
            cv=3, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 6),
            n_jobs=-1,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_sizes, train_scores.mean(axis=1), color=PRIMARY, lw=2,
                label="Training Accuracy")
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.15, color=PRIMARY)
        ax.plot(train_sizes, test_scores.mean(axis=1), color=ACCENT, lw=2,
                linestyle="--", label="CV Accuracy")
        ax.fill_between(train_sizes,
                        test_scores.mean(axis=1) - test_scores.std(axis=1),
                        test_scores.mean(axis=1) + test_scores.std(axis=1),
                        alpha=0.15, color=ACCENT)
        ax.set_title("Learning Curve (Random Forest proxy)", fontsize=13)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.legend(framealpha=0.3)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / "learning_curve.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved learning_curve.png")
    except Exception as exc:
        logger.warning(f"Learning curve skipped: {exc}")


# ─── Main comparison entry point ──────────────────────────────────────────────

def generate_comparison() -> Dict:
    """
    Run full model comparison, save all reports and plots.

    Returns:
        Dict with ``comparison_df``, ``best_model``, ``ann_metrics``,
        ``best_ml_metrics`` keys.
    """
    logger.info("=" * 65)
    logger.info("HEALTH IMPACT MODEL COMPARISON & REPORT GENERATION")
    logger.info("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Prepare shared test set ───────────────────────────────────
    logger.info("Preparing test data...")
    X_test_proc, y_test_enc, preprocessor, class_names = _prepare_test_data()
    n_classes = len(class_names)
    logger.info(f"Test set: {X_test_proc.shape} | Classes: {class_names}")

    # ── Evaluate best ML model (live) ────────────────────────────
    logger.info("Evaluating best ML model...")
    best_ml_metrics = _evaluate_ml_model(X_test_proc, y_test_enc, class_names)
    if best_ml_metrics:
        best_ml_metrics["y_true_for_roc"] = y_test_enc
        logger.info(
            f"  Best ML — Acc={best_ml_metrics['accuracy']:.4f} "
            f"F1={best_ml_metrics['f1_weighted']:.4f} "
            f"ROC={best_ml_metrics['roc_auc'] or 'N/A'}"
        )

    # ── Evaluate ANN (live) ───────────────────────────────────────
    logger.info("Evaluating ANN model...")
    # ANN uses its own preprocessor (fitted the same way, same split)
    ann_metrics = _evaluate_ann_model(X_test_proc, y_test_enc, class_names)
    if ann_metrics:
        ann_metrics["y_true_for_roc"] = y_test_enc
        logger.info(
            f"  ANN — Acc={ann_metrics['accuracy']:.4f} "
            f"F1={ann_metrics['f1_weighted']:.4f} "
            f"ROC={ann_metrics['roc_auc'] or 'N/A'}"
        )

    # ── Load stored ML candidate metrics from log ─────────────────
    ml_candidates = _load_ml_candidate_metrics()

    # ── Build comparison DataFrame ────────────────────────────────
    comparison_df = _build_comparison_df(ml_candidates, best_ml_metrics, ann_metrics)
    logger.info(f"\n{'='*65}\nModel Comparison:\n{comparison_df.to_string(index=False)}\n{'='*65}")

    # ── Auto-select best model ────────────────────────────────────
    best_model_name = _select_best_model(comparison_df)
    logger.info(f"\nBest model selected: {best_model_name}")

    # Determine model type of winner
    winner_row = comparison_df[comparison_df["Model"] == best_model_name]
    winner_type = winner_row["Type"].values[0] if not winner_row.empty else "?"

    # ── Save CSV reports ──────────────────────────────────────────
    logger.info("\nSaving CSV reports...")
    _save_comparison_csv(comparison_df, REPORTS_DIR)
    _save_metrics_csv(comparison_df, best_model_name, REPORTS_DIR)
    _save_training_history_csv(REPORTS_DIR)

    # Classification report for best ML model
    if best_ml_metrics and best_ml_metrics.get("y_pred") is not None:
        _save_classification_report_csv(
            y_test_enc, best_ml_metrics["y_pred"],
            class_names,
            best_ml_metrics["model_name"],
            REPORTS_DIR,
        )

    # ── Save PNG plots ────────────────────────────────────────────
    logger.info("\nGenerating plots...")

    # Bar charts for each metric
    _bar_comparison_plot(comparison_df, "Accuracy",      "Model Accuracy Comparison",   "model_accuracy_comparison.png",   REPORTS_DIR)
    _bar_comparison_plot(comparison_df, "Precision",     "Model Precision Comparison",  "model_precision_comparison.png",  REPORTS_DIR)
    _bar_comparison_plot(comparison_df, "Recall",        "Model Recall Comparison",     "model_recall_comparison.png",     REPORTS_DIR)
    _bar_comparison_plot(comparison_df, "F1 (Weighted)", "Model F1 Score Comparison",   "model_f1_comparison.png",         REPORTS_DIR)

    # ROC curve
    _plot_roc_curve(comparison_df, best_ml_metrics, ann_metrics, class_names, REPORTS_DIR)

    # Confusion matrix — best ML
    if best_ml_metrics and best_ml_metrics.get("y_pred") is not None:
        _plot_confusion_matrix(
            y_test_enc, best_ml_metrics["y_pred"], class_names,
            f"Confusion Matrix — {best_ml_metrics['model_name']}",
            "confusion_matrix.png", REPORTS_DIR,
        )

    # ANN training curves
    _plot_ann_history(REPORTS_DIR)

    # Learning curve
    _plot_learning_curve(X_test_proc, y_test_enc, class_names, REPORTS_DIR)

    # ── Update metrics log with comparison result ─────────────────
    _update_comparison_log(comparison_df, best_model_name, winner_type)

    logger.info("\nAll reports and plots saved to reports/")
    logger.info(f"Best model: {best_model_name} ({winner_type})")

    return {
        "comparison_df": comparison_df,
        "best_model": best_model_name,
        "best_model_type": winner_type,
        "ann_metrics": ann_metrics,
        "best_ml_metrics": best_ml_metrics,
        "class_names": class_names,
    }


def _update_comparison_log(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    winner_type: str,
) -> None:
    """Persist comparison result to logs/health_impact_model_metrics.json."""
    log_path = LOGS_DIR / "health_impact_model_metrics.json"
    log_data: Dict = {}
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                log_data = json.load(fh)
        except Exception:
            pass

    # Convert DataFrame to list of dicts, replacing NaN with None
    comparison_records = comparison_df.where(
        comparison_df.notna(), other=None
    ).to_dict(orient="records")

    log_data["best_model_overall"] = best_model_name
    log_data["best_model_type"] = winner_type
    log_data["comparison_generated_at"] = datetime.now().isoformat()
    log_data["full_comparison"] = comparison_records

    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log_data, fh, indent=2, default=str)
        logger.info(f"Updated metrics log -> {log_path}")
    except Exception as exc:
        logger.warning(f"Could not update metrics log: {exc}")


# ─── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = generate_comparison()
    print(f"\nBest model: {results['best_model']} ({results['best_model_type']})")
    print(f"Reports saved to: {REPORTS_DIR}")
