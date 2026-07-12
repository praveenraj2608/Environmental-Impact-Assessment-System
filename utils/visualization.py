"""
Visualization utilities — Plotly and Matplotlib wrappers for consistent styling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import COLORS, RISK_LEVELS


# ─── Common Layout Defaults ───────────────────────────────────────────────────
_DARK_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30,41,59,0.6)",
    font=dict(color=COLORS["text_light"], family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=[COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["success"], COLORS["danger"], COLORS["warning"]],
)


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(**_DARK_TEMPLATE)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.3)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.3)")
    return fig


def plot_class_distribution(series: pd.Series, title: str = "Class Distribution") -> go.Figure:
    """
    Pie chart of class distribution.

    Args:
        series: Value counts series.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(go.Pie(
        labels=series.index,
        values=series.values,
        hole=0.4,
        marker=dict(colors=[COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                             COLORS["success"], COLORS["warning"]]),
        textinfo="label+percent",
        textfont_size=12,
    ))
    fig.update_layout(title=title, **_DARK_TEMPLATE)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """
    Correlation heatmap for numeric columns.

    Args:
        df: DataFrame to correlate.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont_size=10,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(title=title, **_DARK_TEMPLATE)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of feature importances.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        top_n: Number of top features to display.

    Returns:
        Plotly Figure.
    """
    df = importance_df.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, COLORS["primary"]], [1, COLORS["accent"]]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in df["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        **_DARK_TEMPLATE,
    )
    return fig


def plot_prediction_probabilities(proba: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    Bar chart of prediction probabilities per class.

    Args:
        proba: Array of shape (n_classes,).
        class_names: Class label names.

    Returns:
        Plotly Figure.
    """
    colors = [COLORS["primary"] if p == max(proba) else COLORS["text_muted"] for p in proba]
    fig = go.Figure(go.Bar(
        x=class_names,
        y=[round(p * 100, 1) for p in proba],
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in proba],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Confidence by Class",
        xaxis_title="Class",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 115],
        **_DARK_TEMPLATE,
    )
    return fig


def plot_risk_gauge(risk_score: float, risk_level: str) -> go.Figure:
    """
    Gauge chart for environmental risk score.

    Args:
        risk_score: Score in [0, 100].
        risk_level: 'Safe', 'Caution', 'Alert', or 'Critical'.

    Returns:
        Plotly Figure.
    """
    color = RISK_LEVELS.get(risk_level, ("", "", "#94a3b8"))[2]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        number={"suffix": "/100", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_muted"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(30,41,59,0.8)",
            "borderwidth": 2,
            "bordercolor": COLORS["text_muted"],
            "steps": [
                {"range": [0, 25], "color": "rgba(34,197,94,0.25)"},
                {"range": [25, 50], "color": "rgba(234,179,8,0.25)"},
                {"range": [50, 75], "color": "rgba(249,115,22,0.25)"},
                {"range": [75, 100], "color": "rgba(239,68,68,0.25)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": risk_score,
            },
        },
        title={"text": f"Environmental Risk Score<br><span style='color:{color};font-size:1.1em'>{risk_level}</span>"},
    ))
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_light"], family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_risk_factor_breakdown(factors: Dict[str, float]) -> go.Figure:
    """
    Donut chart of risk factor contributions.

    Args:
        factors: Dict of factor_name → score.

    Returns:
        Plotly Figure.
    """
    weights = {"pollution_factor": 0.4, "health_impact_factor": 0.3,
               "industrial_factor": 0.2, "trend_factor": 0.1}
    labels = []
    values = []
    for k, w in weights.items():
        labels.append(k.replace("_", " ").title())
        values.append(round(factors.get(k, 0) * w, 2))

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=[COLORS["danger"], COLORS["warning"], COLORS["accent"], COLORS["primary"]]),
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Risk Score Factor Breakdown",
        **_DARK_TEMPLATE,
    )
    return fig


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Time Series",
    add_moving_avg: bool = True,
    window: int = 168,
) -> go.Figure:
    """
    Interactive time-series line chart with optional moving average.

    Args:
        df: DataFrame with datetime and target column.
        x_col: Column name for x-axis (datetime).
        y_col: Column name for y-axis (values).
        title: Chart title.
        add_moving_avg: Whether to overlay 7-day moving average.
        window: Moving average window size.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        name=y_col,
        line=dict(color=COLORS["primary"], width=1),
        opacity=0.7,
    ))

    if add_moving_avg and len(df) > window:
        ma = df[y_col].rolling(window=window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=ma,
            name=f"Moving Avg ({window}h)",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_col,
        hovermode="x unified",
        **_DARK_TEMPLATE,
    )
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
) -> go.Figure:
    """
    Scatter plot of actual vs predicted values with ideal line.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(color=COLORS["primary"], size=4, opacity=0.5),
    ))
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        name="Perfect Fit",
        line=dict(color=COLORS["accent"], dash="dash", width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Actual",
        yaxis_title="Predicted",
        **_DARK_TEMPLATE,
    )
    return fig


def plot_pollutant_distributions(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Box plots of pollutant distributions grouped side-by-side.

    Args:
        df: DataFrame with pollutant columns.
        features: List of column names to plot.

    Returns:
        Plotly Figure.
    """
    n = len(features)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)
    color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                   COLORS["success"], COLORS["warning"], COLORS["danger"]]

    for i, feat in enumerate(features):
        row, col = divmod(i, cols)
        if feat in df.columns:
            fig.add_trace(
                go.Box(y=df[feat], name=feat,
                       marker_color=color_cycle[i % len(color_cycle)],
                       showlegend=False),
                row=row + 1, col=col + 1,
            )
    fig.update_layout(title="Pollutant Distributions", height=400, **_DARK_TEMPLATE)
    return fig
