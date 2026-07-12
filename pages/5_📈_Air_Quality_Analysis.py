"""
Page 5 — Air Quality Analysis
Historical time-series analysis and pollution forecasting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.config import (
    AIR_QUALITY_TARGET, AIR_QUALITY_TARGET_FALLBACK,
    AIR_QUALITY_MODEL_PATH, AIR_QUALITY_SCALER_PATH, LOGS_DIR
)
from utils.visualization import plot_time_series, plot_actual_vs_predicted, apply_dark_theme
from utils.model_utils import load_model

st.set_page_config(page_title="Air Quality Analysis — EIA", page_icon="📈", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_aq_artifacts():
    meta_path = AIR_QUALITY_MODEL_PATH.parent / "model_metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    return {
        "model": load_model(AIR_QUALITY_MODEL_PATH, "air_quality_model"),
        "scaler": load_model(AIR_QUALITY_SCALER_PATH, "air_quality_scaler"),
        "metadata": metadata,
    }


@st.cache_data(ttl=3600)
def load_and_prep_aq_data():
    from src.data_loader import load_air_quality_dataset
    from src.preprocessing import AirQualityPreprocessor
    df = load_air_quality_dataset()
    target_col = AIR_QUALITY_TARGET if AIR_QUALITY_TARGET in df.columns else AIR_QUALITY_TARGET_FALLBACK
    prep = AirQualityPreprocessor(target_col=target_col)
    X_scaled, y, X_df = prep.fit_transform_df(df)
    return df, X_scaled, y, X_df, target_col, prep


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <h1 class='page-title'>📈 Air Quality Analysis</h1>
    <p class='page-subtitle'>Historical pollution trends, seasonal patterns, and ML-powered forecasting</p>
</div>
""", unsafe_allow_html=True)

arts = get_aq_artifacts()
model_ready = arts["model"] is not None

# Load model performance
aq_log = {}
try:
    lp = LOGS_DIR / "air_quality_model_metrics.json"
    if lp.exists():
        with open(lp) as f:
            aq_log = json.load(f)
except Exception:
    pass

# Model metrics bar
if arts["metadata"]:
    m = arts["metadata"].get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Algorithm", arts["metadata"].get("best_model_name", "N/A"))
    with c2: st.metric("RMSE", f"{m.get('rmse', 'N/A')}")
    with c3: st.metric("MAE", f"{m.get('mae', 'N/A')}")
    with c4: st.metric("R² Score", f"{m.get('r2', 'N/A')}")
    st.markdown("---")

if not model_ready:
    st.warning("⏳ Air quality model not found. Run: `python src/air_quality_model.py`")

# ─── Load Data ─────────────────────────────────────────────────────────────────
try:
    df, X_scaled, y, X_df, target_col, prep = load_and_prep_aq_data()

    # ── Tab Layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Historical Trends",
        "🗓️ Seasonal Patterns",
        "🔮 Predictions vs Actual",
        "🔎 Insights",
    ])

    # ── Tab 1: Historical Time Series ────────────────────────────────────────
    with tab1:
        st.markdown(f"### {target_col} — Full Historical Time Series")

        if "DateTime" in df.columns and target_col in df.columns:
            df_plot = df.dropna(subset=[target_col]).copy()

            # Date range filter
            if len(df_plot) > 0:
                min_date = df_plot["DateTime"].min().date()
                max_date = df_plot["DateTime"].max().date()
                date_range = st.date_input(
                    "Filter date range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if len(date_range) == 2:
                    mask = (df_plot["DateTime"].dt.date >= date_range[0]) & \
                           (df_plot["DateTime"].dt.date <= date_range[1])
                    df_filtered = df_plot[mask]
                else:
                    df_filtered = df_plot

                # Moving average window slider
                ma_window = st.slider("Moving average window (hours):", 24, 720, 168, 24)
                fig = plot_time_series(df_filtered, "DateTime", target_col,
                                       title=f"{target_col} — Historical Trend", window=ma_window)
                st.plotly_chart(fig, use_container_width=True)

                # Basic stats
                cc1, cc2, cc3, cc4 = st.columns(4)
                with cc1: st.metric("Mean", f"{df_filtered[target_col].mean():.2f}")
                with cc2: st.metric("Max", f"{df_filtered[target_col].max():.2f}")
                with cc3: st.metric("Min", f"{df_filtered[target_col].min():.2f}")
                with cc4: st.metric("Std Dev", f"{df_filtered[target_col].std():.2f}")
        else:
            st.warning("DateTime or target column not available for time-series plotting.")

    # ── Tab 2: Seasonal Patterns ──────────────────────────────────────────────
    with tab2:
        st.markdown(f"### {target_col} — Seasonal & Temporal Patterns")

        if "DateTime" in df.columns and target_col in df.columns:
            df_s = df.dropna(subset=[target_col]).copy()
            df_s["Hour"] = df_s["DateTime"].dt.hour
            df_s["DayOfWeek"] = df_s["DateTime"].dt.day_name()
            df_s["Month"] = df_s["DateTime"].dt.month_name()

            import plotly.express as px
            DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            MONTH_ORDER = ["January","February","March","April","May","June",
                           "July","August","September","October","November","December"]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**By Hour of Day**")
                hourly = df_s.groupby("Hour")[target_col].mean().reset_index()
                fig_h = px.line(hourly, x="Hour", y=target_col, template="plotly_dark",
                                color_discrete_sequence=["#00d4aa"])
                fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.6)")
                st.plotly_chart(fig_h, use_container_width=True)

            with col_b:
                st.markdown("**By Day of Week**")
                dow = df_s.groupby("DayOfWeek")[target_col].mean().reindex(DOW_ORDER).reset_index()
                fig_d = px.bar(dow, x="DayOfWeek", y=target_col, template="plotly_dark",
                               color_discrete_sequence=["#7c3aed"])
                fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.6)")
                st.plotly_chart(fig_d, use_container_width=True)

            st.markdown("**By Month**")
            monthly = df_s.groupby("Month")[target_col].mean().reindex(MONTH_ORDER).reset_index()
            fig_m = px.bar(monthly, x="Month", y=target_col, template="plotly_dark",
                           color_discrete_sequence=["#f59e0b"])
            fig_m.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.6)")
            st.plotly_chart(fig_m, use_container_width=True)

    # ── Tab 3: Predictions vs Actual ──────────────────────────────────────────
    with tab3:
        st.markdown("### Model Predictions vs Actual Values (Test Set)")

        if model_ready:
            with st.spinner("Generating predictions on test set..."):
                split_idx = int(len(y) * 0.80)
                X_test = X_scaled[split_idx:]
                y_test = y[split_idx:]
                y_pred = arts["model"].predict(X_test)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.plotly_chart(
                    plot_actual_vs_predicted(y_test, y_pred,
                                             title=f"Actual vs Predicted — {target_col}"),
                    use_container_width=True
                )
            with col_b:
                # Time-based predictions plot
                df_pred_plot = pd.DataFrame({
                    "Index": range(len(y_test)),
                    "Actual": y_test,
                    "Predicted": y_pred,
                }).head(500)
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(
                    x=df_pred_plot["Index"], y=df_pred_plot["Actual"],
                    name="Actual", line=dict(color="#00d4aa", width=1.5)
                ))
                fig_compare.add_trace(go.Scatter(
                    x=df_pred_plot["Index"], y=df_pred_plot["Predicted"],
                    name="Predicted", line=dict(color="#f59e0b", width=1.5, dash="dash")
                ))
                fig_compare.update_layout(
                    title="Actual vs Predicted (first 500 test samples)",
                    xaxis_title="Sample Index",
                    yaxis_title=target_col,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30,41,59,0.6)",
                    font=dict(color="#f8fafc"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_compare, use_container_width=True)

            # Error distribution
            residuals = y_test - y_pred
            fig_res = go.Figure(go.Histogram(
                x=residuals, nbinsx=50,
                marker_color="#7c3aed", opacity=0.8,
            ))
            fig_res.update_layout(
                title="Residual Distribution (Actual − Predicted)",
                xaxis_title="Residual",
                yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.6)",
                font=dict(color="#f8fafc"),
            )
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.warning("Model not trained yet. Run `python src/air_quality_model.py` to train.")

    # ── Tab 4: Insights ───────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔎 Key Pollution Insights")

        if "DateTime" in df.columns and target_col in df.columns:
            df_ins = df.dropna(subset=[target_col]).copy()

            from utils.config import POLLUTION_SEVERITY_THRESHOLDS
            severity_counts = {}
            for sev, (lo, hi) in POLLUTION_SEVERITY_THRESHOLDS.items():
                count = ((df_ins[target_col] >= lo) & (df_ins[target_col] < hi)).sum()
                severity_counts[sev] = int(count)

            sev_cols = st.columns(len(severity_counts))
            sev_colors = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}
            for col, (sev, cnt) in zip(sev_cols, severity_counts.items()):
                with col:
                    pct = cnt / len(df_ins) * 100
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color:{sev_colors.get(sev,"#fff")}; font-size:1.5rem; font-weight:700;'>{sev}</div>
                        <div class='metric-value' style='font-size:2rem;'>{pct:.1f}%</div>
                        <div class='metric-label'>{cnt:,} records</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Peak pollution periods
            if "DateTime" in df_ins.columns:
                st.markdown("**🔝 Top 10 Peak Pollution Periods**")
                peak = df_ins.nlargest(10, target_col)[["DateTime", target_col]].reset_index(drop=True)
                st.dataframe(peak, use_container_width=True)

            # Overall trend
            st.markdown("**📊 Pollution Trend Analysis**")
            half = len(df_ins) // 2
            first_half_mean = df_ins[target_col].iloc[:half].mean()
            second_half_mean = df_ins[target_col].iloc[half:].mean()
            trend_pct = ((second_half_mean - first_half_mean) / first_half_mean) * 100

            if trend_pct > 5:
                trend_label, trend_color = "📈 Worsening", "#ef4444"
                st.session_state["aq_trend"] = "worsening"
            elif trend_pct < -5:
                trend_label, trend_color = "📉 Improving", "#22c55e"
                st.session_state["aq_trend"] = "improving"
            else:
                trend_label, trend_color = "➡️ Stable", "#eab308"
                st.session_state["aq_trend"] = "stable"

            st.markdown(f"""
            <div class='info-box'>
                <strong>Trend: <span style='color:{trend_color};'>{trend_label}</span></strong><br>
                First half average: {first_half_mean:.2f} → Second half average: {second_half_mean:.2f}<br>
                Change: <strong style='color:{trend_color};'>{trend_pct:+.1f}%</strong>
            </div>
            """, unsafe_allow_html=True)

            st.session_state["aq_prediction_level"] = float(df_ins[target_col].tail(168).mean())

except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}")
    st.info("Please place `UCI_AirQuality.csv` in the `data/` folder.")
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
