"""
Page 2 — Dataset Insights (EDA)
Exploratory Data Analysis for all 3 datasets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.config import CITY_TYPE_FEATURES, CITY_TYPE_TARGET, HEALTH_IMPACT_TARGET
from utils.visualization import (
    plot_class_distribution, plot_correlation_heatmap,
    plot_pollutant_distributions, plot_time_series, apply_dark_theme
)

st.set_page_config(page_title="Dataset Insights — EIA", page_icon="📊", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <h1 class='page-title'>📊 Dataset Insights</h1>
    <p class='page-subtitle'>Exploratory Data Analysis across all 3 air quality datasets</p>
</div>
""", unsafe_allow_html=True)

# ─── Tab Navigation ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏭 City Types Dataset",
    "🫁 Health Impact Dataset",
    "📈 UCI Air Quality Dataset",
])

# ─── CITY TYPES ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🏭 City Type Classification Dataset")
    try:
        from src.data_loader import load_city_types_dataset
        with st.spinner("Loading city types dataset..."):
            df_ct = load_city_types_dataset()

        # Summary
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Records", f"{len(df_ct):,}")
        with c2: st.metric("Features", len(CITY_TYPE_FEATURES))
        with c3: st.metric("Classes", df_ct[CITY_TYPE_TARGET].nunique())
        with c4: st.metric("Missing Values", df_ct.isnull().sum().sum())

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Class Distribution")
            class_dist = df_ct[CITY_TYPE_TARGET].value_counts()
            fig = plot_class_distribution(class_dist, "City Type Distribution")
            st.plotly_chart(fig, width='stretch')

        with col_right:
            st.markdown("#### Summary Statistics")
            st.dataframe(df_ct[CITY_TYPE_FEATURES].describe().round(2), width='stretch')

        st.markdown("#### Pollutant Distributions by City Type")
        fig_box = px.box(
            df_ct.melt(id_vars=[CITY_TYPE_TARGET], value_vars=CITY_TYPE_FEATURES,
                       var_name="Pollutant", value_name="Concentration"),
            x="Pollutant", y="Concentration", color=CITY_TYPE_TARGET,
            color_discrete_map={"Industrial": "#ef4444", "Residential": "#22c55e"},
            template="plotly_dark",
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.6)")
        st.plotly_chart(fig_box, width='stretch')

        st.markdown("#### Correlation Heatmap")
        numeric_df = df_ct[CITY_TYPE_FEATURES].copy()
        numeric_df["Type_encoded"] = (df_ct[CITY_TYPE_TARGET] == "Industrial").astype(int)
        st.plotly_chart(plot_correlation_heatmap(numeric_df, "Pollutant Correlations"), width='stretch')

        st.markdown("#### Raw Data Sample (first 100 rows)")
        st.dataframe(df_ct.head(100), width='stretch')

    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.info("Please place `city_types_dataset.csv` in the `data/` folder.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ─── HEALTH IMPACT ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🫁 Health Impact Dataset")
    try:
        from src.data_loader import load_health_impact_dataset
        with st.spinner("Loading health impact dataset..."):
            df_hi = load_health_impact_dataset()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Records", f"{len(df_hi):,}")
        with c2: st.metric("Features", len(df_hi.columns) - 1)
        with c3: st.metric("Health Classes", df_hi[HEALTH_IMPACT_TARGET].nunique())
        with c4: st.metric("Missing Values", df_hi.isnull().sum().sum())

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### Health Impact Class Distribution")
            hi_dist = df_hi[HEALTH_IMPACT_TARGET].value_counts()
            fig_hi = plot_class_distribution(hi_dist, "Health Impact Class Distribution")
            st.plotly_chart(fig_hi, width='stretch')

        with col_right:
            st.markdown("#### Missing Values Analysis")
            missing = df_hi.isnull().sum()
            if missing.any():
                st.dataframe(
                    missing[missing > 0].rename("Missing Count").to_frame()
                    .assign(Percentage=lambda x: (x["Missing Count"] / len(df_hi) * 100).round(2)),
                    width='stretch',
                )
            else:
                st.success("✅ No missing values detected!")

        # Health metric distributions
        st.markdown("#### Key Health Metric Distributions")
        health_num_cols = [c for c in df_hi.select_dtypes(include=[np.number]).columns
                           if c not in ["RecordID"]][:8]
        if health_num_cols:
            fig_hist = px.histogram(
                df_hi[health_num_cols].melt(var_name="Metric", value_name="Value"),
                x="Value", facet_col="Metric", facet_col_wrap=4,
                template="plotly_dark", height=500,
                color_discrete_sequence=["#00d4aa"],
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.6)")
            st.plotly_chart(fig_hist, width='stretch')

        st.markdown("#### Correlation Heatmap")
        st.plotly_chart(
            plot_correlation_heatmap(df_hi.select_dtypes(include=[np.number]).drop(columns=["RecordID"], errors="ignore"),
                                     "Health Impact Correlations"),
            width='stretch'
        )

    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.info("Please place `health_impact_dataset.csv` in the `data/` folder.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ─── UCI AIR QUALITY ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📈 UCI Air Quality Time-Series Dataset")
    try:
        from src.data_loader import load_air_quality_dataset
        with st.spinner("Loading UCI Air Quality dataset..."):
            df_aq = load_air_quality_dataset()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Records", f"{len(df_aq):,}")
        with c2: st.metric("Columns", len(df_aq.columns))
        with c3:
            if "DateTime" in df_aq.columns:
                date_range = f"{df_aq['DateTime'].min().strftime('%Y-%m')} → {df_aq['DateTime'].max().strftime('%Y-%m')}"
                st.metric("Date Range", date_range)
        with c4:
            st.metric("Missing Values", df_aq.isnull().sum().sum())

        # Target column
        from utils.config import AIR_QUALITY_TARGET, AIR_QUALITY_TARGET_FALLBACK
        target_col = AIR_QUALITY_TARGET if AIR_QUALITY_TARGET in df_aq.columns else AIR_QUALITY_TARGET_FALLBACK

        if "DateTime" in df_aq.columns and target_col in df_aq.columns:
            st.markdown(f"#### {target_col} — Historical Time Series")
            fig_ts = plot_time_series(
                df_aq.dropna(subset=[target_col]),
                "DateTime", target_col,
                title=f"{target_col} Over Time",
                add_moving_avg=True,
            )
            st.plotly_chart(fig_ts, width='stretch')

        # Monthly distribution
        if "DateTime" in df_aq.columns and target_col in df_aq.columns:
            st.markdown(f"#### {target_col} — Monthly Distribution")
            df_aq["Month"] = df_aq["DateTime"].dt.month_name()
            month_order = ["January","February","March","April","May","June",
                           "July","August","September","October","November","December"]
            fig_monthly = px.box(
                df_aq.dropna(subset=[target_col]),
                x="Month", y=target_col,
                category_orders={"Month": month_order},
                template="plotly_dark",
                color_discrete_sequence=["#00d4aa"],
            )
            fig_monthly.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.6)"
            )
            st.plotly_chart(fig_monthly, width='stretch')

        # Numeric col distributions
        numeric_cols = [c for c in df_aq.select_dtypes(include=[np.number]).columns][:6]
        if numeric_cols:
            st.markdown("#### Sensor Reading Distributions")
            st.plotly_chart(plot_pollutant_distributions(df_aq, numeric_cols), width='stretch')

        st.markdown("#### Raw Data Sample")
        st.dataframe(df_aq.head(50), width='stretch')

    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.info("Please place `UCI_AirQuality.csv` in the `data/` folder.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
