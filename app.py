"""
Streamlit App Entry Point — Environmental Impact Assessment System.

Run with: streamlit run app.py
"""

import sys
import os
from pathlib import Path

# Ensure src and utils are importable from any working directory
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ─── Page Configuration (must be first st call) ───────────────────────────────
st.set_page_config(
    page_title="Environmental Impact Assessment System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "# 🌍 Environmental Impact Assessment System\n"
                 "A college-level Data Science/ML project analyzing air pollution data.\n\n"
                 "**Models:** Random Forest | XGBoost | Regression\n"
                 "**Pages:** 8 interactive pages\n"
                 "**Datasets:** 3 real-world air quality datasets",
    },
)

# ─── Load custom CSS ──────────────────────────────────────────────────────────
css_path = ROOT / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Cached Model & Data Loaders ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading City Type model...")
def load_city_type_artifacts():
    """Load and cache city type model artifacts."""
    from utils.model_utils import load_model
    from utils.config import CITY_TYPE_MODEL_PATH, CITY_TYPE_SCALER_PATH, CITY_TYPE_ENCODER_PATH
    return {
        "model": load_model(CITY_TYPE_MODEL_PATH, "city_type_model"),
        "scaler": load_model(CITY_TYPE_SCALER_PATH, "city_type_scaler"),
        "encoder": load_model(CITY_TYPE_ENCODER_PATH, "city_type_encoder"),
    }


@st.cache_resource(show_spinner="Loading Health Impact model...")
def load_health_impact_artifacts():
    """Load and cache health impact model artifacts."""
    from utils.model_utils import load_model
    from utils.config import HEALTH_IMPACT_MODEL_PATH, HEALTH_IMPACT_SCALER_PATH, HEALTH_IMPACT_ENCODER_PATH
    return {
        "model": load_model(HEALTH_IMPACT_MODEL_PATH, "health_impact_model"),
        "scaler": load_model(HEALTH_IMPACT_SCALER_PATH, "health_impact_scaler"),
        "encoder": load_model(HEALTH_IMPACT_ENCODER_PATH, "health_impact_encoder"),
    }


@st.cache_resource(show_spinner="Loading Air Quality model...")
def load_air_quality_artifacts():
    """Load and cache air quality model artifacts."""
    import json
    from utils.model_utils import load_model
    from utils.config import AIR_QUALITY_MODEL_PATH, AIR_QUALITY_SCALER_PATH

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


@st.cache_data(show_spinner="Loading datasets...", ttl=3600)
def load_all_datasets():
    """Load and cache all 3 datasets."""
    from src.data_loader import validate_datasets
    return validate_datasets()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🌍</div>
        <div style='font-size:1.1rem; font-weight:700; color:#00d4aa;'>
            Env Impact Assessment
        </div>
        <div style='font-size:0.75rem; color:#94a3b8; margin-top:0.25rem;'>
            Data Science ML Project
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status indicators
    st.markdown("### 🤖 Model Status")
    from utils.model_utils import check_models_exist
    model_status = check_models_exist()

    status_items = [
        ("City Type RF", model_status.get("city_type_model")),
        ("Health Impact", model_status.get("health_impact_model")),
        ("Air Quality Pred", model_status.get("air_quality_model")),
    ]
    for name, ready in status_items:
        icon = "🟢" if ready else "🔴"
        status = "Ready" if ready else "Not trained"
        st.markdown(f"{icon} **{name}**: {status}")

    st.markdown("---")

    # Dataset status
    st.markdown("### 📁 Dataset Status")
    from utils.config import CITY_TYPES_CSV, HEALTH_IMPACT_CSV, AIR_QUALITY_CSV
    datasets = [
        ("City Types", CITY_TYPES_CSV),
        ("Health Impact", HEALTH_IMPACT_CSV),
        ("UCI Air Quality", AIR_QUALITY_CSV),
    ]
    for name, path in datasets:
        icon = "✅" if path.exists() else "❌"
        st.markdown(f"{icon} {name}")

    st.markdown("---")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.75rem; text-align:center;'>"
        "v1.0 · July 2026<br>"
        "⚠️ Not for medical use"
        "</div>",
        unsafe_allow_html=True
    )


# ─── Redirect to Home ────────────────────────────────────────────────────────
# app.py is the entry point; Streamlit auto-discovers pages/
# This page redirects users to the Home page content

st.markdown("""
<div style='text-align:center; padding: 4rem 2rem;'>
    <div style='font-size:5rem;'>🌍</div>
    <h1 style='background:linear-gradient(135deg,#00d4aa,#7c3aed);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-size:2.5rem; margin:1rem 0;'>
        Environmental Impact Assessment System
    </h1>
    <p style='color:#94a3b8; font-size:1.1rem; max-width:600px; margin:0 auto;'>
        Navigate using the <b>sidebar</b> to explore datasets, run predictions, 
        and generate environmental reports.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🏠 Start Here**\nVisit the Home page for an overview")
with col2:
    st.info("**🏭 Predict**\nUse City Type or Health Impact prediction pages")
with col3:
    st.info("**📋 Report**\nGenerate a full Environmental Impact Report")

# Pre-load models in background so pages feel instant
try:
    load_city_type_artifacts()
    load_health_impact_artifacts()
    load_air_quality_artifacts()
except Exception:
    pass  # Models not yet trained — pages handle gracefully
