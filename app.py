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
    <style>
    .sidebar-brand {
        text-align: center;
        padding: 1.5rem 0.5rem 1rem;
    }
    .brand-icon {
        font-size: 3.5rem;
        filter: drop-shadow(0 0 18px rgba(6,182,212,0.6));
        animation: orbFloat 5s ease-in-out infinite;
        display: block;
        margin-bottom: 0.5rem;
    }
    .brand-name {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        background: linear-gradient(135deg, #60a5fa, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 0.01em;
    }
    .brand-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.2rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-pill {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        background: rgba(17,24,39,0.6);
        border: 1px solid rgba(255,255,255,0.06);
        font-size: 0.82rem;
        transition: border-color 0.2s;
    }
    .status-pill:hover { border-color: rgba(6,182,212,0.25); }
    .dot-ready   { width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow: 0 0 6px #10b981; flex-shrink:0; }
    .dot-missing { width:8px; height:8px; border-radius:50%; background:#ef4444; box-shadow: 0 0 6px #ef4444; flex-shrink:0; }
    .dot-warn    { width:8px; height:8px; border-radius:50%; background:#f59e0b; box-shadow: 0 0 6px #f59e0b; flex-shrink:0; }
    .pill-name { color: #e2e8f0; font-weight: 500; flex: 1; }
    .pill-status { font-size: 0.72rem; color: #64748b; }
    .sidebar-section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #475569;
        padding: 0.25rem 0.25rem 0.5rem;
        margin-top: 0.5rem;
    }
    .sidebar-footer {
        text-align: center;
        padding: 0.75rem 0;
        font-size: 0.7rem;
        color: #334155;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 0.5rem;
    }
    </style>
    <div class="sidebar-brand">
        <span class="brand-icon">🌍</span>
        <div class="brand-name">EIA System</div>
        <div class="brand-sub">Environmental · AI · Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">🤖 Model Status</div>', unsafe_allow_html=True)

    from utils.model_utils import check_models_exist
    model_status = check_models_exist()

    status_items = [
        ("City Type RF", model_status.get("city_type_model")),
        ("Health Impact", model_status.get("health_impact_model")),
        ("Air Quality", model_status.get("air_quality_model")),
    ]
    for name, ready in status_items:
        dot_class = "dot-ready" if ready else "dot-missing"
        status_text = "Ready" if ready else "Not trained"
        st.markdown(f"""
        <div class="status-pill">
            <span class="{dot_class}"></span>
            <span class="pill-name">{name}</span>
            <span class="pill-status">{status_text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">📁 Datasets</div>', unsafe_allow_html=True)

    from utils.config import CITY_TYPES_CSV, HEALTH_IMPACT_CSV, AIR_QUALITY_CSV
    datasets = [
        ("City Types", CITY_TYPES_CSV),
        ("Health Impact", HEALTH_IMPACT_CSV),
        ("UCI Air Quality", AIR_QUALITY_CSV),
    ]
    for name, path in datasets:
        exists = path.exists()
        dot_class = "dot-ready" if exists else "dot-missing"
        status_text = "Found" if exists else "Missing"
        st.markdown(f"""
        <div class="status-pill">
            <span class="{dot_class}"></span>
            <span class="pill-name">{name}</span>
            <span class="pill-status">{status_text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        v1.0 &nbsp;·&nbsp; July 2026<br>
        <span style="color:#1e3a5f">⚠️ Not for medical use</span>
    </div>
    """, unsafe_allow_html=True)




# ─── App Entry Page ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 4rem 2rem 2rem;">
    <div style="font-size:5rem; filter:drop-shadow(0 0 30px rgba(6,182,212,0.5));
                animation: orbFloat 5s ease-in-out infinite; display:inline-block;">🌍</div>
    <h1 style="font-family:'Outfit',sans-serif; font-weight:800;
               background: linear-gradient(135deg,#93c5fd,#22d3ee,#a5f3fc);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               background-clip:text; font-size:2.6rem; margin:1rem 0; letter-spacing:-0.03em;">
        Environmental Impact Assessment System
    </h1>
    <p style="color:#94a3b8; font-size:1.05rem; max-width:640px; margin:0 auto 2rem; line-height:1.7;">
        An AI-powered platform for analysing air quality data, predicting health impacts,
        and computing environmental risk scores. Use the <strong style="color:#22d3ee;">sidebar</strong>
        to navigate between the 8 interactive modules.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
quickstart = [
    (col1, "🏠", "Start on Home", "Get a full project overview with live model status and dataset summaries.", "#60a5fa"),
    (col2, "🏭", "Run Predictions", "Enter pollutant values to classify city areas or predict health risks.", "#34d399"),
    (col3, "📋", "Generate Report", "Synthesise all predictions into a professional environmental report.", "#f59e0b"),
]
for col, icon, title, desc, color in quickstart:
    with col:
        st.markdown(f"""
        <div style="background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.07);
                    border-top:3px solid {color}; border-radius:14px; padding:1.4rem 1.25rem;
                    text-align:center; transition:all 0.25s ease;">
            <div style="font-size:2.2rem; margin-bottom:0.6rem;">{icon}</div>
            <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1rem;
                        color:{color}; margin-bottom:0.4rem;">{title}</div>
            <div style="color:#64748b; font-size:0.83rem; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# Pre-load models in background so pages feel instant
try:
    load_city_type_artifacts()
    load_health_impact_artifacts()
    load_air_quality_artifacts()
except Exception:
    pass  # Models not yet trained — pages handle gracefully
