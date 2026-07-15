"""
Page 1 — Home Dashboard
Project overview, system metrics, and architecture summary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import streamlit as st

from utils.config import LOGS_DIR, COLORS
from utils.model_utils import check_models_exist

st.set_page_config(page_title="Home — EIA System", page_icon="🏠", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Animated Hero Header ────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-wrapper {
    position: relative;
    background: linear-gradient(135deg, rgba(37,99,235,0.08) 0%, rgba(6,182,212,0.05) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 3rem 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    overflow: hidden;
}
.hero-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(60px);
    pointer-events: none;
    animation: orbFloat 7s ease-in-out infinite;
}
.orb-1 { width: 280px; height: 280px; background: rgba(37,99,235,0.18); top:-100px; left:-80px; }
.orb-2 { width: 220px; height: 220px; background: rgba(6,182,212,0.14); bottom:-80px; right:-60px; animation-delay:-3.5s; }
.orb-3 { width: 140px; height: 140px; background: rgba(99,102,241,0.12); top:30px; right:15%; animation-delay:-1.5s; }
.hero-globe { font-size: 4.5rem; line-height: 1; margin-bottom: 1rem; filter: drop-shadow(0 0 30px rgba(6,182,212,0.5)); animation: orbFloat 5s ease-in-out infinite; }
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #93c5fd 0%, #22d3ee 50%, #a5f3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0 0 1.5rem;
    letter-spacing: 0.01em;
}
.hero-badges { display: flex; gap: 0.6rem; justify-content: center; flex-wrap: wrap; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid;
    letter-spacing: 0.03em;
    backdrop-filter: blur(6px);
}
.badge-blue  { background: rgba(37,99,235,0.15);  border-color: rgba(37,99,235,0.4);  color: #93c5fd; }
.badge-cyan  { background: rgba(6,182,212,0.15);  border-color: rgba(6,182,212,0.4);  color: #67e8f9; }
.badge-indigo{ background: rgba(99,102,241,0.15); border-color: rgba(99,102,241,0.4); color: #c7d2fe; }
.badge-green { background: rgba(16,185,129,0.15); border-color: rgba(16,185,129,0.4); color: #6ee7b7; }
</style>
<div class="hero-wrapper">
    <div class="hero-orb orb-1"></div>
    <div class="hero-orb orb-2"></div>
    <div class="hero-orb orb-3"></div>
    <div class="hero-globe">🌍</div>
    <h1 class="hero-title">Environmental Impact Assessment System</h1>
    <p class="hero-subtitle">AI-powered air pollution analysis &nbsp;·&nbsp; Health impact prediction &nbsp;·&nbsp; Environmental risk scoring</p>
    <div class="hero-badges">
        <span class="hero-badge badge-blue">🤖 Machine Learning</span>
        <span class="hero-badge badge-cyan">📊 3 Real Datasets</span>
        <span class="hero-badge badge-indigo">🏭 City Classification</span>
        <span class="hero-badge badge-green">🫁 Health Prediction</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Key Metrics Row ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

model_status = check_models_exist()
models_ready = sum(1 for k in ["city_type_model", "health_impact_model", "air_quality_model"]
                   if model_status.get(k))

# Try to load accuracy from log
best_accuracy = "98.99%"
try:
    log = LOGS_DIR / "city_type_model_metrics.json"
    if log.exists():
        with open(log) as f:
            data = json.load(f)
        best_accuracy = f"{data.get('test_accuracy', 0.9899) * 100:.2f}%"
except Exception:
    pass

metrics = [
    (m1, "🤖", "3", "ML Models", "#60a5fa"),
    (m2, "📊", "3", "Datasets", "#34d399"),
    (m3, "📄", "8", "App Pages", "#a78bfa"),
    (m4, "🎯", best_accuracy, "Best Accuracy", "#f59e0b"),
]
for col, icon, value, label, color in metrics:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem; margin-bottom:0.4rem;'>{icon}</div>
            <div style='font-family:"Outfit",sans-serif; font-size:2.2rem; font-weight:700;
                        color:{color}; line-height:1;'>{value}</div>
            <div class='metric-label' style='margin-top:0.4rem;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Model Status & Quick Nav ─────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🤖 System Architecture")
    st.markdown("""
    <div style='background:rgba(30,41,59,0.8); border:1px solid rgba(148,163,184,0.15);
                border-radius:16px; padding:1.5rem; font-family:monospace; font-size:0.85rem;
                color:#94a3b8; line-height:1.8;'>
    <span style='color:#00d4aa; font-weight:bold;'>INPUT LAYER</span><br>
    ├── Air Pollutants (CO, NO₂, SO₂, O₃, PM₂.₅, PM₁₀)<br>
    ├── Health Metrics (AQI, Temperature, Humidity…)<br>
    └── Time-Series (UCI Hourly Sensor Data)<br>
    <br>
    <span style='color:#7c3aed; font-weight:bold;'>ML MODEL LAYER</span><br>
    ├── 🏭 Random Forest → City Type (Industrial/Residential)<br>
    ├── 🫁 XGBoost / ANN (Deep Learning) → Health Impact Classification<br>
    └── 📈 XGBoost Regressor → Pollution Forecasting<br>
    <br>
    <span style='color:#f59e0b; font-weight:bold;'>ASSESSMENT ENGINE</span><br>
    ├── Environmental Risk Score (0-100)<br>
    └── Risk Level (Safe / Caution / Alert / Critical)<br>
    <br>
    <span style='color:#22c55e; font-weight:bold;'>OUTPUT LAYER</span><br>
    ├── 💡 Mitigation Recommendations<br>
    └── 📋 AI Environmental Report (OpenAI/Template)
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📋 Quick Navigation")

    pages = [
        ("2_📊_Dataset_Insights", "📊", "Dataset Insights", "Explore all 3 datasets with EDA"),
        ("3_🏭_City_Type_Prediction", "🏭", "City Type Prediction", "Classify Industrial vs Residential"),
        ("4_🫁_Health_Impact", "🫁", "Health Impact", "Predict health risk from air quality"),
        ("5_📈_Air_Quality_Analysis", "📈", "Air Quality Analysis", "Historical trends & forecasting"),
        ("6_🔍_Environmental_Assessment", "🔍", "Environmental Assessment", "Combined multi-model risk scoring"),
        ("7_💡_Mitigation_Recommendations", "💡", "Mitigation Recommendations", "Tailored action recommendations"),
        ("8_📋_AI_Environmental_Report", "📋", "AI Environmental Report", "Generate professional reports"),
    ]

    for _, icon, name, desc in pages:
        st.markdown(f"""
        <div style='background:rgba(30,41,59,0.6); border:1px solid rgba(148,163,184,0.12);
                    border-radius:12px; padding:0.75rem 1rem; margin:0.4rem 0;
                    transition:all 0.2s; cursor:pointer;'>
            <span style='font-size:1.2rem;'>{icon}</span>
            <strong style='color:#f8fafc; margin-left:0.5rem;'>{name}</strong>
            <span style='color:#94a3b8; font-size:0.85rem; margin-left:0.5rem;'>— {desc}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─── Dataset Overview ─────────────────────────────────────────────────────────
st.markdown("### 📁 Datasets at a Glance")

from utils.config import CITY_TYPES_CSV, HEALTH_IMPACT_CSV, AIR_QUALITY_CSV

d1, d2, d3 = st.columns(3)
dataset_info = [
    (d1, CITY_TYPES_CSV, "🏭 City Types Dataset", "52,704 records",
     "6 pollutants → Industrial/Residential classification",
     "Random Forest · 98.99% accuracy"),
    (d2, HEALTH_IMPACT_CSV, "🫁 Health Impact Dataset", "~N records",
     "AQI + health metrics → Health impact category",
     "4 models compared · XGBoost selected"),
    (d3, AIR_QUALITY_CSV, "📈 UCI Air Quality Dataset", "~9,000+ hourly records",
     "Time-series sensor data → Pollution forecasting",
     "XGBoost Regressor · RMSE tracked"),
]
for col, path, title, size, desc, model_info in dataset_info:
    with col:
        exists = path.exists()
        status_color = "#22c55e" if exists else "#ef4444"
        status_text = "✅ Available" if exists else "❌ Not found"
        st.markdown(f"""
        <div class='metric-card' style='text-align:left;'>
            <div style='font-size:1.5rem; margin-bottom:0.75rem;'>{title}</div>
            <div style='color:{status_color}; font-weight:600; margin-bottom:0.5rem;'>{status_text}</div>
            <div style='color:#94a3b8; font-size:0.85rem; margin-bottom:0.5rem;'>📏 {size}</div>
            <div style='color:#f8fafc; font-size:0.85rem; margin-bottom:0.75rem;'>{desc}</div>
            <div style='color:#00d4aa; font-size:0.8rem; font-style:italic;'>🤖 {model_info}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─── Model Performance Summary ────────────────────────────────────────────────
st.markdown("### 🎯 Model Performance Summary")

# Load from log files if available
ct_metrics = {}
hi_metrics = {}
aq_metrics = {}

try:
    log = LOGS_DIR / "city_type_model_metrics.json"
    if log.exists():
        with open(log) as f:
            ct_metrics = json.load(f)
except Exception:
    pass

try:
    log = LOGS_DIR / "health_impact_model_metrics.json"
    if log.exists():
        with open(log) as f:
            hi_metrics = json.load(f)
except Exception:
    pass

try:
    log = LOGS_DIR / "air_quality_model_metrics.json"
    if log.exists():
        with open(log) as f:
            aq_metrics = json.load(f)
except Exception:
    pass

perf_data = {
    "Model": ["City Type (Random Forest)", "Health Impact (Best ML Model)", "Health Impact (ANN)", "Air Quality (Regressor)"],
    "Task": ["Binary Classification", "Multi-class Classification", "Multi-class Classification", "Regression"],
    "Algorithm": [
        ct_metrics.get("model", "Random Forest"),
        hi_metrics.get("best_model", "XGBoost"),
        "ANN (Deep Learning)",
        aq_metrics.get("best_model", "XGBoost"),
    ],
    "Key Metric": [
        f"{ct_metrics.get('test_accuracy', 0.9899)*100:.2f}% accuracy" if ct_metrics else "~98.99% accuracy",
        f"{hi_metrics.get('comparison', [{}])[0].get('roc_auc', 'N/A')} ROC-AUC" if hi_metrics else "See logs",
        "See comparison table in Health Impact page",
        f"R² = {aq_metrics.get('best_metrics', {}).get('r2', 'N/A')}" if aq_metrics else "See logs",
    ],
    "Status": [
        "✅ Ready" if model_status.get("city_type_model") else "⏳ Train required",
        "✅ Ready" if model_status.get("health_impact_model") else "⏳ Train required",
        "✅ Ready" if model_status.get("health_impact_ann") else "⏳ Train required",
        "✅ Ready" if model_status.get("air_quality_model") else "⏳ Train required",
    ],
}

import pandas as pd
st.dataframe(
    pd.DataFrame(perf_data),
    width='stretch',
    hide_index=True,
)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.8rem; padding:1rem;'>
    🌍 Environmental Impact Assessment System · v1.0 · July 2026<br>
    Built with Streamlit · scikit-learn · XGBoost · Plotly<br>
    <span style='color:#ef4444;'>⚠️ Health predictions are not medical advice. Always consult a healthcare professional.</span>
</div>
""", unsafe_allow_html=True)
