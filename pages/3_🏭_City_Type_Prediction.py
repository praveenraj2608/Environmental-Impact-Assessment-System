"""
Page 3 — City Type Prediction
Industrial vs Residential area classification from pollutant inputs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from utils.config import CITY_TYPE_FEATURES, POLLUTANT_RANGES, COLORS
from utils.validators import validate_pollutant_inputs, format_validation_errors
from utils.visualization import plot_feature_importance, plot_prediction_probabilities
from utils.model_utils import get_feature_importance, load_model
from utils.config import CITY_TYPE_MODEL_PATH, CITY_TYPE_SCALER_PATH, CITY_TYPE_ENCODER_PATH

st.set_page_config(page_title="City Type Prediction — EIA", page_icon="🏭", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_city_type_artifacts():
    return {
        "model": load_model(CITY_TYPE_MODEL_PATH, "city_type_model"),
        "scaler": load_model(CITY_TYPE_SCALER_PATH, "city_type_scaler"),
        "encoder": load_model(CITY_TYPE_ENCODER_PATH, "city_type_encoder"),
    }


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <h1 class='page-title'>🏭 City Type Prediction</h1>
    <p class='page-subtitle'>Classify area as Industrial or Residential based on air pollutant concentrations</p>
</div>
""", unsafe_allow_html=True)

arts = get_city_type_artifacts()
model_ready = arts["model"] is not None

if not model_ready:
    st.warning("⏳ City Type model not found. Please run: `python src/city_type_model.py`")

# ─── Model Info Box ────────────────────────────────────────────────────────────
with st.expander("ℹ️ About This Model", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Algorithm:** Random Forest Classifier  
        **Hyperparameter Tuning:** RandomizedSearchCV (5-fold CV)  
        **Training Data:** 52,704 balanced records  
        **Test Accuracy:** ~98.99%  
        """)
    with col_b:
        st.markdown("""
        **Features Used:** CO, NO₂, SO₂, O₃, PM₂.₅, PM₁₀  
        **Target:** Industrial vs Residential  
        **Preprocessing:** StandardScaler + LabelEncoder  
        **Random State:** 42 (reproducible)  
        """)

st.markdown("---")

# ─── Input Section ────────────────────────────────────────────────────────────
st.markdown("### 🔢 Enter Pollutant Concentrations")
st.markdown(
    "<p style='color:#94a3b8; font-size:0.9rem;'>"
    "Enter measured air pollutant concentrations. Typical ranges shown in brackets.</p>",
    unsafe_allow_html=True
)

# Preset examples
preset_col1, preset_col2, preset_col3 = st.columns(3)
with preset_col1:
    if st.button("📋 Load Industrial Example"):
        st.session_state.update({
            "co_val": 5.0, "no2_val": 120.0, "so2_val": 180.0,
            "o3_val": 60.0, "pm25_val": 80.0, "pm10_val": 200.0
        })
with preset_col2:
    if st.button("🏘️ Load Residential Example"):
        st.session_state.update({
            "co_val": 0.1, "no2_val": 5.0, "so2_val": 2.0,
            "o3_val": 20.0, "pm25_val": 5.0, "pm10_val": 10.0
        })
with preset_col3:
    if st.button("🔄 Reset Values"):
        for k in ["co_val","no2_val","so2_val","o3_val","pm25_val","pm10_val"]:
            if k in st.session_state:
                del st.session_state[k]

st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
inputs = {}
fields = [
    (c1, "CO", "co_val", "CO (mg/m³)", 0.0, 30.0, 1.0),
    (c1, "NO2", "no2_val", "NO₂ (µg/m³)", 0.0, 400.0, 40.0),
    (c2, "SO2", "so2_val", "SO₂ (µg/m³)", 0.0, 500.0, 20.0),
    (c2, "O3", "o3_val", "O₃ (µg/m³)", 0.0, 300.0, 60.0),
    (c3, "PM2.5", "pm25_val", "PM₂.₅ (µg/m³)", 0.0, 500.0, 25.0),
    (c3, "PM10", "pm10_val", "PM₁₀ (µg/m³)", 0.0, 600.0, 50.0),
]
for col, feat, key, label, min_v, max_v, default in fields:
    with col:
        inputs[feat] = st.number_input(
            label,
            min_value=0.0,
            max_value=max_v * 3,
            value=float(st.session_state.get(key, default)),
            step=0.1,
            key=key,
            help=f"Typical range: 0 – {max_v} {POLLUTANT_RANGES[feat]['unit']}",
        )

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict City Type", disabled=not model_ready, width='stretch')

# ─── Prediction ────────────────────────────────────────────────────────────────
if predict_btn and model_ready:
    valid, errors = validate_pollutant_inputs(inputs)
    if not valid:
        st.error(format_validation_errors(errors))
    else:
        with st.spinner("Running prediction..."):
            model = arts["model"]
            scaler = arts["scaler"]
            encoder = arts["encoder"]

            X = np.array([[inputs[f] for f in CITY_TYPE_FEATURES]])
            X_scaled = scaler.transform(X)
            pred_enc = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            pred_class = encoder.inverse_transform([pred_enc])[0]
            confidence = float(max(proba)) * 100
            class_names = list(encoder.classes_)

        # Store in session for downstream pages
        st.session_state["city_type_result"] = {
            "prediction": pred_class,
            "confidence": confidence / 100,
            "probabilities": dict(zip(class_names, proba.tolist())),
            "inputs": inputs,
        }

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        r1, r2 = st.columns([1, 1])
        with r1:
            badge_class = "badge-industrial" if pred_class == "Industrial" else "badge-residential"
            badge_icon = "🏭" if pred_class == "Industrial" else "🏘️"
            st.markdown(f"""
            <div style='text-align:center; padding:2rem;'>
                <div style='font-size:4rem;'>{badge_icon}</div>
                <div class='prediction-badge {badge_class}' style='margin:1rem auto; display:inline-block;'>
                    {pred_class}
                </div>
                <div style='color:#94a3b8; margin-top:1rem; font-size:1rem;'>
                    Confidence: <strong style='color:#00d4aa;'>{confidence:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.plotly_chart(
                plot_prediction_probabilities(proba, class_names),
                width='stretch'
            )

        st.markdown("---")
        st.markdown("### 📊 Feature Importance & Interpretation")

        fi_col, interp_col = st.columns([1, 1])
        with fi_col:
            importance_df = get_feature_importance(model, CITY_TYPE_FEATURES)
            if not importance_df.empty:
                st.plotly_chart(
                    plot_feature_importance(importance_df, top_n=6),
                    width='stretch'
                )

        with interp_col:
            st.markdown("#### 🧠 Interpretation")
            if pred_class == "Industrial":
                st.markdown("""
                <div class='health-disclaimer' style='border-color:rgba(239,68,68,0.5);'>
                    <strong style='color:#f87171;'>🏭 Industrial Area Detected</strong><br><br>
                    High pollutant concentrations — especially CO, SO₂, and PM₁₀ — 
                    are characteristic of industrial zones with manufacturing, 
                    power generation, or heavy transportation activity.<br><br>
                    <strong>Key indicators:</strong>
                    <ul>
                        <li>Elevated SO₂ → combustion/smelting activity</li>
                        <li>High PM₁₀ → dust from industrial processes</li>
                        <li>High CO → incomplete combustion sources</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box'>
                    <strong style='color:#22c55e;'>🏘️ Residential Area Detected</strong><br><br>
                    Lower pollutant concentrations consistent with residential 
                    neighborhoods. Pollution primarily from domestic heating, 
                    light traffic, and natural sources.<br><br>
                    <strong>Key indicators:</strong>
                    <ul>
                        <li>Low SO₂ → minimal industrial combustion</li>
                        <li>Low PM₁₀ → minimal dust sources</li>
                        <li>Moderate O₃ → photochemical activity</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Show top 3 influential features
            if not importance_df.empty:
                top3 = importance_df.head(3)
                st.markdown("**Top 3 Most Influential Pollutants:**")
                for _, row in top3.iterrows():
                    val = inputs.get(row["feature"], 0)
                    st.markdown(
                        f"• **{row['feature']}**: {val:.2f} "
                        f"(importance: {row['importance']:.3f})"
                    )
