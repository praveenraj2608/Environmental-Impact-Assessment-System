"""
Page 6 — Environmental Assessment
Multi-model synthesis, composite risk score, and gauge visualization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from utils.config import CITY_TYPE_FEATURES, POLLUTANT_RANGES, COLORS
from utils.visualization import plot_risk_gauge, plot_risk_factor_breakdown
from src.assessment_engine import (
    synthesize_assessment, interpret_risk_level, get_pollution_severity
)

st.set_page_config(page_title="Environmental Assessment — EIA", page_icon="🔍", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <h1 class='page-title'>🔍 Environmental Assessment</h1>
    <p class='page-subtitle'>
        Composite environmental risk scoring combining city type, health impact, and air quality predictions
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Check session state for previous predictions ─────────────────────────────
has_ct = "city_type_result" in st.session_state
has_hi = "health_impact_result" in st.session_state
has_aq = "aq_prediction_level" in st.session_state

if has_ct or has_hi:
    st.success("✅ Previous prediction results detected — pre-filling from session.")

# ─── Input Section ────────────────────────────────────────────────────────────
with st.expander("📥 Configure Assessment Inputs", expanded=True):
    st.markdown("#### Assessment Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏭 City Classification**")
        city_default = st.session_state.get("city_type_result", {}).get("prediction", "Industrial")
        ct_conf_default = st.session_state.get("city_type_result", {}).get("confidence", 0.8)

        city_type = st.selectbox(
            "Area Type",
            ["Industrial", "Residential"],
            index=0 if city_default == "Industrial" else 1,
        )
        ct_confidence = st.slider(
            "City Type Confidence",
            0.0, 1.0,
            float(ct_conf_default), 0.01
        )

    with col2:
        st.markdown("**🫁 Health Impact**")
        health_default = st.session_state.get("health_impact_result", {}).get("prediction", "Moderate")
        hi_conf_default = st.session_state.get("health_impact_result", {}).get("confidence", 0.7)
        health_classes = ["Low", "Moderate", "High", "Severe", "Very High"]
        hi_idx = health_classes.index(health_default) if health_default in health_classes else 1

        health_impact = st.selectbox("Health Impact Class", health_classes, index=hi_idx)
        hi_confidence = st.slider("Health Impact Confidence", 0.0, 1.0, float(hi_conf_default), 0.01)

    with col3:
        st.markdown("**📊 Pollution & Trend**")
        aq_default = float(st.session_state.get("aq_prediction_level", 100.0))
        pollution_level = st.number_input(
            "Pollution Level (AQI / concentration)",
            0.0, 500.0, aq_default, 1.0,
            help="Use AQI or the predicted pollutant concentration value"
        )
        trend_default = st.session_state.get("aq_trend", "stable")
        trend = st.selectbox(
            "Pollution Trend",
            ["improving", "stable", "worsening"],
            index=["improving", "stable", "worsening"].index(trend_default),
        )

assess_btn = st.button("🔍 Compute Environmental Risk Score", use_container_width=True)

# ─── Assessment Results ────────────────────────────────────────────────────────
if assess_btn:
    with st.spinner("Computing environmental risk score..."):
        assessment = synthesize_assessment(
            city_type_pred=city_type,
            city_type_confidence=ct_confidence,
            health_impact_pred=health_impact,
            health_impact_confidence=hi_confidence,
            pollution_level=pollution_level,
            predicted_pollution=None,
            trend=trend,
        )

    # Store in session for downstream pages
    st.session_state["assessment"] = assessment

    risk_score = assessment["risk_score"]
    risk_level = assessment["risk_level"]
    risk_color = assessment["risk_color"]
    factors = assessment["factor_breakdown"]

    st.markdown("---")
    st.markdown("### 🎯 Assessment Results")

    # ── Gauge + Factor Breakdown ───────────────────────────────────────────
    g1, g2 = st.columns([1, 1])

    with g1:
        st.plotly_chart(plot_risk_gauge(risk_score, risk_level), use_container_width=True)

    with g2:
        st.plotly_chart(plot_risk_factor_breakdown(factors), use_container_width=True)

    # ── Summary Cards ──────────────────────────────────────────────────────
    st.markdown("### 📊 Assessment Summary")
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem;'>🏭</div>
            <div class='metric-value' style='font-size:1.5rem; color:{("#ef4444" if city_type=="Industrial" else "#22c55e")};'>
                {city_type}
            </div>
            <div class='metric-label'>Area Classification</div>
            <div style='color:#00d4aa; font-size:0.9rem;'>{ct_confidence*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        hi_colors = {"Low":"#22c55e","Moderate":"#eab308","High":"#f97316","Severe":"#ef4444","Very High":"#dc2626"}
        hi_color = hi_colors.get(health_impact, "#94a3b8")
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem;'>🫁</div>
            <div class='metric-value' style='font-size:1.5rem; color:{hi_color};'>{health_impact}</div>
            <div class='metric-label'>Health Impact</div>
            <div style='color:#00d4aa; font-size:0.9rem;'>{hi_confidence*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        sev = get_pollution_severity(pollution_level)
        sev_color = {"Low":"#22c55e","Moderate":"#eab308","High":"#f97316","Severe":"#ef4444"}.get(sev,"#94a3b8")
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem;'>🌫️</div>
            <div class='metric-value' style='font-size:1.5rem; color:{sev_color};'>{sev}</div>
            <div class='metric-label'>Pollution Severity</div>
            <div style='color:#94a3b8; font-size:0.9rem;'>Level: {pollution_level:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with sc4:
        trend_icons = {"improving":"📉","stable":"➡️","worsening":"📈"}
        trend_colors = {"improving":"#22c55e","stable":"#eab308","worsening":"#ef4444"}
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem;'>{trend_icons.get(trend,"➡️")}</div>
            <div class='metric-value' style='font-size:1.5rem; color:{trend_colors.get(trend,"#94a3b8")};'>
                {trend.capitalize()}
            </div>
            <div class='metric-label'>Pollution Trend</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Formula Transparency ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧮 Risk Score Formula")
    st.markdown(f"""
    <div style='background:rgba(30,41,59,0.8); border:1px solid rgba(148,163,184,0.15);
                border-radius:16px; padding:1.5rem; font-family:monospace;'>
        <div style='color:#00d4aa; font-weight:bold; margin-bottom:0.75rem;'>
            Environmental Risk Score = Σ(Weight × Factor)
        </div>
        <div style='color:#f8fafc; line-height:2;'>
            &nbsp;&nbsp;0.4 × Pollution Factor&nbsp;&nbsp;&nbsp;
            <span style='color:#ef4444;'>{factors['pollution_factor']:.1f}</span>
            &nbsp;→&nbsp;
            <span style='color:#ef4444;'>{factors['pollution_factor']*0.4:.2f}</span><br>
            + 0.3 × Health Impact Factor&nbsp;
            <span style='color:#f97316;'>{factors['health_impact_factor']:.1f}</span>
            &nbsp;→&nbsp;
            <span style='color:#f97316;'>{factors['health_impact_factor']*0.3:.2f}</span><br>
            + 0.2 × Industrial Factor&nbsp;&nbsp;&nbsp;&nbsp;
            <span style='color:#eab308;'>{factors['industrial_factor']:.1f}</span>
            &nbsp;→&nbsp;
            <span style='color:#eab308;'>{factors['industrial_factor']*0.2:.2f}</span><br>
            + 0.1 × Trend Factor&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <span style='color:#22c55e;'>{factors['trend_factor']:.1f}</span>
            &nbsp;→&nbsp;
            <span style='color:#22c55e;'>{factors['trend_factor']*0.1:.2f}</span><br>
            <hr style='border-color:rgba(148,163,184,0.3); margin:0.5rem 0;'>
            = <span style='color:{risk_color}; font-size:1.5rem; font-weight:700;'>
                {risk_score:.1f} / 100 — {risk_level}
              </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interpretation ─────────────────────────────────────────────────────
    st.markdown("---")
    risk_descriptions = {
        "Safe": "🟢 **Safe**: Air quality is within acceptable limits. Standard monitoring is sufficient.",
        "Caution": "🟡 **Caution**: Moderate risk. Implement preventive measures and increase monitoring frequency.",
        "Alert": "🟠 **Alert**: Elevated risk. Immediate action required. Activate pollution response protocols.",
        "Critical": "🔴 **Critical**: Environmental emergency. Invoke emergency powers. Protect all residents immediately.",
    }
    st.markdown(f"""
    <div style='background:rgba(30,41,59,0.8); border:2px solid {risk_color}66;
                border-radius:16px; padding:1.5rem;'>
        <h3 style='color:{risk_color}; margin-top:0;'>Risk Level Interpretation</h3>
        <p style='color:#f8fafc; font-size:1rem;'>{assessment['summary']}</p>
        <br>
        <strong>Navigate to:</strong>
        <ul style='color:#94a3b8;'>
            <li>💡 <em>Mitigation Recommendations</em> — for tailored action plans</li>
            <li>📋 <em>AI Environmental Report</em> — for a full professional report</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show risk level legend
    st.markdown("### 📖 Risk Level Reference")
    levels = [("Safe", "0–25", "#22c55e", "Normal operations. Standard monitoring."),
              ("Caution", "26–50", "#eab308", "Preventive measures. Increased monitoring."),
              ("Alert", "51–75", "#f97316", "Active response. Restrict emissions."),
              ("Critical", "76–100", "#ef4444", "Emergency protocol. Immediate action.")]
    for name, rng, color, desc in levels:
        st.markdown(f"""
        <div style='background:rgba(30,41,59,0.6); border-left:4px solid {color};
                    border-radius:8px; padding:0.75rem 1rem; margin:0.5rem 0;'>
            <strong style='color:{color};'>{name}</strong> (Score: {rng}) — {desc}
        </div>
        """, unsafe_allow_html=True)
