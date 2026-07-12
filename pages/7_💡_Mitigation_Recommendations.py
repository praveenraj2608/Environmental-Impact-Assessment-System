"""
Page 7 — Mitigation Recommendations
Categorized, prioritized environmental action recommendations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.mitigation_engine import (
    generate_recommendations, get_recommendation_priority,
    format_recommendations_for_display
)

st.set_page_config(page_title="Mitigation Recommendations — EIA", page_icon="💡", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <h1 class='page-title'>💡 Mitigation Recommendations</h1>
    <p class='page-subtitle'>
        Categorized, prioritized environmental action plans tailored to your assessment results
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Load from session or manual input ────────────────────────────────────────
has_assessment = "assessment" in st.session_state

if has_assessment:
    assessment = st.session_state["assessment"]
    city_type = assessment.get("city_type", "Industrial")
    health_impact = assessment.get("health_impact", "Moderate")
    risk_level = assessment.get("risk_level", "Caution")
    pollution_level = assessment.get("pollution_level", 100.0)
    st.success(f"✅ Loaded from assessment: **{risk_level}** risk — {city_type} area")
else:
    st.info("ℹ️ No assessment found. Configure inputs below (or run Page 6 first).")

with st.expander("⚙️ Recommendation Settings", expanded=not has_assessment):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        city_type = st.selectbox(
            "Area Type",
            ["Industrial", "Residential"],
            index=0 if (has_assessment and city_type == "Industrial") else 0
        )
    with col2:
        health_classes = ["Low", "Moderate", "High", "Severe", "Very High"]
        hi_default = st.session_state.get("assessment", {}).get("health_impact", "Moderate")
        health_impact = st.selectbox(
            "Health Impact",
            health_classes,
            index=health_classes.index(hi_default) if hi_default in health_classes else 1,
        )
    with col3:
        risk_classes = ["Safe", "Caution", "Alert", "Critical"]
        rl_default = st.session_state.get("assessment", {}).get("risk_level", "Caution")
        risk_level = st.selectbox(
            "Risk Level",
            risk_classes,
            index=risk_classes.index(rl_default) if rl_default in risk_classes else 1,
        )
    with col4:
        pl_default = st.session_state.get("assessment", {}).get("pollution_level", 100.0)
        pollution_level = st.number_input("Pollution Level", 0.0, 500.0, float(pl_default), 1.0)

generate_btn = st.button("💡 Generate Recommendations", use_container_width=True)

# ─── Generate on load if from session, or on button click ─────────────────────
run_generation = generate_btn or has_assessment

if run_generation:
    recs = generate_recommendations(pollution_level, city_type, health_impact, risk_level)
    priority = get_recommendation_priority(risk_level)
    formatted = format_recommendations_for_display(recs)

    # Store for report
    st.session_state["mitigation_recs"] = recs
    if "assessment" in st.session_state:
        st.session_state["assessment"]["mitigation_recommendations"] = recs

    st.markdown("---")

    # ── Priority Banner ───────────────────────────────────────────────────
    p_color = priority["color"]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg, {p_color}22, {p_color}11);
                border:2px solid {p_color}66; border-radius:16px;
                padding:1.25rem 2rem; margin-bottom:1.5rem; text-align:center;'>
        <span style='font-size:1.75rem;'>{priority["icon"]}</span>
        <span style='color:{p_color}; font-size:1.25rem; font-weight:700; margin-left:0.75rem;'>
            {priority["badge"]}
        </span>
        <span style='color:#94a3b8; margin-left:1rem;'>
            Recommended timeframe: <strong style='color:#f8fafc;'>{priority["timeframe"]}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Urgent Actions First ───────────────────────────────────────────────
    if "urgent_actions" in formatted:
        cat = formatted["urgent_actions"]
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.15); border:2px solid rgba(239,68,68,0.5);
                    border-radius:16px; padding:1.5rem; margin-bottom:1.5rem;'>
            <h3 style='color:#f87171; margin-top:0;'>{cat["title"]}</h3>
        """, unsafe_allow_html=True)
        for item in cat["items"]:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Regular Recommendations Grid ──────────────────────────────────────
    st.markdown("### 📋 Detailed Recommendations by Category")

    regular_cats = [k for k in formatted if k != "urgent_actions"]
    cols = st.columns(2)

    for i, cat_key in enumerate(regular_cats):
        cat = formatted[cat_key]
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:rgba(30,41,59,0.8); border:1px solid rgba(148,163,184,0.15);
                        border-radius:16px; padding:1.25rem; margin-bottom:1rem; min-height:200px;'>
                <h4 style='color:#00d4aa; margin-top:0;'>{cat["title"]}</h4>
            """, unsafe_allow_html=True)
            for item in cat["items"]:
                st.markdown(f"• {item}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Target Audience Matrix ─────────────────────────────────────────────
    st.markdown("### 🎯 Recommendations by Target Audience")

    aud1, aud2, aud3 = st.columns(3)

    with aud1:
        st.markdown("""
        <div style='background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.3);
                    border-radius:12px; padding:1.25rem;'>
            <h4 style='color:#a78bfa; margin-top:0;'>🏛️ Government / Policy Makers</h4>
        """, unsafe_allow_html=True)
        gov_recs = recs.get("policy", []) + recs.get("monitoring", [])[:2]
        for r in gov_recs[:5]:
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

    with aud2:
        st.markdown("""
        <div style='background:rgba(0,212,170,0.1); border:1px solid rgba(0,212,170,0.3);
                    border-radius:12px; padding:1.25rem;'>
            <h4 style='color:#00d4aa; margin-top:0;'>👤 Individual / Residents</h4>
        """, unsafe_allow_html=True)
        ind_recs = recs.get("public_health", [])
        for r in ind_recs[:5]:
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

    with aud3:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
                    border-radius:12px; padding:1.25rem;'>
            <h4 style='color:#fbbf24; margin-top:0;'>🏭 Industry / Businesses</h4>
        """, unsafe_allow_html=True)
        ind_biz_recs = recs.get("industrial_controls", recs.get("traffic_management", []))
        for r in ind_biz_recs[:5]:
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 **Next Step:** Visit **Page 8 — AI Environmental Report** to generate a full professional report including these recommendations.")
