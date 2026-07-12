"""
Page 8 — AI Environmental Report
Professional report generation with OpenAI API + template fallback.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from datetime import datetime

from src.report_generator import generate_structured_report
from src.mitigation_engine import generate_recommendations
from src.assessment_engine import synthesize_assessment

st.set_page_config(page_title="AI Environmental Report — EIA", page_icon="📋", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <h1 class='page-title'>📋 AI Environmental Report</h1>
    <p class='page-subtitle'>
        Generate a professional environmental impact report — powered by AI or structured template
    </p>
</div>
""", unsafe_allow_html=True)

# ─── API Key Status ───────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
from utils.config import API_KEY_ENV_VAR

load_dotenv()
api_key = os.getenv(API_KEY_ENV_VAR)
ai_available = api_key and api_key not in ("your_openai_api_key_here", "")

col_status, col_info = st.columns([1, 3])
with col_status:
    if ai_available:
        st.markdown("""
        <div style='background:rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.4);
                    border-radius:12px; padding:1rem; text-align:center;'>
            <div style='font-size:2rem;'>✨</div>
            <div style='color:#22c55e; font-weight:700;'>AI Available</div>
            <div style='color:#94a3b8; font-size:0.8rem;'>OpenAI GPT-4o-mini</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(148,163,184,0.1); border:1px solid rgba(148,163,184,0.3);
                    border-radius:12px; padding:1rem; text-align:center;'>
            <div style='font-size:2rem;'>📋</div>
            <div style='color:#94a3b8; font-weight:700;'>Template Mode</div>
            <div style='color:#64748b; font-size:0.8rem;'>Set ENV_AI_API_KEY for AI</div>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    if not ai_available:
        st.info(
            "**AI reports are optional.** The template-based report is fully professional and complete. "
            "To enable AI: create a `.env` file with `ENV_AI_API_KEY=your_openai_api_key`"
        )
    else:
        st.success("AI report generation is enabled. Reports will be enriched with GPT-4o-mini analysis.")

st.markdown("---")

# ─── Assessment Configuration ─────────────────────────────────────────────────
has_assessment = "assessment" in st.session_state

if has_assessment:
    assessment = st.session_state["assessment"]
    st.success("✅ Assessment data loaded from session. You can generate the report below.")
else:
    st.info("ℹ️ No assessment in session. Configure parameters below or run Pages 3–6 first.")

with st.expander("📥 Report Configuration", expanded=not has_assessment):
    col1, col2 = st.columns(2)
    with col1:
        city_type = st.selectbox(
            "Area Type",
            ["Industrial", "Residential"],
            index=0 if st.session_state.get("assessment", {}).get("city_type", "Industrial") == "Industrial" else 1,
        )
        ct_conf = st.slider("City Type Confidence", 0.0, 1.0,
                             float(st.session_state.get("assessment", {}).get("city_type_confidence", 0.8) / 100
                                   if st.session_state.get("assessment", {}).get("city_type_confidence", 1) > 1
                                   else st.session_state.get("assessment", {}).get("city_type_confidence", 0.8)), 0.01)
        health_classes = ["Low", "Moderate", "High", "Severe", "Very High"]
        hi_default = st.session_state.get("assessment", {}).get("health_impact", "Moderate")
        health_impact = st.selectbox(
            "Health Impact",
            health_classes,
            index=health_classes.index(hi_default) if hi_default in health_classes else 1,
        )
        hi_conf = st.slider("Health Impact Confidence", 0.0, 1.0,
                             float(st.session_state.get("assessment", {}).get("health_impact_confidence", 0.7) / 100
                                   if st.session_state.get("assessment", {}).get("health_impact_confidence", 1) > 1
                                   else st.session_state.get("assessment", {}).get("health_impact_confidence", 0.7)), 0.01)
    with col2:
        pollution_level = st.number_input(
            "Pollution Level",
            0.0, 500.0,
            float(st.session_state.get("assessment", {}).get("pollution_level", 100.0)), 1.0,
        )
        trend = st.selectbox(
            "Pollution Trend",
            ["improving", "stable", "worsening"],
            index=["improving","stable","worsening"].index(
                st.session_state.get("assessment", {}).get("trend", "stable")
            ),
        )

    # Build assessment if not from session
    if not has_assessment:
        assessment = synthesize_assessment(
            city_type_pred=city_type,
            city_type_confidence=ct_conf,
            health_impact_pred=health_impact,
            health_impact_confidence=hi_conf,
            pollution_level=pollution_level,
            trend=trend,
        )

# ─── Generate Report Button ────────────────────────────────────────────────────
gen_btn = st.button("📋 Generate Environmental Report", use_container_width=True)

if gen_btn:
    # Ensure mitigation recommendations are included
    if "mitigation_recommendations" not in assessment:
        recs = generate_recommendations(
            assessment.get("pollution_level", 100),
            assessment.get("city_type", "Industrial"),
            assessment.get("health_impact", "Moderate"),
            assessment.get("risk_level", "Caution"),
        )
        assessment["mitigation_recommendations"] = recs

    with st.spinner("Generating environmental report... This may take a few seconds..."):
        result = generate_structured_report(assessment)

    report_text = result["report"]
    method = result["method"]
    note = result["note"]

    # Store for reference
    st.session_state["generated_report"] = report_text

    st.markdown("---")
    st.markdown(f"### 📋 Environmental Impact Assessment Report")

    # Method badge
    method_color = "#22c55e" if method == "AI" else "#7c3aed"
    method_icon = "✨" if method == "AI" else "📋"
    st.markdown(f"""
    <div style='background:rgba(30,41,59,0.8); border:1px solid {method_color}44;
                border-radius:12px; padding:0.75rem 1.25rem; margin-bottom:1rem;
                display:flex; align-items:center; gap:0.75rem;'>
        <span style='font-size:1.5rem;'>{method_icon}</span>
        <span style='color:{method_color}; font-weight:600;'>{note}</span>
    </div>
    """, unsafe_allow_html=True)

    # Report display
    with st.container():
        st.markdown(report_text)

    st.markdown("---")

    # ── Export Options ─────────────────────────────────────────────────────
    st.markdown("### 💾 Export Options")
    exp1, exp2, exp3 = st.columns(3)

    # Markdown download
    with exp1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download as Markdown",
            data=report_text,
            file_name=f"environmental_report_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Text download
    with exp2:
        # Strip markdown for plain text
        import re
        plain = re.sub(r"[#*_>`\-]", "", report_text).strip()
        st.download_button(
            label="📄 Download as Text",
            data=plain,
            file_name=f"environmental_report_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # HTML download
    with exp3:
        html_report = f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <title>Environmental Impact Assessment Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width:900px; margin:2rem auto; 
                color:#1e293b; line-height:1.7; padding:0 1.5rem; }}
        h1,h2,h3 {{ color:#0f172a; }}
        table {{ border-collapse:collapse; width:100%; }}
        th,td {{ border:1px solid #e2e8f0; padding:0.75rem; text-align:left; }}
        th {{ background:#f1f5f9; }}
        blockquote {{ border-left:4px solid #00d4aa; padding:0.5rem 1rem; color:#475569; }}
    </style>
</head>
<body>
<pre style='white-space:pre-wrap; font-family:inherit;'>{report_text}</pre>
</body>
</html>"""
        st.download_button(
            label="🌐 Download as HTML",
            data=html_report,
            file_name=f"environmental_report_{timestamp}.html",
            mime="text/html",
            use_container_width=True,
        )

    # Save to reports directory
    try:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        report_path = reports_dir / f"report_{timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        st.success(f"✅ Report also saved to: `reports/report_{timestamp}.md`")
    except Exception as e:
        pass  # Silent - download is the primary export

# ─── Footer Disclaimer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
            border-radius:12px; padding:1rem; font-size:0.85rem; color:#fca5a5; text-align:center;'>
    ⚠️ <strong>Health & Medical Disclaimer:</strong> All health impact predictions in this report are 
    analytical estimates based on air quality data only. They are NOT medical diagnoses and should not 
    replace consultation with qualified healthcare professionals. For environmental emergencies, 
    contact your local environmental protection authority.
</div>
""", unsafe_allow_html=True)
