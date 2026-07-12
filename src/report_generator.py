"""
Report generation module with OpenAI API integration and template fallback.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
from typing import Dict, Optional

from dotenv import load_dotenv

from utils.config import API_KEY_ENV_VAR, OPENAI_MODEL, MAX_REPORT_TOKENS, API_TIMEOUT

load_dotenv()
logger = logging.getLogger(__name__)


# ─── Health Disclaimer ────────────────────────────────────────────────────────
HEALTH_DISCLAIMER = """
> ⚠️ **IMPORTANT HEALTH & MEDICAL DISCLAIMER**
> 
> Health Impact predictions are **analytical estimates** based on air quality data ONLY.
> They are **NOT** medical diagnoses, professional medical advice, or a substitute for
> consultation with qualified healthcare professionals. Do not use this report for
> medical decision-making. Always consult a licensed physician for health concerns.
"""


def build_report_sections(assessment: Dict) -> Dict[str, str]:
    """
    Structure report data into labeled sections.

    Args:
        assessment: Combined assessment dict from synthesize_assessment().

    Returns:
        Dict mapping section name to markdown content string.
    """
    now = datetime.now().strftime("%B %d, %Y at %H:%M")

    city_type = assessment.get("city_type", "Unknown")
    ct_conf = assessment.get("city_type_confidence", 0)
    health = assessment.get("health_impact", "Unknown")
    hi_conf = assessment.get("health_impact_confidence", 0)
    pollution = assessment.get("pollution_level", 0)
    severity = assessment.get("pollution_severity", "Unknown")
    risk_score = assessment.get("risk_score", 0)
    risk_level = assessment.get("risk_level", "Unknown")
    trend = assessment.get("trend", "stable").capitalize()
    predicted = assessment.get("predicted_pollution")
    factors = assessment.get("factor_breakdown", {})
    mitigation = assessment.get("mitigation_recommendations", {})

    sections = {
        "header": f"# 🌍 Environmental Impact Assessment Report\n*Generated: {now}*\n",

        "executive_summary": (
            f"## Executive Summary\n\n"
            f"This assessment evaluates environmental conditions for a **{city_type}** area "
            f"(model confidence: {ct_conf:.1f}%) with a composite Environmental Risk Score of "
            f"**{risk_score:.1f}/100 ({risk_level} level)**. "
            f"Current pollution levels are classified as **{severity}** "
            f"with a **{trend}** trend. "
            f"Health impact is predicted as **{health}** "
            f"(confidence: {hi_conf:.1f}%). "
            f"{'Immediate action is recommended.' if risk_level in ('Alert', 'Critical') else 'Continue monitoring and preventive measures.'}\n"
        ),

        "air_quality": (
            f"## Air Quality Assessment\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Current Pollution Level | {pollution:.1f} |\n"
            f"| Pollution Severity | {severity} |\n"
            f"| Pollution Trend | {trend} |\n"
            f"| Predicted Next Period | {f'{predicted:.2f}' if predicted else 'N/A'} |\n\n"
            f"Pollution levels have been assessed against WHO and EPA reference thresholds. "
            f"{'Current levels exceed safe thresholds — protective action is warranted.' if severity in ('High', 'Severe') else 'Current levels are within manageable ranges.'}\n"
        ),

        "city_classification": (
            f"## Area Classification\n\n"
            f"The ML model classifies this area as **{city_type}** with "
            f"**{ct_conf:.1f}% confidence** based on analysis of six key air pollutants "
            f"(CO, NO₂, SO₂, O₃, PM₂.₅, PM₁₀). "
            f"{'Industrial classification indicates elevated emission sources requiring enhanced monitoring.' if city_type == 'Industrial' else 'Residential classification suggests lower baseline industrial emissions.'}\n"
        ),

        "health_impact": (
            f"## Health Impact Assessment\n\n"
            f"{HEALTH_DISCLAIMER}\n\n"
            f"Predicted health impact category: **{health}** (confidence: {hi_conf:.1f}%)\n\n"
            f"{'⚠️ High-risk health impact detected. Vulnerable populations (children, elderly, people with respiratory conditions) should take immediate protective measures.' if health in ('High', 'Severe', 'Very High') else 'Health impacts remain at moderate or lower levels. Standard precautions are advised.'}\n"
        ),

        "risk_score": (
            f"## Environmental Risk Level\n\n"
            f"**Overall Risk Score: {risk_score:.1f} / 100 — {risk_level}**\n\n"
            f"| Component | Factor Value | Weight | Contribution |\n"
            f"|-----------|-------------|--------|--------------|\n"
            f"| Pollution Factor | {factors.get('pollution_factor', 0):.1f} | 40% | {factors.get('pollution_factor', 0)*0.4:.1f} |\n"
            f"| Health Impact Factor | {factors.get('health_impact_factor', 0):.1f} | 30% | {factors.get('health_impact_factor', 0)*0.3:.1f} |\n"
            f"| Industrial Factor | {factors.get('industrial_factor', 0):.1f} | 20% | {factors.get('industrial_factor', 0)*0.2:.1f} |\n"
            f"| Trend Factor | {factors.get('trend_factor', 0):.1f} | 10% | {factors.get('trend_factor', 0)*0.1:.1f} |\n"
            f"| **TOTAL** | | **100%** | **{risk_score:.1f}** |\n"
        ),

        "recommendations": _format_recommendations_section(mitigation, risk_level),

        "conclusion": (
            f"## Conclusion & Next Steps\n\n"
            f"Based on the comprehensive analysis, this {city_type} area presents a "
            f"**{risk_level}** environmental risk with a score of {risk_score:.1f}/100. "
            f"{'URGENT: Implement the critical actions listed above within 24-48 hours. Coordinate with environmental and health authorities immediately.' if risk_level == 'Critical' else ''}"
            f"{'ALERT: Begin implementing recommended mitigation strategies within 1-2 weeks. Increase monitoring frequency.' if risk_level == 'Alert' else ''}"
            f"{'Continue regular monitoring and implement preventive measures to avoid deterioration.' if risk_level in ('Safe', 'Caution') else ''}\n\n"
            f"*This report was generated by the Automated Environmental Impact Assessment System. "
            f"All predictions are model-based estimates and should be validated against ground-truth measurements.*\n"
        ),
    }
    return sections


def _format_recommendations_section(mitigation: Dict, risk_level: str) -> str:
    """Format mitigation recommendations as markdown."""
    if not mitigation:
        return "## Mitigation Recommendations\n\n*No recommendations available.*\n"

    lines = ["## Mitigation Recommendations\n"]
    emoji_map = {
        "urgent_actions": "🚨",
        "industrial_controls": "🏭",
        "traffic_management": "🚗",
        "monitoring": "📡",
        "green_infrastructure": "🌳",
        "public_health": "🫁",
        "policy": "📋",
    }
    title_map = {
        "urgent_actions": "Urgent Actions",
        "industrial_controls": "Industrial Controls",
        "traffic_management": "Traffic Management",
        "monitoring": "Monitoring",
        "green_infrastructure": "Green Infrastructure",
        "public_health": "Public Health",
        "policy": "Policy",
    }

    for category, items in mitigation.items():
        if not items:
            continue
        icon = emoji_map.get(category, "📌")
        title = title_map.get(category, category.replace("_", " ").title())
        lines.append(f"\n### {icon} {title}\n")
        for item in items:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def assemble_report(sections: Dict[str, str]) -> str:
    """
    Assemble ordered sections into a complete markdown report.

    Args:
        sections: Dict from build_report_sections().

    Returns:
        Full markdown report string.
    """
    order = [
        "header", "executive_summary", "air_quality",
        "city_classification", "health_impact", "risk_score",
        "recommendations", "conclusion"
    ]
    return "\n\n---\n\n".join(sections.get(k, "") for k in order if k in sections)


def generate_ai_report(assessment: Dict, api_key: str) -> Optional[str]:
    """
    Generate an AI-enhanced report using OpenAI GPT.

    Args:
        assessment: Assessment data dict.
        api_key: OpenAI API key.

    Returns:
        AI-generated markdown report string, or None on failure.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=API_TIMEOUT)

        # Build a compact data summary for the prompt
        summary = (
            f"City Type: {assessment.get('city_type')} ({assessment.get('city_type_confidence')}% confidence)\n"
            f"Health Impact: {assessment.get('health_impact')} ({assessment.get('health_impact_confidence')}% confidence)\n"
            f"Pollution Level: {assessment.get('pollution_level'):.1f} ({assessment.get('pollution_severity')})\n"
            f"Risk Score: {assessment.get('risk_score'):.1f}/100 ({assessment.get('risk_level')})\n"
            f"Trend: {assessment.get('trend')}\n"
        )

        prompt = f"""You are an expert environmental scientist. Based on the following air quality assessment data, write a professional, concise environmental impact report in markdown format. Include: executive summary, key findings, health implications (with medical disclaimer), and actionable recommendations. Keep it under 600 words.

Assessment Data:
{summary}

Write the report now:"""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_REPORT_TOKENS,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except ImportError:
        logger.warning("OpenAI package not installed. Falling back to template.")
        return None
    except Exception as e:
        logger.warning(f"AI report generation failed: {e}. Falling back to template.")
        return None


def generate_structured_report(assessment: Dict) -> Dict[str, str]:
    """
    Main entry point: generate environmental report with fallback logic.

    Priority:
        1. OpenAI API (if key available)
        2. Template-based (always works)

    Args:
        assessment: Full assessment dict with all model outputs.

    Returns:
        Dict with keys 'report' (markdown), 'method' ('AI' or 'template'),
        'note' (user-facing message).
    """
    api_key = os.getenv(API_KEY_ENV_VAR)
    sections = build_report_sections(assessment)

    # Try AI report first
    if api_key and api_key not in ("your_openai_api_key_here", ""):
        logger.info("Attempting AI-enhanced report generation...")
        ai_report = generate_ai_report(assessment, api_key)
        if ai_report:
            # Prepend health disclaimer if not present
            if "HEALTH" not in ai_report.upper():
                ai_report = HEALTH_DISCLAIMER + "\n\n" + ai_report
            return {
                "report": ai_report,
                "method": "AI",
                "note": f"✨ Report generated using OpenAI {OPENAI_MODEL}",
            }

    # Fallback: template report
    logger.info("Using template-based report generation.")
    report = assemble_report(sections)
    return {
        "report": report,
        "method": "template",
        "note": "📋 Report generated using structured template (AI unavailable or not configured)",
    }
