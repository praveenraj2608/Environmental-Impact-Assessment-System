"""
Mitigation Recommendation Engine.

Rule-based recommendation generation based on pollution level,
city type, health impact, and risk level.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# ─── Recommendation Knowledge Base ───────────────────────────────────────────

INDUSTRIAL_RECOMMENDATIONS = {
    "Safe": [
        "Maintain existing emission control systems and scheduled maintenance.",
        "Continue regular stack emission monitoring as per compliance requirements.",
        "Encourage use of cleaner fuels in production processes.",
    ],
    "Caution": [
        "Review and tighten emission permits for major industrial facilities.",
        "Implement continuous emission monitoring systems (CEMS) at all stacks.",
        "Promote adoption of Best Available Techniques (BAT) for emission reduction.",
        "Establish industry-wide voluntary emission reduction targets.",
    ],
    "Alert": [
        "Mandate immediate emission audits for top 10 industrial polluters.",
        "Enforce stricter industrial discharge limits under Clean Air Act provisions.",
        "Install advanced scrubbers and filtration systems at major facilities.",
        "Reduce production output by 20-30% during high pollution episodes.",
        "Issue industrial zone pollution warnings to nearby communities.",
    ],
    "Critical": [
        "🚨 EMERGENCY: Suspend operations at highest-emitting facilities immediately.",
        "Activate emergency industrial pollution response protocol.",
        "Require real-time reporting from all industrial emitters.",
        "Enforce mandatory pollution controls before resuming full operations.",
        "Consider temporary industrial output restrictions.",
    ],
}

TRAFFIC_RECOMMENDATIONS = {
    "Safe": [
        "Promote carpooling and public transit usage among residents.",
        "Encourage shift to electric or hybrid vehicles through incentives.",
    ],
    "Caution": [
        "Implement congestion pricing during peak traffic hours.",
        "Expand bicycle lanes and pedestrian pathways.",
        "Introduce emission testing requirements for older vehicles.",
        "Promote flexible work-from-home arrangements to reduce commute traffic.",
    ],
    "Alert": [
        "Restrict high-emission diesel vehicles in urban centres.",
        "Create low-emission zones with mandatory compliance.",
        "Increase frequency of public transit to reduce private vehicle use.",
        "Suspend non-essential construction vehicle operations during peak pollution.",
    ],
    "Critical": [
        "🚨 Enforce emergency vehicle traffic restrictions.",
        "Ban all non-essential vehicle traffic in the most polluted zones.",
        "Deploy additional rapid transit capacity immediately.",
        "Coordinate with police for active traffic pollution enforcement.",
    ],
}

MONITORING_RECOMMENDATIONS = {
    "Safe": [
        "Maintain current air quality monitoring station network.",
        "Continue hourly data reporting to public air quality index platforms.",
    ],
    "Caution": [
        "Deploy additional low-cost sensor nodes in residential neighborhoods.",
        "Increase monitoring frequency to 30-minute intervals.",
        "Establish real-time public dashboard for air quality data.",
        "Conduct community air quality awareness workshops.",
    ],
    "Alert": [
        "Activate full monitoring network at all stations simultaneously.",
        "Deploy mobile air quality labs to high-risk areas.",
        "Establish hourly briefings for local government emergency teams.",
        "Issue air quality alerts via SMS and emergency broadcast systems.",
    ],
    "Critical": [
        "🚨 Activate emergency air quality monitoring protocol.",
        "Deploy all available monitoring assets for 24/7 coverage.",
        "Coordinate with national/regional environmental agencies.",
        "Establish incident command center for pollution emergency response.",
    ],
}

GREEN_INFRASTRUCTURE_RECOMMENDATIONS = {
    "Safe": [
        "Plant 1,000+ trees annually as part of urban greening initiatives.",
        "Develop rooftop gardens and green corridors.",
        "Create community parks with native vegetation.",
    ],
    "Caution": [
        "Fast-track urban forestry programs targeting pollution hotspots.",
        "Install green barriers (hedges, trees) along major roadways.",
        "Develop wetlands and natural filtration zones.",
        "Implement green building codes requiring rooftop vegetation.",
    ],
    "Alert": [
        "Emergency tree planting campaign in highest-pollution zones.",
        "Install biofiltration walls in residential areas near industrial zones.",
        "Create green buffer zones between industrial and residential areas.",
        "Fund rapid deployment of urban air purification installations.",
    ],
    "Critical": [
        "🚨 Immediate installation of HEPA air filtration units in public spaces.",
        "Deploy portable air purifiers in schools and hospitals.",
        "Accelerate industrial buffer zone creation with tall vegetation.",
        "Commission large-scale moss/green wall installations on key buildings.",
    ],
}

PUBLIC_HEALTH_RECOMMENDATIONS = {
    "Safe": [
        "Maintain public awareness of current air quality index readings.",
        "Encourage outdoor physical activity during low-pollution hours.",
    ],
    "Caution": [
        "Advise sensitive groups (children, elderly, asthmatics) to limit prolonged outdoor exposure.",
        "Distribute N95/FFP2 masks to healthcare facilities and schools.",
        "Update school outdoor activity policies based on AQI thresholds.",
        "Issue advisory to residents to keep windows closed during peak pollution hours.",
    ],
    "Alert": [
        "Issue public health advisory: limit outdoor activity for all residents.",
        "Provide free N95 masks at community centres, schools, and transit hubs.",
        "Activate respiratory clinic capacity surge protocols.",
        "Advise vulnerable populations to shelter in place.",
        "Alert hospitals to prepare for increased respiratory and cardiovascular admissions.",
    ],
    "Critical": [
        "🚨 HEALTH EMERGENCY: Order residents to stay indoors with sealed ventilation.",
        "Activate emergency health response teams and field medical units.",
        "Open emergency clean-air shelters for vulnerable populations.",
        "Issue mandatory outdoor mask orders for the affected areas.",
        "Coordinate with hospitals to defer elective procedures and prepare ICU capacity.",
    ],
}

POLICY_RECOMMENDATIONS = {
    "Safe": [
        "Continue enforcing existing environmental regulations.",
        "Invest in long-term renewable energy transition programs.",
    ],
    "Caution": [
        "Review and update emission standards to align with WHO 2021 guidelines.",
        "Introduce carbon pricing mechanisms for major industrial emitters.",
        "Fund research into cleaner production technologies.",
    ],
    "Alert": [
        "Fast-track legislative changes to strengthen air quality standards.",
        "Declare pollution hotspots as environmental priority zones.",
        "Increase environmental enforcement staffing by 50%.",
        "Mandate environmental impact assessments for all new industrial projects.",
    ],
    "Critical": [
        "🚨 Declare environmental emergency in affected area.",
        "Invoke emergency executive powers for immediate pollution control.",
        "Suspend all new industrial permit approvals pending pollution review.",
        "Coordinate national and international emergency environmental response.",
    ],
}


def generate_recommendations(
    pollution_level: float,
    city_type: str,
    health_impact: str,
    risk_level: str,
) -> Dict[str, List[str]]:
    """
    Generate categorized, prioritized environmental mitigation recommendations.

    Args:
        pollution_level: Current AQI or pollution concentration.
        city_type: 'Industrial' or 'Residential'.
        health_impact: Health impact class label (e.g., 'High', 'Severe').
        risk_level: 'Safe', 'Caution', 'Alert', or 'Critical'.

    Returns:
        Dict with recommendation categories as keys and lists of strings as values.
    """
    rl = risk_level if risk_level in ["Safe", "Caution", "Alert", "Critical"] else "Caution"

    recs = {
        "monitoring": MONITORING_RECOMMENDATIONS.get(rl, []),
        "green_infrastructure": GREEN_INFRASTRUCTURE_RECOMMENDATIONS.get(rl, []),
        "public_health": PUBLIC_HEALTH_RECOMMENDATIONS.get(rl, []),
        "policy": POLICY_RECOMMENDATIONS.get(rl, []),
    }

    # Add industrial controls for Industrial areas
    if city_type.lower() == "industrial":
        recs["industrial_controls"] = INDUSTRIAL_RECOMMENDATIONS.get(rl, [])
    else:
        recs["traffic_management"] = TRAFFIC_RECOMMENDATIONS.get(rl, [])

    # Add urgent actions for Alert/Critical
    if rl in ("Alert", "Critical"):
        urgent = []
        if rl == "Critical":
            urgent.extend([
                "🚨 DECLARE ENVIRONMENTAL EMERGENCY immediately.",
                "Evacuate sensitive populations from highest-risk zones.",
                "Activate multi-agency emergency response coordination.",
            ])
        else:
            urgent.extend([
                "⚠️ Issue public air quality alert across all channels.",
                "Activate emergency pollution reduction protocol.",
                "Coordinate immediate response with environmental agencies.",
            ])
        recs["urgent_actions"] = urgent

    logger.info(f"Generated recommendations: {list(recs.keys())} for risk level '{rl}'")
    return recs


def get_recommendation_priority(risk_level: str) -> Dict[str, str]:
    """
    Return display priority metadata for recommendations.

    Returns:
        Dict with 'color', 'badge', 'timeframe' for the given risk level.
    """
    priority_map = {
        "Safe": {
            "color": "#22c55e",
            "badge": "🟢 Low Priority",
            "timeframe": "Long-term (6-12 months)",
            "icon": "✅",
        },
        "Caution": {
            "color": "#eab308",
            "badge": "🟡 Medium Priority",
            "timeframe": "Short-term (1-3 months)",
            "icon": "⚠️",
        },
        "Alert": {
            "color": "#f97316",
            "badge": "🟠 High Priority",
            "timeframe": "Immediate (1-2 weeks)",
            "icon": "🔔",
        },
        "Critical": {
            "color": "#ef4444",
            "badge": "🔴 CRITICAL — Act Now",
            "timeframe": "EMERGENCY (24-48 hours)",
            "icon": "🚨",
        },
    }
    return priority_map.get(risk_level, priority_map["Caution"])


def format_recommendations_for_display(
    recommendations: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    Attach display metadata to each recommendation category.

    Returns:
        Dict with category → {title, icon, items}.
    """
    category_meta = {
        "urgent_actions": {"title": "⚡ Urgent Actions Required", "icon": "🚨"},
        "industrial_controls": {"title": "🏭 Industrial Emission Controls", "icon": "🔧"},
        "traffic_management": {"title": "🚗 Traffic Management", "icon": "🛣️"},
        "monitoring": {"title": "📡 Air Quality Monitoring", "icon": "📊"},
        "green_infrastructure": {"title": "🌳 Green Infrastructure", "icon": "🌱"},
        "public_health": {"title": "🫁 Public Health Precautions", "icon": "💊"},
        "policy": {"title": "📋 Policy Recommendations", "icon": "⚖️"},
    }

    result = {}
    for cat, items in recommendations.items():
        meta = category_meta.get(cat, {"title": cat.replace("_", " ").title(), "icon": "📌"})
        result[cat] = {
            "title": meta["title"],
            "icon": meta["icon"],
            "items": items,
        }
    return result
