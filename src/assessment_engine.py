"""
Environmental Risk Assessment Engine.

Computes composite environmental risk score (0-100) by combining
pollution level, health impact, area type, and trend data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Dict, Optional, Tuple

from utils.config import (
    RISK_SCORE_WEIGHTS, INDUSTRIAL_FACTOR_SCORES, TREND_FACTOR_SCORES,
    RISK_LEVELS, AQI_MAX_REFERENCE, HEALTH_IMPACT_CLASSES_NUMERIC,
    POLLUTION_SEVERITY_THRESHOLDS,
)

logger = logging.getLogger(__name__)


def normalize_pollution_level(level: float, reference: float = AQI_MAX_REFERENCE) -> float:
    """
    Normalize a pollution level to 0-100 scale.

    Args:
        level: Raw pollution/AQI value.
        reference: Reference max for normalization (default: EPA AQI 150).

    Returns:
        Normalized score clamped to [0, 100].
    """
    return float(min(100.0, max(0.0, (level / reference) * 100.0)))


def get_health_impact_score(health_impact_class: str) -> float:
    """
    Map a health impact class label to a 0-100 numeric score.

    Args:
        health_impact_class: String class label (e.g., 'High', 'Severe', '2', '2.0').

    Returns:
        Numeric score in [0, 100].
    """
    from utils.config import HEALTH_IMPACT_CLASSES_NUMERIC

    label = str(health_impact_class).strip()

    # Direct match (handles 'Low','High','0','1', etc.)
    if label in HEALTH_IMPACT_CLASSES_NUMERIC:
        return float(HEALTH_IMPACT_CLASSES_NUMERIC[label])

    # Float string ('2.0' → '2')
    try:
        int_key = str(int(float(label)))
        if int_key in HEALTH_IMPACT_CLASSES_NUMERIC:
            return float(HEALTH_IMPACT_CLASSES_NUMERIC[int_key])
    except (ValueError, TypeError):
        pass

    # Fuzzy match (case-insensitive)
    for key, val in HEALTH_IMPACT_CLASSES_NUMERIC.items():
        if key.lower() in label.lower():
            return float(val)

    logger.warning(f"Unknown health impact class '{health_impact_class}', defaulting to 50")
    return 50.0


def calculate_environmental_risk_score(
    pollution_level: float,
    health_impact_class: str,
    is_industrial: bool,
    trend: str,
    aqi_reference: float = AQI_MAX_REFERENCE,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the composite Environmental Risk Score (0-100).

    Formula:
        Risk = 0.4 × Pollution_Factor
             + 0.3 × HealthImpact_Factor
             + 0.2 × Industrial_Factor
             + 0.1 × Trend_Factor

    Args:
        pollution_level: AQI or average pollutant concentration.
        health_impact_class: Predicted health impact class label.
        is_industrial: True if area classified as Industrial.
        trend: One of 'improving', 'stable', 'worsening'.
        aqi_reference: Reference value for normalizing pollution (default 150).

    Returns:
        Tuple of (risk_score [0-100], factor_breakdown dict).
    """
    # Normalize each factor to [0, 100]
    pollution_factor = normalize_pollution_level(pollution_level, aqi_reference)
    health_factor = get_health_impact_score(health_impact_class)
    city_type_str = "Industrial" if is_industrial else "Residential"
    industrial_factor = float(INDUSTRIAL_FACTOR_SCORES.get(city_type_str, 10))
    trend_str = trend.lower().strip()
    trend_factor = float(TREND_FACTOR_SCORES.get(trend_str, TREND_FACTOR_SCORES["stable"]))

    # Weighted sum
    w = RISK_SCORE_WEIGHTS
    risk_score = (
        w["pollution_factor"] * pollution_factor
        + w["health_impact_factor"] * health_factor
        + w["industrial_factor"] * industrial_factor
        + w["trend_factor"] * trend_factor
    )
    risk_score = round(float(min(100.0, max(0.0, risk_score))), 2)

    factors = {
        "pollution_factor": round(pollution_factor, 2),
        "health_impact_factor": round(health_factor, 2),
        "industrial_factor": round(industrial_factor, 2),
        "trend_factor": round(trend_factor, 2),
    }

    logger.info(
        f"Risk score: {risk_score:.2f} | "
        f"Pollution={pollution_factor:.1f}, Health={health_factor:.1f}, "
        f"Industrial={industrial_factor:.1f}, Trend={trend_factor:.1f}"
    )
    return risk_score, factors


def interpret_risk_level(score: float) -> Tuple[str, str]:
    """
    Map a risk score to a risk level and its hex color.

    Args:
        score: Composite risk score in [0, 100].

    Returns:
        Tuple of (risk_level_label, color_hex).
    """
    for level, (lo, hi, color) in RISK_LEVELS.items():
        if lo <= score <= hi:
            return level, color
    # Edge case: exactly 0 or 100
    if score <= 0:
        return "Safe", RISK_LEVELS["Safe"][2]
    return "Critical", RISK_LEVELS["Critical"][2]


def get_pollution_severity(aqi: float) -> str:
    """
    Classify AQI value into a severity category.

    Args:
        aqi: AQI or normalized pollution value.

    Returns:
        Severity label: 'Low', 'Moderate', 'High', or 'Severe'.
    """
    for severity, (lo, hi) in POLLUTION_SEVERITY_THRESHOLDS.items():
        if lo <= aqi < hi:
            return severity
    return "Severe"


def synthesize_assessment(
    city_type_pred: str,
    city_type_confidence: float,
    health_impact_pred: str,
    health_impact_confidence: float,
    pollution_level: float,
    predicted_pollution: Optional[float] = None,
    trend: str = "stable",
) -> Dict:
    """
    Synthesize outputs from all 3 models into a unified assessment.

    Args:
        city_type_pred: 'Industrial' or 'Residential'.
        city_type_confidence: Model confidence (0-1).
        health_impact_pred: Health impact class label.
        health_impact_confidence: Model confidence (0-1).
        pollution_level: Current AQI / pollutant level.
        predicted_pollution: ML-predicted next-period pollution.
        trend: 'improving', 'stable', or 'worsening'.

    Returns:
        Dict with risk_score, risk_level, risk_color, factor_breakdown,
        pollution_severity, and summary strings.
    """
    is_industrial = city_type_pred.lower() == "industrial"
    risk_score, factors = calculate_environmental_risk_score(
        pollution_level=pollution_level,
        health_impact_class=health_impact_pred,
        is_industrial=is_industrial,
        trend=trend,
    )
    risk_level, risk_color = interpret_risk_level(risk_score)
    pollution_severity = get_pollution_severity(pollution_level)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "factor_breakdown": factors,
        "city_type": city_type_pred,
        "city_type_confidence": round(city_type_confidence * 100, 1),
        "health_impact": health_impact_pred,
        "health_impact_confidence": round(health_impact_confidence * 100, 1),
        "pollution_level": round(pollution_level, 2),
        "pollution_severity": pollution_severity,
        "predicted_pollution": round(predicted_pollution, 2) if predicted_pollution else None,
        "trend": trend,
        "summary": (
            f"{risk_level} risk area — {city_type_pred} classification with "
            f"{pollution_severity.lower()} pollution levels and "
            f"{health_impact_pred.lower()} health impact."
        ),
    }


if __name__ == "__main__":
    # Example calculation from the spec
    score, factors = calculate_environmental_risk_score(
        pollution_level=127.5,  # ~85 on 0-100 scale at 150 ref
        health_impact_class="High",
        is_industrial=True,
        trend="worsening",
    )
    level, color = interpret_risk_level(score)
    print(f"Risk Score: {score:.1f} → {level} ({color})")
    print(f"Factors: {factors}")
