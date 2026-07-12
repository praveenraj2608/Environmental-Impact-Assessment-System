"""
Input validation utilities with user-friendly error messages.
"""

from typing import Dict, List, Optional, Tuple, Any
from utils.config import POLLUTANT_RANGES, HEALTH_IMPACT_FEATURES


def validate_positive_float(value: Any, field_name: str) -> Tuple[bool, str]:
    """
    Validate that a value is a positive float.

    Args:
        value: Value to validate.
        field_name: Name of the field (for error messages).

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"'{field_name}' must be a number, got: {value}"

    if v < 0:
        return False, f"'{field_name}' must be ≥ 0, got: {v:.2f}"

    return True, ""


def validate_pollutant_inputs(inputs: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate the 6 pollutant inputs for city type prediction.

    Args:
        inputs: Dict mapping pollutant name to value.

    Returns:
        Tuple of (all_valid, list_of_error_messages).
    """
    errors = []
    required = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]

    for pollutant in required:
        if pollutant not in inputs:
            errors.append(f"Missing required pollutant: {pollutant}")
            continue

        val = inputs[pollutant]
        valid, msg = validate_positive_float(val, pollutant)
        if not valid:
            errors.append(msg)
            continue

        # Range check
        if pollutant in POLLUTANT_RANGES:
            rng = POLLUTANT_RANGES[pollutant]
            if val > rng["max"] * 2:  # Soft upper limit (2× max typical)
                errors.append(
                    f"⚠️ {pollutant} value {val:.1f} seems unusually high. "
                    f"Typical max: {rng['max']} {rng['unit']}"
                )

    return len(errors) == 0, errors


def validate_health_inputs(inputs: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate health impact model inputs.

    Args:
        inputs: Dict mapping feature name to value.

    Returns:
        Tuple of (all_valid, list_of_error_messages).
    """
    errors = []

    for feature in HEALTH_IMPACT_FEATURES:
        if feature not in inputs:
            errors.append(f"Missing required field: {feature}")
            continue

        valid, msg = validate_positive_float(inputs[feature], feature)
        if not valid:
            errors.append(msg)

    # Additional semantic checks
    if "AQI" in inputs and 0 <= float(inputs.get("AQI", 0)) > 500:
        errors.append("AQI must be between 0 and 500")

    if "Humidity" in inputs:
        h = float(inputs.get("Humidity", 0))
        if not (0 <= h <= 100):
            errors.append(f"Humidity must be between 0% and 100%, got: {h}")

    if "Temperature" in inputs:
        t = float(inputs.get("Temperature", 0))
        if not (-50 <= t <= 60):
            errors.append(f"Temperature must be between -50°C and 60°C, got: {t}")

    return len(errors) == 0, errors


def validate_risk_score_inputs(
    pollution_level: float,
    health_impact: str,
    city_type: str,
    trend: str,
) -> Tuple[bool, List[str]]:
    """
    Validate inputs for environmental risk score calculation.

    Args:
        pollution_level: AQI or pollutant level value.
        health_impact: Health impact class string.
        city_type: 'Industrial' or 'Residential'.
        trend: 'improving', 'stable', or 'worsening'.

    Returns:
        Tuple of (all_valid, list_of_error_messages).
    """
    errors = []

    valid, msg = validate_positive_float(pollution_level, "pollution_level")
    if not valid:
        errors.append(msg)

    if city_type not in ("Industrial", "Residential"):
        errors.append(f"city_type must be 'Industrial' or 'Residential', got: {city_type}")

    if trend not in ("improving", "stable", "worsening"):
        errors.append(f"trend must be 'improving', 'stable', or 'worsening', got: {trend}")

    return len(errors) == 0, errors


def format_validation_errors(errors: List[str]) -> str:
    """
    Format a list of validation errors into a user-friendly string.

    Args:
        errors: List of error messages.

    Returns:
        Formatted error string.
    """
    if not errors:
        return ""
    if len(errors) == 1:
        return f"❌ {errors[0]}"
    lines = ["❌ Please fix the following issues:"]
    for err in errors:
        lines.append(f"  • {err}")
    return "\n".join(lines)
