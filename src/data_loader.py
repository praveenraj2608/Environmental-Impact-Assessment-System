"""
Data loading module with validation for all 3 datasets.
Provides safe loading functions with column/shape checks.

Actual file names (found in data/):
  - City_Types.csv              → has extra 'Date','City' cols; 6 pollutants + 'Type' target
  - air_quality_health_impact_data.csv → HealthImpactClass is numeric float (0-4)
  - AirQualityUCI.csv           → UTF-8-BOM + comma delimiter; -200 = missing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from utils.config import (
    CITY_TYPES_CSV, HEALTH_IMPACT_CSV, AIR_QUALITY_CSV,
    CITY_TYPE_FEATURES, CITY_TYPE_TARGET,
    HEALTH_IMPACT_TARGET, HEALTH_RECORD_ID_COL,
    AIR_QUALITY_MISSING_VALUE,
    HEALTH_IMPACT_CLASS_DISPLAY,
)

logger = logging.getLogger(__name__)


def load_city_types_dataset() -> pd.DataFrame:
    """
    Load and validate the city types classification dataset.
    Extra columns (Date, City) are dropped automatically; only the 6
    pollutant features + Type target are required.

    Returns:
        pd.DataFrame with columns: CO, NO2, SO2, O3, PM2.5, PM10, Type
    Raises:
        FileNotFoundError: If CSV file is missing.
        ValueError: If required columns are absent.
    """
    if not CITY_TYPES_CSV.exists():
        raise FileNotFoundError(
            f"City types dataset not found at: {CITY_TYPES_CSV}\n"
            "Please place 'City_Types.csv' in the 'data/' folder."
        )

    try:
        df = pd.read_csv(CITY_TYPES_CSV)
    except Exception as e:
        raise ValueError(f"Failed to read city types CSV: {e}")

    # Check required columns (features + target)
    required = CITY_TYPE_FEATURES + [CITY_TYPE_TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"City types dataset missing columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Keep only required columns (drop Date, City, etc.)
    df = df[required].copy()

    # Drop any rows with NaN in required columns
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with NaN from city types dataset")

    logger.info(f"City types dataset loaded: {df.shape}")
    return df


def load_health_impact_dataset() -> pd.DataFrame:
    """
    Load and validate the health impact dataset.
    HealthImpactClass is stored as numeric float (0.0–4.0); mapped to
    string labels (Low/Moderate/High/Severe/Very High) automatically.

    Returns:
        pd.DataFrame with HealthImpactClass as string labels.
    Raises:
        FileNotFoundError: If CSV file is missing.
        ValueError: If required columns are absent.
    """
    if not HEALTH_IMPACT_CSV.exists():
        raise FileNotFoundError(
            f"Health impact dataset not found at: {HEALTH_IMPACT_CSV}\n"
            "Please place 'air_quality_health_impact_data.csv' in the 'data/' folder."
        )

    try:
        df = pd.read_csv(HEALTH_IMPACT_CSV)
    except Exception as e:
        raise ValueError(f"Failed to read health impact CSV: {e}")

    # Required columns check
    required = [
        "RecordID", "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
        "Temperature", "Humidity", "WindSpeed",
        "RespiratoryCases", "CardiovascularCases", "HospitalAdmissions",
        "HealthImpactScore", "HealthImpactClass"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Health impact dataset missing columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Convert numeric HealthImpactClass → string labels
    if df[HEALTH_IMPACT_TARGET].dtype in [np.float64, np.int64, float, int]:
        df[HEALTH_IMPACT_TARGET] = df[HEALTH_IMPACT_TARGET].astype(int).map(HEALTH_IMPACT_CLASS_DISPLAY)
        n_unmapped = df[HEALTH_IMPACT_TARGET].isna().sum()
        if n_unmapped > 0:
            logger.warning(f"{n_unmapped} HealthImpactClass values could not be mapped; dropping those rows")
            df = df.dropna(subset=[HEALTH_IMPACT_TARGET])

    logger.info(
        f"Health impact dataset loaded: {df.shape}\n"
        f"Class distribution: {df[HEALTH_IMPACT_TARGET].value_counts().to_dict()}"
    )
    return df


def load_air_quality_dataset() -> pd.DataFrame:
    """
    Load the UCI Air Quality dataset.
    File is UTF-8-BOM + comma-delimited. -200 values are treated as NaN.
    Unnamed/empty trailing columns are dropped.
    DateTime is parsed from Date + Time columns and dataset sorted chronologically.

    Returns:
        pd.DataFrame with parsed DateTime column, -200 replaced with NaN.
    Raises:
        FileNotFoundError: If CSV file is missing.
    """
    if not AIR_QUALITY_CSV.exists():
        raise FileNotFoundError(
            f"UCI Air Quality dataset not found at: {AIR_QUALITY_CSV}\n"
            "Please place 'AirQualityUCI.csv' in the 'data/' folder."
        )

    try:
        # UTF-8-BOM encoding, comma-delimited, -200 = missing
        df = pd.read_csv(
            AIR_QUALITY_CSV,
            sep=",",
            encoding="utf-8-sig",
            na_values=["", " "],
        )
    except Exception as e:
        raise ValueError(f"Failed to read UCI Air Quality CSV: {e}")

    # Drop fully-empty/unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    # Replace -200 sentinel with NaN (as specified in dataset)
    df = df.replace(AIR_QUALITY_MISSING_VALUE, np.nan)

    # Apply median imputation for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Parse DateTime
    if "Date" in df.columns and "Time" in df.columns:
        try:
            df["DateTime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                dayfirst=True,
                errors="coerce",
            )
            df = df.dropna(subset=["DateTime"])
            df = df.sort_values("DateTime").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not parse datetime: {e}")

    logger.info(f"UCI Air Quality dataset loaded: {df.shape}")
    return df


def validate_datasets() -> Dict[str, Dict]:
    """
    Check presence and basic validity of all 3 datasets.

    Returns:
        dict: Status for each dataset with keys 'available', 'shape', 'error'.
    """
    results = {}
    for name, loader in [
        ("city_types", load_city_types_dataset),
        ("health_impact", load_health_impact_dataset),
        ("air_quality", load_air_quality_dataset),
    ]:
        try:
            df = loader()
            results[name] = {
                "available": True,
                "shape": df.shape,
                "columns": list(df.columns),
                "error": None,
            }
        except FileNotFoundError as e:
            results[name] = {"available": False, "shape": None, "error": str(e)}
        except Exception as e:
            results[name] = {"available": False, "shape": None, "error": str(e)}

    return results


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """Compute summary statistics for a dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = validate_datasets()
    for name, info in status.items():
        if info["available"]:
            print(f"✅ {name}: {info['shape']}")
        else:
            print(f"❌ {name}: {info['error']}")
