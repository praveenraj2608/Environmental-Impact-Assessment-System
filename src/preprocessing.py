"""
Preprocessing pipelines for all 3 datasets.
Each class provides fit/transform/fit_transform consistent with sklearn API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from utils.config import (
    CITY_TYPE_FEATURES, CITY_TYPE_TARGET,
    HEALTH_IMPACT_TARGET, HEALTH_RECORD_ID_COL,
    AIR_QUALITY_TARGET, LAG_HOURS, ROLLING_WINDOWS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


class CityTypePreprocessor:
    """
    Preprocessing pipeline for city type classification.

    Steps:
        1. Select 6 pollutant features
        2. Encode target (Industrial→0, Residential→1)
        3. StandardScaler on features
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = CITY_TYPE_FEATURES
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CityTypePreprocessor":
        """Fit scaler and label encoder."""
        X_sel = X[self.feature_names].copy()
        self.scaler.fit(X_sel)
        self.label_encoder.fit(y)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scale feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X_sel = X[self.feature_names].copy()
        return self.scaler.transform(X_sel)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(X, y)
        X_scaled = self.transform(X)
        y_encoded = self.label_encoder.transform(y)
        return X_scaled, y_encoded

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels."""
        return self.label_encoder.transform(y)

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode numeric labels back to class names."""
        return self.label_encoder.inverse_transform(y_encoded)

    def transform_single(self, inputs: dict) -> np.ndarray:
        """Transform a single sample from a dict."""
        row = [[inputs[f] for f in self.feature_names]]
        return self.scaler.transform(row)


class HealthImpactPreprocessor:
    """
    Preprocessing pipeline for health impact classification.

    Steps:
        1. Drop RecordID (non-predictive)
        2. Separate features and target
        3. Median imputation for missing values (<5%)
        4. StandardScaler on features
        5. LabelEncoder on target
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names: List[str] = []
        self._is_fitted = False

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude RecordID and target)."""
        exclude = [HEALTH_RECORD_ID_COL, HEALTH_IMPACT_TARGET]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HealthImpactPreprocessor":
        """Fit imputer, scaler, and label encoder."""
        self.feature_names = list(X.columns)
        X_imp = pd.DataFrame(self.imputer.fit_transform(X), columns=self.feature_names)
        self.scaler.fit(X_imp)
        self.label_encoder.fit(y)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Impute then scale, preserving DataFrame feature names."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X_sel = X[self.feature_names]
        X_imp = pd.DataFrame(self.imputer.transform(X_sel), columns=self.feature_names)
        return self.scaler.transform(X_imp)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform."""
        self.fit(X, y)
        X_proc = self.transform(X)
        y_encoded = self.label_encoder.transform(y)
        return X_proc, y_encoded

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Encode target."""
        return self.label_encoder.transform(y)

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode target."""
        return self.label_encoder.inverse_transform(y_encoded)

    def transform_single(self, inputs: dict) -> np.ndarray:
        """Transform a single sample from a dict."""
        row = pd.DataFrame([{f: inputs.get(f, 0) for f in self.feature_names}])
        return self.transform(row)

    @property
    def classes_(self):
        return self.label_encoder.classes_


class AirQualityPreprocessor:
    """
    Preprocessing pipeline for UCI Air Quality time-series prediction.

    Steps:
        1. Parse and sort by datetime
        2. Engineer lag features (t-1, t-7, t-24 hours)
        3. Engineer rolling statistics (mean, max, min over 24h and 7d)
        4. Add time-based features (hour, day_of_week, month)
        5. Drop rows with NaN from lag operations
        6. StandardScaler on all features
    """

    def __init__(self, target_col: str = AIR_QUALITY_TARGET):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False

    def engineer_lag_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Create lag features for a given column.

        Args:
            df: DataFrame with the target column.
            col: Column to create lags for.

        Returns:
            DataFrame with added lag columns.
        """
        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def engineer_rolling_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Create rolling statistics for a given column.

        Args:
            df: DataFrame with the target column.
            col: Column to create rolling stats for.

        Returns:
            DataFrame with rolling mean/max/min columns.
        """
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_roll_max_{window}"] = df[col].rolling(window=window, min_periods=1).max()
            df[f"{col}_roll_min_{window}"] = df[col].rolling(window=window, min_periods=1).min()
        return df

    def engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features from 'DateTime' column.

        Args:
            df: DataFrame with DateTime column.

        Returns:
            DataFrame with hour, day_of_week, month columns.
        """
        if "DateTime" in df.columns:
            df["hour"] = df["DateTime"].dt.hour
            df["day_of_week"] = df["DateTime"].dt.dayofweek
            df["month"] = df["DateTime"].dt.month
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full feature engineering pipeline.

        Args:
            df: Raw preprocessed UCI DataFrame.

        Returns:
            Tuple of (X features, y target).
        """
        df = df.copy()

        # Engineer lag and rolling features on target
        df = self.engineer_lag_features(df, self.target_col)
        df = self.engineer_rolling_features(df, self.target_col)

        # Engineer time features
        df = self.engineer_time_features(df)

        # Drop non-numeric and datetime columns
        drop_cols = ["Date", "Time", "DateTime"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Drop rows with NaN (from lag)
        df = df.dropna()

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found after engineering.")

        y = df[self.target_col].copy()
        X = df.drop(columns=[self.target_col])

        return X, y

    def fit(self, X: pd.DataFrame) -> "AirQualityPreprocessor":
        """Fit scaler on feature matrix."""
        self.feature_names = list(X.columns)
        self.scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scale feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        return self.scaler.transform(X[self.feature_names])

    def fit_transform_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Full pipeline: feature engineering → fit → transform.

        Returns:
            Tuple of (X_scaled, y, original_X_df_for_reference).
        """
        X, y = self.prepare_features(df)
        self.fit(X)
        X_scaled = self.transform(X)
        return X_scaled, y.values, X
