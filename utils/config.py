"""
Centralized configuration for the Environmental Impact Assessment System.
All constants, paths, thresholds, and settings live here.
"""

from pathlib import Path

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ─── Directory Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# ─── Dataset File Paths ───────────────────────────────────────────────────────
CITY_TYPES_CSV = DATA_DIR / "City_Types.csv"
HEALTH_IMPACT_CSV = DATA_DIR / "air_quality_health_impact_data.csv"
AIR_QUALITY_CSV = DATA_DIR / "AirQualityUCI.csv"

# ─── Model Artifact Paths ─────────────────────────────────────────────────────
CITY_TYPE_MODEL_DIR = MODELS_DIR / "city_type"
HEALTH_IMPACT_MODEL_DIR = MODELS_DIR / "health_impact"
AIR_QUALITY_MODEL_DIR = MODELS_DIR / "air_quality"

CITY_TYPE_MODEL_PATH = CITY_TYPE_MODEL_DIR / "random_forest_model.pkl"
CITY_TYPE_SCALER_PATH = CITY_TYPE_MODEL_DIR / "scaler.pkl"
CITY_TYPE_ENCODER_PATH = CITY_TYPE_MODEL_DIR / "label_encoder.pkl"

HEALTH_IMPACT_MODEL_PATH = HEALTH_IMPACT_MODEL_DIR / "best_model.pkl"
HEALTH_IMPACT_SCALER_PATH = HEALTH_IMPACT_MODEL_DIR / "scaler.pkl"
HEALTH_IMPACT_ENCODER_PATH = HEALTH_IMPACT_MODEL_DIR / "label_encoder.pkl"

# ANN-specific artifacts — separate from the ML pipeline so existing pkl files
# are never overwritten by a deep-learning training run.
HEALTH_IMPACT_ANN_PATH = HEALTH_IMPACT_MODEL_DIR / "health_impact_ann.pt"
HEALTH_IMPACT_ANN_HISTORY_PATH = HEALTH_IMPACT_MODEL_DIR / "ann_history.json"
HEALTH_IMPACT_ANN_PREPROCESSOR_PATH = HEALTH_IMPACT_MODEL_DIR / "ann_preprocessor.pkl"
# Architecture config saved alongside .pt so inference can reconstruct the model
HEALTH_IMPACT_ANN_CONFIG_PATH = HEALTH_IMPACT_MODEL_DIR / "ann_config.json"

AIR_QUALITY_MODEL_PATH = AIR_QUALITY_MODEL_DIR / "prediction_model.pkl"
AIR_QUALITY_SCALER_PATH = AIR_QUALITY_MODEL_DIR / "scaler.pkl"

# LSTM-specific artifacts — separate from ML pipeline; ML scaler.pkl is never overwritten
AIR_QUALITY_LSTM_MODEL_PATH   = AIR_QUALITY_MODEL_DIR / "air_quality_lstm.pt"
AIR_QUALITY_LSTM_SCALER_PATH  = AIR_QUALITY_MODEL_DIR / "lstm_scaler.pkl"
AIR_QUALITY_LSTM_CONFIG_PATH  = AIR_QUALITY_MODEL_DIR / "lstm_config.json"
AIR_QUALITY_LSTM_HISTORY_PATH = AIR_QUALITY_MODEL_DIR / "lstm_history.json"
LSTM_LOOK_BACK = 24  # default sliding-window length (hours); configurable

# ─── Dataset 1: City Type Classification ──────────────────────────────────────
CITY_TYPE_FEATURES = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]
CITY_TYPE_TARGET = "Type"
CITY_TYPE_LABELS = {"Industrial": 0, "Residential": 1}
CITY_TYPE_LABEL_NAMES = {0: "Industrial", 1: "Residential"}

# Typical pollutant ranges for UI display (in µg/m³ or ppm)
POLLUTANT_RANGES = {
    "CO": {"min": 0.0, "max": 30.0, "unit": "mg/m³", "typical": 1.0},
    "NO2": {"min": 0.0, "max": 400.0, "unit": "µg/m³", "typical": 40.0},
    "SO2": {"min": 0.0, "max": 500.0, "unit": "µg/m³", "typical": 20.0},
    "O3": {"min": 0.0, "max": 300.0, "unit": "µg/m³", "typical": 60.0},
    "PM2.5": {"min": 0.0, "max": 500.0, "unit": "µg/m³", "typical": 25.0},
    "PM10": {"min": 0.0, "max": 600.0, "unit": "µg/m³", "typical": 50.0},
}

# ─── Dataset 2: Health Impact Classification ──────────────────────────────────
HEALTH_IMPACT_TARGET = "HealthImpactClass"
HEALTH_RECORD_ID_COL = "RecordID"
HEALTH_IMPACT_FEATURES = [
    "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
    "Temperature", "Humidity", "WindSpeed",
    "RespiratoryCases", "CardiovascularCases", "HospitalAdmissions",
    "HealthImpactScore"
]

# Health impact class numeric mapping — actual class values are 0,1,2,3,4 (floats)
# Map numeric class → display name AND numeric score for risk engine
HEALTH_IMPACT_CLASSES_NUMERIC = {
    "0": 20, "1": 40, "2": 60, "3": 80, "4": 100,
    "Low": 20, "Moderate": 40, "High": 60, "Severe": 80, "Very High": 100,
}
# Map numeric class to human-readable label for display
HEALTH_IMPACT_CLASS_DISPLAY = {
    2: "Low", 1: "Moderate", 3: "High", 4: "Severe", 0: "Very High",
    "2.0": "Low", "1.0": "Moderate", "3.0": "High", "4.0": "Severe", "0.0": "Very High",
    "2": "Low", "1": "Moderate", "3": "High", "4": "Severe", "0": "Very High",
}

# ─── Dataset 3: Air Quality Time Series (UCI) ─────────────────────────────────
AIR_QUALITY_TARGET = "CO(GT)"  # Primary target; fallback to NO2(GT)
AIR_QUALITY_TARGET_FALLBACK = "NO2(GT)"
AIR_QUALITY_DATE_COL = "Date"
AIR_QUALITY_TIME_COL = "Time"
AIR_QUALITY_MISSING_VALUE = -200  # Sentinel for missing in raw data

# Lag feature windows (hours)
LAG_HOURS = [1, 7, 24]
# Rolling statistics windows (hours)
ROLLING_WINDOWS = [24, 168]  # 1 day, 7 days

# ─── Environmental Risk Score ─────────────────────────────────────────────────
RISK_SCORE_WEIGHTS = {
    "pollution_factor": 0.4,
    "health_impact_factor": 0.3,
    "industrial_factor": 0.2,
    "trend_factor": 0.1,
}

INDUSTRIAL_FACTOR_SCORES = {"Industrial": 40, "Residential": 10}
TREND_FACTOR_SCORES = {"improving": 30, "stable": 50, "worsening": 80}

RISK_LEVELS = {
    "Safe": (0, 25, "#22c55e"),
    "Caution": (26, 50, "#eab308"),
    "Alert": (51, 75, "#f97316"),
    "Critical": (76, 100, "#ef4444"),
}

# ─── Pollution Severity Thresholds (AQI-based) ────────────────────────────────
POLLUTION_SEVERITY_THRESHOLDS = {
    "Low": (0, 50),
    "Moderate": (50, 100),
    "High": (100, 150),
    "Severe": (150, 999),
}
AQI_MAX_REFERENCE = 150.0  # Reference for normalizing to 0-100

# ─── API Configuration ────────────────────────────────────────────────────────
API_KEY_ENV_VAR = "ENV_AI_API_KEY"
API_TIMEOUT = 30
OPENAI_MODEL = "gpt-4o-mini"
MAX_REPORT_TOKENS = 1500

# ─── UI Color Scheme ──────────────────────────────────────────────────────────
COLORS = {
    "primary": "#00d4aa",
    "secondary": "#7c3aed",
    "accent": "#f59e0b",
    "danger": "#ef4444",
    "success": "#22c55e",
    "warning": "#f97316",
    "bg_dark": "#0f172a",
    "bg_card": "#1e293b",
    "text_light": "#f8fafc",
    "text_muted": "#94a3b8",
}

# ─── Hyperparameter Search Space ──────────────────────────────────────────────
CITY_TYPE_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

CITY_TYPE_N_ITER = 20  # RandomizedSearchCV iterations

# ─── Model Display Names ──────────────────────────────────────────────────────
MODEL_DISPLAY_NAMES = {
    "RandomForestClassifier": "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "XGBClassifier": "XGBoost",
    "XGBRegressor": "XGBoost Regressor",
    "RandomForestRegressor": "Random Forest Regressor",
    "ANNClassifier": "ANN (Deep Learning)",
}
