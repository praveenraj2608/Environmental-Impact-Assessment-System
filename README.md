# 🌍 Automated Environmental Impact Assessment System

A college-level Data Science/ML project built with **Streamlit** that analyzes air pollution data, predicts environmental classifications, assesses health impacts, and recommends mitigation strategies using trained ML models.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Setup & Installation](#setup--installation)
- [Training Models](#training-models)
- [Running the App](#running-the-app)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Application Pages](#application-pages)
- [Environment Variables](#environment-variables)

---

## Overview

| Feature | Details |
|---------|---------|
| ML Models | 3 (Classification × 2 + Regression × 1) |
| Datasets | City Types (52,704 rows) · Health Impact · UCI Air Quality |
| App Pages | 8 interactive Streamlit pages |
| AI Reports | Optional OpenAI GPT-4o-mini integration |
| Fallback | Template-based reports always work |

---

## Setup & Installation

### Prerequisites
- Python 3.8+ (tested on 3.10)
- pip

### 1. Clone / Navigate to project directory
```bash
cd "d:\New folder (2)"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place datasets in `data/` folder
```
data/
├── city_types_dataset.csv
├── health_impact_dataset.csv
└── UCI_AirQuality.csv
```

### 4. (Optional) Configure AI API key
```bash
copy .env.example .env
# Edit .env and add your OpenAI API key
```

---

## Training Models

Run each training script in order:

```bash
# 1. City Type Classification (Random Forest)
python src/city_type_model.py

# 2. Health Impact Classification (4 models compared)
python src/health_impact_model.py

# 3. Air Quality Regression (XGBoost / RF)
python src/air_quality_model.py
```

Models are saved to `models/` directory. Training logs saved to `logs/`.

---

## Running the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## Architecture

```
environmental-impact-assessment/
├── app.py                    # Streamlit entry point
├── pages/                    # 8 app pages (auto-discovered)
├── src/                      # ML & business logic
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── city_type_model.py    # Training script
│   ├── health_impact_model.py
│   ├── air_quality_model.py
│   ├── assessment_engine.py  # Risk scoring
│   ├── mitigation_engine.py  # Recommendations
│   └── report_generator.py  # OpenAI + template
├── utils/
│   ├── config.py             # All constants & paths
│   ├── validators.py         # Input validation
│   ├── visualization.py      # Plotly wrappers
│   └── model_utils.py        # Model I/O
├── models/                   # Saved model .pkl files
├── data/                     # Input datasets
└── reports/                  # Generated reports
```

---

## Model Performance

| Model | Task | Algorithm | Accuracy | Notes |
|-------|------|-----------|----------|-------|
| City Type | Binary Classification | Random Forest | ~98.99% | RandomizedSearchCV tuned |
| Health Impact | Multi-class | XGBoost (best of 4) | See logs | Compared vs LR, DT, RF |
| Air Quality | Regression | XGBoost Regressor | R² tracked | Time-based 80/20 split |

Performance metrics saved to `logs/` as JSON files after training.

---

## Application Pages

| Page | Purpose |
|------|---------|
| 🏠 Home | Dashboard with metrics and architecture |
| 📊 Dataset Insights | EDA for all 3 datasets |
| 🏭 City Type Prediction | Industrial vs Residential classification |
| 🫁 Health Impact | Multi-class health risk prediction |
| 📈 Air Quality Analysis | Historical trends and forecasting |
| 🔍 Environmental Assessment | Composite risk score (0-100) |
| 💡 Mitigation Recommendations | Tailored action recommendations |
| 📋 AI Environmental Report | Professional report generation |

---

## Environmental Risk Score Formula

```
Risk_Score = 0.4 × Pollution_Factor
           + 0.3 × HealthImpact_Factor
           + 0.2 × Industrial_Factor
           + 0.1 × Trend_Factor

Risk Levels:
  0-25:  Safe    (🟢)
  26-50: Caution (🟡)
  51-75: Alert   (🟠)
  76-100: Critical (🔴)
```

---

## Environment Variables

Create `.env` from template:

```bash
copy .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `ENV_AI_API_KEY` | Optional | OpenAI API key for AI reports |
| `DEBUG_MODE` | Optional | Set to `True` for verbose logging |

---

## ⚠️ Disclaimers

- Health impact predictions are **analytical estimates** only — NOT medical diagnoses
- Always consult qualified healthcare professionals for medical decisions
- Model predictions are based on historical data and may not reflect current conditions
- Do not use this application for emergency medical decision-making

---

*Built with Streamlit · scikit-learn · XGBoost · Plotly · OpenAI*  
*Version 1.0 · July 2026*
