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
| Models | 5 ML Models + 2 Deep Learning Models |
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

You can train all models (Machine Learning + Deep Learning) sequentially using the master script:

```bash
python train_all_models.py
```

Alternatively, you can run individual scripts:

```bash
# 1. City Type Classification (Random Forest)
python src/city_type_model.py

# 2. Health Impact ML Models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
python src/health_impact_model.py

# 3. Health Impact DL Model (PyTorch ANN)
python src/health_impact_dl_model.py

# 4. Air Quality ML Models (XGBoost, Random Forest)
python src/air_quality_model.py

# 5. Air Quality DL Model (PyTorch LSTM)
python src/air_quality_lstm_model.py

# 6. Air Quality Model Comparison
python src/air_quality_comparison.py
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
├── src/                      # ML & DL business logic
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── city_type_model.py    # City Type training (RF)
│   ├── health_impact_model.py # Health Impact ML training
│   ├── health_impact_dl_model.py # Health Impact PyTorch ANN training
│   ├── predict_health_impact.py  # Health Impact ANN inference
│   ├── air_quality_model.py   # Air Quality ML training (XGBoost/RF)
│   ├── air_quality_lstm_model.py # Air Quality PyTorch LSTM training
│   ├── air_quality_comparison.py # Air Quality model comparison
│   ├── assessment_engine.py  # Risk scoring
│   ├── mitigation_engine.py  # Recommendations
│   └── report_generator.py   # OpenAI + template
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

| Model | Task | Algorithm / Framework | Accuracy / Metric | Notes |
|-------|------|-----------------------|-------------------|-------|
| City Type | Binary Classification | Random Forest (scikit-learn) | ~98.99% Accuracy | RandomizedSearchCV tuned |
| Health Impact (ML) | Multi-class Classification | Random Forest (scikit-learn) | ~96.50% Accuracy | Best of 4 ML models (compared vs LR, DT, XGBoost) |
| Health Impact (DL) | Multi-class Classification | Artificial Neural Network (PyTorch) | ~95.01% Accuracy | Custom PyTorch ANN |
| Air Quality (ML) | Time-series Regression | Random Forest (scikit-learn) | R² ~0.84 | Time-based 80/20 split. Best ML model is auto-selected |
| Air Quality (DL) | Time-series Regression | Stacked LSTM (PyTorch) | R² ~0.64 | 24h sliding window sequence forecasting |

Performance metrics saved to `logs/` as JSON files after training.

---

## 🧠 Deep Learning Architectures

### 1. Health Impact ANN (PyTorch)
A fully connected Artificial Neural Network built using PyTorch (`torch.nn`) to predict the health impact index (5 classes):
* **Input Layer**: 13 features (air quality metrics, demographic features)
* **Hidden Layer 1**: 64 neurons with ReLU activation, Batch Normalization, and Dropout (0.2)
* **Hidden Layer 2**: 32 neurons with ReLU activation, Batch Normalization, and Dropout (0.2)
* **Hidden Layer 3**: 16 neurons with ReLU activation
* **Output Layer**: 5 neurons with Softmax activation (logits representation)
* **Optimizer**: Adam (learning rate = 1e-3)
* **Loss Function**: Cross-Entropy Loss
* **Callbacks**: Early Stopping (patience = 10), Save Best State Dict (`health_impact_ann.pt`)

### 2. Air Quality LSTM (PyTorch)
A stacked Recurrent Neural Network built with PyTorch to capture short-term temporal dependencies in meteorological features and forecast $CO(GT)$ concentrations:
* **Look-Back Window**: 24 hours of sequence history
* **Input Features**: 24 features (engineered rolling averages, lags, time indicators)
* **LSTM Layer 1**: 64 units (return sequences = True) followed by Dropout (0.2)
* **LSTM Layer 2**: 32 units (return sequences = False) followed by Dropout (0.2)
* **Dense Layer 1**: 16 units with ReLU activation
* **Dense Layer 2 (Output)**: 1 unit with Linear activation for forecasting
* **Preprocessing**: Dedicated MinMaxScaler `[0, 1]` applied locally to sequence inputs to prevent gate saturation (independent of ML standard scaler)
* **Optimizer**: Adam (learning rate = 1e-3)
* **Loss Function**: Mean Squared Error (MSE)
* **Callbacks**: Early Stopping (patience = 10) & Learning Rate Decay (`ReduceLROnPlateau`)
* **Auto-Selection**: During app runtime, the best model (RF vs XGBoost vs LSTM) is selected strictly based on test set RMSE. If the ML model performs better than LSTM, the system automatically uses ML for production forecasting to prevent regressions.

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
