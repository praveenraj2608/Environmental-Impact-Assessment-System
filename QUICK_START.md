# ⚡ Quick Start Guide

## 1. Make sure datasets are in `data/`
```
data/
├── City_Types.csv
├── air_quality_health_impact_data.csv
└── AirQualityUCI.csv
```

## 2. Train all models (one command)
```bash
python train_all_models.py
```
*Takes ~3-5 minutes. Models saved to `models/`.*

## 3. Run the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

## 4. Use the app (recommended flow)

| Step | Page | Action |
|------|------|--------|
| 1 | 📊 Dataset Insights | Explore EDA charts |
| 2 | 🏭 City Type Prediction | Enter pollutants → get Industrial/Residential |
| 3 | 🫁 Health Impact | Enter parameters → get health risk class |
| 4 | 📈 Air Quality Analysis | View historical trends & forecasts |
| 5 | 🔍 Environmental Assessment | Auto-fills from steps 2-4 → Risk Score |
| 6 | 💡 Mitigation Recommendations | Auto-fills → Action plan |
| 7 | 📋 AI Environmental Report | Generate & download full report |

## 5. Enable AI reports (optional)
```bash
copy .env.example .env
# Edit .env → set ENV_AI_API_KEY=sk-...
# Restart the app
```
