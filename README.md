# Video Game Sales Dashboard - Streamlit

## Run the dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
- `app.py` — Main dashboard (4 tabs)
- `ADY201m_enhanced_v2.xlsx` — Dataset with 74 columns
- `models/best_model.joblib` — Trained, tuned XGBoost model
- `models/*.csv` — Cross-validation, tuning, test, feature importance, and error analysis results

## Features
1. **RQ1 tab:** Genre, platform, and regional impact, feature importance, and filters
2. **RQ2 tab:** Comparison of 7 models, cross-validation, and tuning log
3. **RQ3 tab:** Genre × console generation heatmap, annual trends, and genre-level prediction errors
4. **Prediction tab:** Enter game details to predict sales, view the confidence interval, and receive out-of-training-range warnings
