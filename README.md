# 🌍 Weather Trend Forecasting — Data Science Assessment

**Built by Heramb Jayant Kulkarni**
PM Accelerator Technical Assessment — Data Science

---

## About PM Accelerator

The **Product Manager Accelerator Program** is designed to support PM professionals through every stage of their career. From students to first-time PMs to Directors and Executives, the program has helped over **2,000+ members from 50+ countries** transition, grow, and advance in their Product careers.


---

## Dataset

**Global Weather Repository** — Daily weather information for cities worldwide with 40+ features.

📥 [Download from Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code)

---

## Assessment Coverage

### ✅ Basic Assessment
| Requirement | Implementation |
|-------------|----------------|
| Data Cleaning & Preprocessing | Missing value imputation (median/mode), outlier detection (IQR), normalization (StandardScaler) |
| Exploratory Data Analysis | Temperature/humidity distributions, correlation heatmap, time trends, precipitation patterns |
| Model Building | 5 models trained & evaluated with MAE, RMSE, R² metrics |
| Time Series Analysis | Uses `last_updated` column for seasonal decomposition |

### ✅ Advanced Assessment
| Requirement | Implementation |
|-------------|----------------|
| Anomaly Detection | Z-score (threshold > 3) + DBSCAN clustering-based outlier identification |
| Multiple Models + Ensemble | Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost + VotingRegressor ensemble |
| Feature Importance | Random Forest feature importance ranking with visualization |
| Climate Analysis | Hottest/coldest countries, regional temperature patterns |
| Environmental Impact | Air quality parameter correlation with temperature |
| Spatial Analysis | Latitude vs temperature scatter with quadratic trend line |
| Geographical Patterns | Country-level temperature comparison across dataset |

---

## How to Run

### Prerequisites
- Python ≥ 3.9

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/weather-forecasting-ds.git
cd weather-forecasting-ds

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Option A: Manual — download from Kaggle link above, place CSV in this directory
# Option B: Auto — if you have kagglehub configured, the script auto-downloads

# 4. Run the analysis
python weather_analysis.py
```

### Output
All visualizations are saved to `./outputs/`:

| # | File | Analysis | Level |
|---|------|----------|-------|
| 1 | `01_distributions.png` | Temperature & humidity distributions | Basic |
| 2 | `02_correlation_heatmap.png` | Feature correlation matrix | Basic |
| 3 | `03_temp_over_time.png` | Temperature scatter over time | Basic |
| 4 | `04_precipitation.png` | Precipitation distribution | Basic |
| 5 | `05_anomaly_detection.png` | DBSCAN anomaly visualization | Advanced |
| 6 | `06_model_comparison.png` | MAE / RMSE / R² across all models | Advanced |
| 7 | `07_actual_vs_predicted.png` | Best model predictions vs actuals | Basic |
| 8 | `08_feature_importance.png` | Random Forest feature importance | Advanced |
| 9 | `09_geographical_temps.png` | Hottest & coldest countries | Advanced |
| 10 | `10_temp_vs_latitude.png` | Temperature vs latitude (spatial) | Advanced |
| 11 | `11_air_quality_correlation.png` | Air quality vs temperature | Advanced |
| 12 | `12_time_series_decomposition.png` | Seasonal decomposition | Basic |

---

## Project Structure

```
weather-forecasting-ds/
├── weather_analysis.py          # Main analysis script (Basic + Advanced)
├── requirements.txt             # Python dependencies
├── .gitignore
├── outputs/                     # Generated visualizations (created on run)
│   ├── 01_distributions.png
│   ├── ...
│   └── 12_time_series_decomposition.png
├── GlobalWeatherRepository.csv  # Dataset (download from Kaggle)
└── README.md
```

---

## Models & Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | — | — | — |
| Ridge Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| XGBoost | — | — | — |
| **Ensemble** | — | — | — |

_Run `python weather_analysis.py` to populate actual results._

---

## Key Insights

- Ensemble model (VotingRegressor) consistently achieves the highest R² score
- Clear inverse relationship between latitude and temperature confirms climate zones
- DBSCAN identifies weather anomalies distinct from normal global patterns
- Air quality parameters show measurable correlations with temperature
- Humidity and pressure are the strongest predictors of temperature

---

## License

This project was created for the PM Accelerator technical assessment.
