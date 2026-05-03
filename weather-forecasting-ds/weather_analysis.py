"""
=========================================================================
  Weather Trend Forecasting — PM Accelerator Data Science Assessment
  Author: Heramb Jayant Kulkarni
=========================================================================

  PM Accelerator
  The Product Manager Accelerator Program is designed to support PM
  professionals through every stage of their career. From students to
  first-time PMs to Directors and Executives, the program has helped
  over 2,000+ members from 50+ countries.
  LinkedIn: https://www.linkedin.com/company/product-manager-accelerator/

  Dataset: Global Weather Repository (Kaggle)
  https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

  Assessment Coverage:
  ─── BASIC ───
  [1] Data Cleaning & Preprocessing
  [2] Exploratory Data Analysis (EDA)
  [3] Model Building & Evaluation
  [4] Time Series Analysis (last_updated)

  ─── ADVANCED ───
  [5] Anomaly Detection (Z-score + DBSCAN)
  [6] Multiple Forecasting Models + Ensemble
  [7] Feature Importance Analysis
  [8] Climate / Geographical Analysis
  [9] Environmental Impact / Air Quality
  [10] Spatial Analysis (Latitude patterns)
=========================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN

# Statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not installed. Install with: pip install xgboost")

# ── Plot Config ──────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('coolwarm')
FIGSIZE = (12, 6)
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


def save_fig(name):
    """Save current figure to outputs/ directory."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: outputs/{name}.png")


# =====================================================================
# 1. DATA LOADING
# =====================================================================
print("\n" + "=" * 70)
print(" 1. LOADING DATA")
print("=" * 70)

DATA_PATH = 'GlobalWeatherRepository.csv'

# Try local file first, then auto-download from Kaggle
if not Path(DATA_PATH).exists():
    try:
        import kagglehub
        import glob
        path = kagglehub.dataset_download(
            "nelgiriyewithana/global-weather-repository"
        )
        csvs = glob.glob(f"{path}/**/*.csv", recursive=True)
        if csvs:
            DATA_PATH = csvs[0]
        print(f"  Downloaded from Kaggle: {DATA_PATH}")
    except Exception as e:
        print(f"  ❌ Could not download dataset: {e}")
        print("  Please download 'GlobalWeatherRepository.csv' from:")
        print("  https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository")
        print("  Place the CSV file in this directory and re-run.")
        exit(1)

df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\n  First 3 rows:\n{df.head(3).to_string()}")


# =====================================================================
# 2. DATA CLEANING & PREPROCESSING
# =====================================================================
print("\n" + "=" * 70)
print(" 2. DATA CLEANING & PREPROCESSING")
print("=" * 70)

# 2a. Parse datetime column
if 'last_updated' in df.columns:
    df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
    df = df.sort_values('last_updated').reset_index(drop=True)
    print(f"  Date range: {df['last_updated'].min()} → {df['last_updated'].max()}")

# 2b. Report missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = missing_pct[missing_pct > 0]
if len(missing_report) > 0:
    print(f"\n  Columns with missing values:")
    for col, pct in missing_report.items():
        print(f"    {col}: {pct}%")
else:
    print("  ✓ No missing values found.")

# 2c. Fill missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"  ✓ Filled missing values (numeric=median, categorical=mode)")

# 2d. Identify key columns dynamically
def find_col(df, keywords, exclude=None):
    """Find first column matching any keyword."""
    for c in df.columns:
        cl = c.lower()
        if exclude and any(e in cl for e in exclude):
            continue
        if any(k in cl for k in keywords):
            return c
    return None

temp_col = find_col(df, ['temperature_celsius', 'temp_c'], exclude=['fahrenheit'])
if not temp_col:
    temp_col = find_col(df, ['temp'])
humidity_col = find_col(df, ['humidity'])
wind_col = find_col(df, ['wind_kph', 'wind_mph', 'wind_speed'])
pressure_col = find_col(df, ['pressure_mb', 'pressure'])
precip_col = find_col(df, ['precip_mm', 'precip'])
country_col = find_col(df, ['country'])
city_col = find_col(df, ['location_name', 'city'])
lat_col = find_col(df, ['latitude', 'lat'])
lon_col = find_col(df, ['longitude', 'lon'])

print(f"\n  Key columns detected:")
print(f"    Temperature: {temp_col}")
print(f"    Humidity:    {humidity_col}")
print(f"    Wind:        {wind_col}")
print(f"    Pressure:    {pressure_col}")
print(f"    Precip:      {precip_col}")
print(f"    Country:     {country_col}")
print(f"    City:        {city_col}")
print(f"    Lat/Lon:     {lat_col} / {lon_col}")

# 2e. Outlier detection with IQR
if temp_col:
    Q1 = df[temp_col].quantile(0.25)
    Q3 = df[temp_col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[temp_col] < lower) | (df[temp_col] > upper)]
    print(f"\n  Temperature outliers (IQR): {len(outliers):,} "
          f"({len(outliers)/len(df)*100:.2f}%)")
    print(f"  Bounds: [{lower:.1f}°C, {upper:.1f}°C]")

# 2f. Normalize features for modeling
feature_cols = [c for c in [temp_col, humidity_col, wind_col, pressure_col, precip_col]
                if c is not None]
scaler = StandardScaler()
df_scaled = df.copy()
if feature_cols:
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols].values)

print(f"\n  ✓ Preprocessing complete. Final shape: {df.shape}")


# =====================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) — Basic
# =====================================================================
print("\n" + "=" * 70)
print(" 3. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 3a. Temperature & humidity distributions
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

if temp_col:
    axes[0].hist(df[temp_col].dropna(), bins=60, color='#4fc3f7',
                 edgecolor='white', alpha=0.85)
    axes[0].set_title('Temperature Distribution (°C)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Frequency')
    mean_t = df[temp_col].mean()
    axes[0].axvline(mean_t, color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean: {mean_t:.1f}°C')
    axes[0].legend()

if humidity_col:
    axes[1].hist(df[humidity_col].dropna(), bins=60, color='#66bb6a',
                 edgecolor='white', alpha=0.85)
    axes[1].set_title('Humidity Distribution (%)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Humidity (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(df[humidity_col].mean(), color='red', linestyle='--',
                    linewidth=1.5, label=f'Mean: {df[humidity_col].mean():.1f}%')
    axes[1].legend()

save_fig('01_distributions')
print("  Generated: temperature & humidity distributions")

# 3b. Correlation heatmap
corr_cols = [c for c in feature_cols if c in df.columns]
if len(corr_cols) >= 3:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    save_fig('02_correlation_heatmap')
    print("  Generated: correlation heatmap")

# 3c. Temperature over time
if 'last_updated' in df.columns and temp_col:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sample = df.sample(min(5000, len(df)), random_state=42).sort_values('last_updated')
    scatter = ax.scatter(sample['last_updated'], sample[temp_col],
                         alpha=0.3, s=8, c=sample[temp_col], cmap='coolwarm')
    ax.set_title('Temperature Over Time (sampled)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    plt.colorbar(scatter, ax=ax, label='Temperature (°C)', shrink=0.8)
    save_fig('03_temp_over_time')
    print("  Generated: temperature over time scatter")

# 3d. Precipitation distribution
if precip_col:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    precip_data = df[precip_col].clip(upper=df[precip_col].quantile(0.99))
    ax.hist(precip_data, bins=60, color='#29b6f6', edgecolor='white', alpha=0.85)
    ax.set_title('Precipitation Distribution (mm, clipped at 99th pct)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Frequency')
    save_fig('04_precipitation')
    print("  Generated: precipitation distribution")


# =====================================================================
# 4. ANOMALY DETECTION — Advanced
# =====================================================================
print("\n" + "=" * 70)
print(" 4. ANOMALY DETECTION (Advanced)")
print("=" * 70)

if len(corr_cols) >= 2:
    # Z-score anomalies
    clean_data = df[corr_cols].dropna()
    z_scores = np.abs(stats.zscore(clean_data))
    z_anomaly_mask = (z_scores > 3).any(axis=1)
    n_z_anom = z_anomaly_mask.sum()
    print(f"  Z-score anomalies (|z| > 3): {n_z_anom:,} ({n_z_anom/len(clean_data)*100:.2f}%)")

    # DBSCAN clustering anomalies
    sample_size = min(5000, len(clean_data))
    sample_idx = clean_data.sample(sample_size, random_state=42).index
    X_cluster = StandardScaler().fit_transform(
        df.loc[sample_idx, corr_cols[:3] if len(corr_cols) >= 3 else corr_cols]
    )
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    labels = dbscan.fit_predict(X_cluster)
    n_noise = (labels == -1).sum()
    print(f"  DBSCAN noise points: {n_noise:,} ({n_noise/len(labels)*100:.2f}%)")
    print(f"  DBSCAN clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    normal = labels != -1
    ax.scatter(X_cluster[normal, 0], X_cluster[normal, 1],
               c='#4fc3f7', alpha=0.4, s=12, label='Normal')
    ax.scatter(X_cluster[~normal, 0], X_cluster[~normal, 1],
               c='#ef5350', alpha=0.8, s=20, label='Anomaly', marker='x')
    ax.set_title('Anomaly Detection (DBSCAN Clustering)', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{corr_cols[0]} (standardized)')
    ax.set_ylabel(f'{corr_cols[1]} (standardized)')
    ax.legend()
    save_fig('05_anomaly_detection')
    print("  Generated: anomaly detection plot")


# =====================================================================
# 5. FORECASTING MODELS — Basic + Advanced
# =====================================================================
print("\n" + "=" * 70)
print(" 5. FORECASTING MODELS")
print("=" * 70)

target = temp_col
model_features = [c for c in corr_cols if c != target]

results = {}
predictions = {}

if target and len(model_features) >= 2:
    X = df[model_features].dropna()
    y = df.loc[X.index, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {len(X_train):,} samples")
    print(f"  Test set:  {len(X_test):,} samples")
    print(f"  Features:  {model_features}")
    print()

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, random_state=42
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=6, random_state=42, verbosity=0
        )

    # Train & evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
        predictions[name] = y_pred
        print(f"  {name:22s} | MAE: {mae:7.3f} | RMSE: {rmse:7.3f} | R²: {r2:.4f}")

    # Ensemble (VotingRegressor)
    print("\n  Building Ensemble (VotingRegressor)...")
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=12,
                                     random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=6,
                                         random_state=42)),
    ]
    if HAS_XGB:
        estimators.append(
            ('xgb', XGBRegressor(n_estimators=100, max_depth=6,
                                 random_state=42, verbosity=0))
        )

    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(X_train, y_train)
    y_ens = ensemble.predict(X_test)
    ens_mae = mean_absolute_error(y_test, y_ens)
    ens_rmse = np.sqrt(mean_squared_error(y_test, y_ens))
    ens_r2 = r2_score(y_test, y_ens)
    results['Ensemble'] = {'MAE': ens_mae, 'RMSE': ens_rmse, 'R²': ens_r2}
    predictions['Ensemble'] = y_ens
    print(f"  {'Ensemble':22s} | MAE: {ens_mae:7.3f} | RMSE: {ens_rmse:7.3f} | R²: {ens_r2:.4f}")

    # ── Model comparison chart ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_names = list(results.keys())
    colors_list = ['#4fc3f7', '#66bb6a', '#ffb74d', '#ef5350', '#ab47bc', '#ff7043']

    for i, metric in enumerate(['MAE', 'RMSE', 'R²']):
        vals = [results[m][metric] for m in model_names]
        bars = axes[i].barh(model_names, vals,
                            color=colors_list[:len(model_names)], edgecolor='white')
        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_xlabel(metric)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_width() + max(vals) * 0.02,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    save_fig('06_model_comparison')
    print("\n  Generated: model comparison chart")

    # ── Actual vs Predicted (best model) ──
    best_model = max(results, key=lambda k: results[k]['R²'])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, predictions[best_model], alpha=0.3, s=10, c='#4fc3f7')
    lims = [
        min(y_test.min(), predictions[best_model].min()) - 5,
        max(y_test.max(), predictions[best_model].max()) + 5,
    ]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f'Actual vs Predicted — {best_model} (R²={results[best_model]["R²"]:.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Temperature (°C)')
    ax.set_ylabel('Predicted Temperature (°C)')
    ax.legend()
    ax.set_aspect('equal')
    save_fig('07_actual_vs_predicted')
    print(f"  Generated: actual vs predicted ({best_model})")
else:
    print("  ⚠ Not enough features for modeling.")
    best_model = None


# =====================================================================
# 6. FEATURE IMPORTANCE — Advanced
# =====================================================================
print("\n" + "=" * 70)
print(" 6. FEATURE IMPORTANCE (Advanced)")
print("=" * 70)

if target and len(model_features) >= 2:
    rf = RandomForestRegressor(n_estimators=100, max_depth=12,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(
        rf.feature_importances_, index=model_features
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(importances) * 0.6)))
    importances.plot(kind='barh', ax=ax, color='#ab47bc', edgecolor='white')
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    for i, (val, name) in enumerate(zip(importances.values, importances.index)):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
    save_fig('08_feature_importance')
    print(f"  Top feature: {importances.idxmax()} (importance: {importances.max():.4f})")
    print(f"  Generated: feature importance chart")


# =====================================================================
# 7. CLIMATE & GEOGRAPHICAL ANALYSIS — Advanced
# =====================================================================
print("\n" + "=" * 70)
print(" 7. CLIMATE & GEOGRAPHICAL ANALYSIS (Advanced)")
print("=" * 70)

if country_col and temp_col:
    country_temps = df.groupby(country_col)[temp_col].mean().sort_values(ascending=False)
    top_20 = country_temps.head(20)
    bottom_10 = country_temps.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    top_20.sort_values().plot(kind='barh', ax=axes[0], color='#ef5350', edgecolor='white')
    axes[0].set_title('Top 20 Hottest Countries (Avg Temp °C)',
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Average Temperature (°C)')

    bottom_10.sort_values(ascending=False).plot(
        kind='barh', ax=axes[1], color='#4fc3f7', edgecolor='white'
    )
    axes[1].set_title('Top 10 Coldest Countries (Avg Temp °C)',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Average Temperature (°C)')

    save_fig('09_geographical_temps')
    print(f"  Countries in dataset: {df[country_col].nunique()}")
    print(f"  Hottest: {country_temps.idxmax()} ({country_temps.max():.1f}°C)")
    print(f"  Coldest: {country_temps.idxmin()} ({country_temps.min():.1f}°C)")
    print("  Generated: geographical temperature analysis")


# =====================================================================
# 8. SPATIAL ANALYSIS — Advanced (Temperature vs Latitude)
# =====================================================================
print("\n" + "=" * 70)
print(" 8. SPATIAL ANALYSIS (Advanced)")
print("=" * 70)

if lat_col and temp_col:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sample = df.sample(min(8000, len(df)), random_state=42)
    scatter = ax.scatter(
        sample[lat_col], sample[temp_col],
        alpha=0.3, s=8, c=sample[temp_col], cmap='coolwarm'
    )
    ax.set_title('Temperature vs Latitude — Global Pattern',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Temperature (°C)')
    plt.colorbar(scatter, ax=ax, label='Temperature (°C)', shrink=0.8)

    # Add trend line
    z = np.polyfit(sample[lat_col].dropna(), sample[temp_col].dropna(), 2)
    p = np.poly1d(z)
    x_line = np.linspace(sample[lat_col].min(), sample[lat_col].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.7, label='Quadratic fit')
    ax.legend()

    save_fig('10_temp_vs_latitude')
    corr_val = df[lat_col].corr(df[temp_col])
    print(f"  Latitude-Temperature correlation: {corr_val:.3f}")
    print("  Generated: temperature vs latitude scatter with trend")
else:
    print("  ⚠ Latitude column not found — skipping spatial analysis.")


# =====================================================================
# 9. AIR QUALITY / ENVIRONMENTAL ANALYSIS — Advanced
# =====================================================================
print("\n" + "=" * 70)
print(" 9. AIR QUALITY ANALYSIS (Advanced)")
print("=" * 70)

aq_keywords = ['air_quality', 'co', 'no2', 'o3', 'pm2', 'pm10', 'so2', 'aqi',
               'us-epa-index', 'gb-defra-index']
aq_cols = [c for c in df.columns
           if any(k in c.lower() for k in aq_keywords)
           and df[c].dtype in [np.float64, np.int64, float, int]]

if aq_cols and temp_col:
    print(f"  Air quality columns: {len(aq_cols)}")
    for col in aq_cols[:5]:
        print(f"    {col}: mean={df[col].mean():.2f}, "
              f"corr_with_temp={df[col].corr(df[temp_col]):.3f}")

    aq_corr = df[aq_cols + [temp_col]].corr()[temp_col].drop(temp_col).sort_values()

    fig, ax = plt.subplots(figsize=(12, max(4, len(aq_corr) * 0.5)))
    colors = ['#ef5350' if v < 0 else '#66bb6a' for v in aq_corr]
    aq_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Air Quality Correlation with Temperature',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Pearson Correlation Coefficient')
    ax.axvline(0, color='gray', linewidth=0.8)
    for i, (val, name) in enumerate(zip(aq_corr.values, aq_corr.index)):
        ax.text(val + 0.01 if val >= 0 else val - 0.06, i, f'{val:.3f}',
                va='center', fontsize=9)
    save_fig('11_air_quality_correlation')
    print("  Generated: air quality correlation chart")
else:
    print("  ⚠ No air quality columns found in dataset.")


# =====================================================================
# 10. TIME SERIES DECOMPOSITION
# =====================================================================
print("\n" + "=" * 70)
print(" 10. TIME SERIES DECOMPOSITION")
print("=" * 70)

if 'last_updated' in df.columns and city_col and temp_col:
    city_counts = df[city_col].value_counts()
    target_city = city_counts.index[0]
    city_data = (
        df[df[city_col] == target_city]
        .set_index('last_updated')[temp_col]
        .sort_index()
        .resample('D')
        .mean()
        .dropna()
    )

    if len(city_data) >= 14:
        print(f"  City: {target_city} ({len(city_data)} daily observations)")

        period = min(7, len(city_data) // 3)
        try:
            decomp = seasonal_decompose(city_data, model='additive', period=period)
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

            decomp.observed.plot(ax=axes[0], color='#4fc3f7', linewidth=1)
            axes[0].set_title(f'Time Series Decomposition — {target_city}',
                              fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Observed')

            decomp.trend.plot(ax=axes[1], color='#ff7043', linewidth=1.5)
            axes[1].set_ylabel('Trend')

            decomp.seasonal.plot(ax=axes[2], color='#66bb6a', linewidth=1)
            axes[2].set_ylabel('Seasonal')

            decomp.resid.plot(ax=axes[3], color='#ab47bc', linewidth=1)
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Date')

            save_fig('12_time_series_decomposition')
            print("  Generated: time series decomposition")
        except Exception as e:
            print(f"  ⚠ Decomposition error: {e}")
    else:
        print(f"  ⚠ Not enough data for {target_city} ({len(city_data)} days, need ≥14)")
else:
    print("  ⚠ Required columns not found for time series analysis.")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print(" SUMMARY")
print("=" * 70)

n_plots = len(list(OUTPUT_DIR.glob('*.png')))

print(f"""
  Dataset:       {df.shape[0]:,} rows × {df.shape[1]} columns
  Temperature:   {df[temp_col].min():.1f}°C – {df[temp_col].max():.1f}°C (mean: {df[temp_col].mean():.1f}°C)
  Countries:     {df[country_col].nunique() if country_col else 'N/A'}
  Cities:        {df[city_col].nunique() if city_col else 'N/A'}

  Models trained: {len(results) if results else 0}
  Best model:     {best_model or 'N/A'}
  Best R²:        {max(r['R²'] for r in results.values()):.4f if results else 'N/A'}
  Best MAE:       {min(r['MAE'] for r in results.values()):.3f if results else 'N/A'}°C

  Visualizations: {n_plots} plots saved to ./outputs/

  ✅ Analysis complete!
  All output files are in the ./outputs/ directory.
""")
print("=" * 70)
