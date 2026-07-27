from datetime import datetime, timedelta
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from caiso import fetch_caiso_load_data, get_california_weather, align_datasets
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression

start = datetime(2020, 1, 1)
end = datetime(2025, 12, 31)

#Generate data sets
gen = False
if gen:
    actual_data, forcast_data = fetch_caiso_load_data(start, end)
    weather_data = get_california_weather(start, end)
    sys.exit()


# Load the actual load data
actual_load = pd.read_csv('../data/model/caiso_actual_load.csv')
forecasted_load = pd.read_csv('../data/model/caiso_forecasted_load.csv')
weather_load = pd.read_csv('../data/model/california_weather.csv')


def quantify_fire_load_impact(actual_load, forecasted_load, weather_load, days=10):
    """
    Quantify load drop using only PRE-FIRE forecasts.
    Includes temperature, smoke AQI proxy, solar irradiance, and time-of-day analysis.
    """
    # Define fire periods
    fire_periods = [
        # 2020 Fires
        ('2020-08-16', '2020-11-30'),  # August Complex
        ('2020-09-04', '2020-12-24'),  # Creek Fire
        ('2020-08-19', '2020-10-16'),  # North Complex
        ('2020-08-16', '2020-10-15'),  # SCU Lightning Complex
        ('2020-08-16', '2020-10-16'),  # LNU Lightning Complex

        # 2021 Fires
        ('2021-07-13', '2021-10-25'),  # Dixie Fire
        ('2021-08-14', '2021-09-21'),  # Caldor Fire
        ('2021-07-30', '2021-09-15'),  # Tamarack Fire

        # 2022 Fires
        ('2022-09-05', '2022-10-05'),  # Mosquito Fire
        ('2022-07-08', '2022-08-15'),  # Fairview Fire

        # 2023 Fires
        ('2023-08-15', '2023-11-17'),  # Park Fire (started late July, major activity Aug)
        ('2023-09-09', '2023-10-12'),  # Coastal Fire

        # 2024 Fires
        ('2024-07-24', '2024-09-26'),  # Park Fire (2024 continuation/major phase)
        ('2024-06-22', '2024-08-10'),  # Borel Fire

        # 2025 Fires
        ('2025-01-07', '2025-01-31'),  # Current/Recent fires
    ]

    weather_load['time'] = pd.to_datetime(weather_load['time'], utc=True)
    weather_load['month_day'] = weather_load['time'].dt.strftime('%m-%d')
    weather_load['year'] = weather_load['time'].dt.year

    results = []

    for fire_start, fire_end in fire_periods:
        fire_start = pd.to_datetime(fire_start, utc=True)
        fire_end_limited = fire_start + pd.Timedelta(days=days - 1)

        # Get actual load during fire
        actual_fire = actual_load[(actual_load['Time'] >= fire_start) &
                                  (actual_load['Time'] <= fire_end_limited)].copy()
        actual_fire['Time'] = pd.to_datetime(actual_fire['Time'], utc=True)
        actual_hourly = actual_fire.set_index('Time').resample('1h').agg({'Load': 'mean'}).reset_index()

        # Get forecasts made BEFORE fire started
        forecast_cutoff = fire_start - pd.Timedelta(days=1)
        forecast_fire = forecasted_load[(forecasted_load['Time'] >= fire_start) &
                                        (forecasted_load['Time'] <= fire_end_limited) &
                                        (pd.to_datetime(forecasted_load['Publish Time'], utc=True) < forecast_cutoff)]

        # Get fire period weather
        fire_weather = weather_load[(weather_load['time'] >= fire_start) &
                                    (weather_load['time'] <= fire_end_limited)]

        # Get baseline weather - same calendar dates in other years
        fire_month_day_range = set()
        for i in range(days):
            date = fire_start + pd.Timedelta(days=i)
            fire_month_day_range.add(date.strftime('%m-%d'))

        baseline_weather = weather_load[
            (weather_load['month_day'].isin(fire_month_day_range)) &
            (weather_load['year'] != fire_start.year)
            ]

        if len(forecast_fire) == 0:
            print(f"No pre-fire forecasts for {fire_start.date()}")
            continue

        actual_mean = actual_hourly['Load'].mean()
        forecast_mean = forecast_fire['Load Forecast'].mean()
        load_drop_pct = ((forecast_mean - actual_mean) / forecast_mean) * 100

        # Cloud cover metrics
        fire_cloudcover = fire_weather['cloudcover'].mean() if len(fire_weather) > 0 else None
        baseline_cloudcover = baseline_weather['cloudcover'].mean() if len(baseline_weather) > 0 else None

        # Temperature metrics
        fire_temperature = fire_weather['temperature'].mean() if len(fire_weather) > 0 else None
        baseline_temperature = baseline_weather['temperature'].mean() if len(baseline_weather) > 0 else None

        # Smoke AQI proxy metrics
        fire_smoke_aqi = fire_weather['smoke_aqi_proxy'].mean() if len(fire_weather) > 0 else None
        baseline_smoke_aqi = baseline_weather['smoke_aqi_proxy'].mean() if len(baseline_weather) > 0 else None

        # Solar irradiance metrics (NEW)
        fire_solar_irradiance = fire_weather['solar_irradiance'].mean() if len(fire_weather) > 0 else None
        baseline_solar_irradiance = baseline_weather['solar_irradiance'].mean() if len(baseline_weather) > 0 else None

        # Time-of-day analysis (NEW) - peak hours are 12-18 (noon to 6pm)
        fire_peak_hours = fire_weather[(fire_weather['hour'] >= 12) & (fire_weather['hour'] <= 18)]
        baseline_peak_hours = baseline_weather[(baseline_weather['hour'] >= 12) & (baseline_weather['hour'] <= 18)]

        fire_peak_solar = fire_peak_hours['solar_irradiance'].mean() if len(fire_peak_hours) > 0 else None
        baseline_peak_solar = baseline_peak_hours['solar_irradiance'].mean() if len(baseline_peak_hours) > 0 else None

        # Calculate fire over baseline cloudcover ratio
        if fire_cloudcover is not None and baseline_cloudcover is not None and baseline_cloudcover != 0:
            fire_over_base_cover = fire_cloudcover / baseline_cloudcover
        else:
            fire_over_base_cover = None

        results.append({
            'fire_start': fire_start,
            'fire_end_analyzed': fire_end_limited,
            'duration_days': days,
            'actual_load': actual_mean,
            'forecast_load': forecast_mean,
            'load_drop_pct': load_drop_pct,
            'fire_cloudcover': fire_cloudcover,
            'baseline_cloudcover': baseline_cloudcover,
            'fire_over_base_cover': fire_over_base_cover,
            'fire_temperature': fire_temperature,
            'baseline_temperature': baseline_temperature,
            'fire_smoke_aqi': fire_smoke_aqi,
            'baseline_smoke_aqi': baseline_smoke_aqi,
            'fire_solar_irradiance': fire_solar_irradiance,
            'baseline_solar_irradiance': baseline_solar_irradiance,
            'fire_peak_solar': fire_peak_solar,
            'baseline_peak_solar': baseline_peak_solar,
            'forecast_data_points': len(forecast_fire)
        })

    results_df = pd.DataFrame(results)
    print(f"FIRE IMPACT ANALYSIS (First {days} days - Pre-fire forecasts only):")
    print(results_df.to_string())
    print(f"\nAverage load drop: {results_df['load_drop_pct'].mean():.2f}%")

    return results_df


def analyze_driving_variables(fire_impact_df):
    """
    Analyze correlations between load drop and weather variables.
    Includes: cloudcover, temperature, smoke AQI proxy, and solar irradiance (all hours).
    """

    analysis_df = fire_impact_df.dropna(subset=['load_drop_pct', 'fire_cloudcover', 'baseline_cloudcover'])

    if len(analysis_df) == 0:
        print("No complete data for analysis")
        return

    print("=" * 80)
    print("DRIVING VARIABLES ANALYSIS")
    print("=" * 80)

    # 1. Solar Irradiance Analysis
    print("\n1. SOLAR IRRADIANCE IMPACT (All Hours):")
    print("-" * 80)

    solar_data = analysis_df.dropna(subset=['fire_solar_irradiance', 'baseline_solar_irradiance'])
    if len(solar_data) > 0:
        solar_diff = solar_data['fire_solar_irradiance'] - solar_data['baseline_solar_irradiance']
        corr_solar, p_value_solar = stats.pearsonr(solar_diff, solar_data['load_drop_pct'])

        print(f"Solar irradiance difference (fire - baseline):")
        print(f"  Mean: {solar_diff.mean():.2f} W/m²")
        print(f"  Std: {solar_diff.std():.2f} W/m²")
        print(f"\nCorrelation with load drop: {corr_solar:.4f}")
        print(f"P-value: {p_value_solar:.4f}")
        print(f"Significance: {'***' if p_value_solar < 0.05 else 'Not significant'}")

        print(f"\nFire Solar Irradiance:")
        print(f"  Mean: {solar_data['fire_solar_irradiance'].mean():.2f} W/m²")
        print(f"  Std: {solar_data['fire_solar_irradiance'].std():.2f} W/m²")

        print(f"\nBaseline Solar Irradiance:")
        print(f"  Mean: {solar_data['baseline_solar_irradiance'].mean():.2f} W/m²")
        print(f"  Std: {solar_data['baseline_solar_irradiance'].std():.2f} W/m²")
    else:
        print("Insufficient solar irradiance data")

    # 2. Cloud Cover Analysis
    print("\n2. CLOUD COVER IMPACT:")
    print("-" * 80)

    cloudcover_diff = analysis_df['fire_cloudcover'] - analysis_df['baseline_cloudcover']
    corr_cloudcover, p_value_cc = stats.pearsonr(cloudcover_diff, analysis_df['load_drop_pct'])

    print(f"Cloud cover difference (fire - baseline):")
    print(f"  Mean: {cloudcover_diff.mean():.2f}%")
    print(f"  Std: {cloudcover_diff.std():.2f}%")
    print(f"\nCorrelation with load drop: {corr_cloudcover:.4f}")
    print(f"P-value: {p_value_cc:.4f}")
    print(f"Significance: {'***' if p_value_cc < 0.05 else 'Not significant'}")

    # 3. Cloud Cover Ratio Analysis
    print("\n3. CLOUD COVER RATIO (Fire/Baseline):")
    print("-" * 80)

    ratio_data = analysis_df.dropna(subset=['fire_over_base_cover'])
    if len(ratio_data) > 0:
        corr_ratio, p_value_ratio = stats.pearsonr(ratio_data['fire_over_base_cover'],
                                                   ratio_data['load_drop_pct'])
        print(f"Mean ratio: {ratio_data['fire_over_base_cover'].mean():.4f}")
        print(f"Correlation with load drop: {corr_ratio:.4f}")
        print(f"P-value: {p_value_ratio:.4f}")
        print(f"Significance: {'***' if p_value_ratio < 0.05 else 'Not significant'}")

    # 4. Temperature Analysis
    print("\n4. TEMPERATURE IMPACT:")
    print("-" * 80)

    temp_data = analysis_df.dropna(subset=['fire_temperature', 'baseline_temperature'])
    if len(temp_data) > 0:
        temp_diff = temp_data['fire_temperature'] - temp_data['baseline_temperature']
        corr_temp, p_value_temp = stats.pearsonr(temp_diff, temp_data['load_drop_pct'])

        print(f"Temperature difference (fire - baseline):")
        print(f"  Mean: {temp_diff.mean():.2f}°F")
        print(f"  Std: {temp_diff.std():.2f}°F")
        print(f"\nCorrelation with load drop: {corr_temp:.4f}")
        print(f"P-value: {p_value_temp:.4f}")
        print(f"Significance: {'***' if p_value_temp < 0.05 else 'Not significant'}")

        print(f"\nFire Temperature:")
        print(f"  Mean: {temp_data['fire_temperature'].mean():.2f}°F")
        print(f"  Std: {temp_data['fire_temperature'].std():.2f}°F")

        print(f"\nBaseline Temperature:")
        print(f"  Mean: {temp_data['baseline_temperature'].mean():.2f}°F")
        print(f"  Std: {temp_data['baseline_temperature'].std():.2f}°F")
    else:
        print("Insufficient temperature data")

    # 5. Smoke AQI Proxy Analysis
    print("\n5. SMOKE AQI PROXY IMPACT:")
    print("-" * 80)

    smoke_data = analysis_df.dropna(subset=['fire_smoke_aqi', 'baseline_smoke_aqi'])
    if len(smoke_data) > 0:
        smoke_diff = smoke_data['fire_smoke_aqi'] - smoke_data['baseline_smoke_aqi']
        corr_smoke, p_value_smoke = stats.pearsonr(smoke_diff, smoke_data['load_drop_pct'])

        print(f"Smoke AQI difference (fire - baseline):")
        print(f"  Mean: {smoke_diff.mean():.2f}")
        print(f"  Std: {smoke_diff.std():.2f}")
        print(f"\nCorrelation with load drop: {corr_smoke:.4f}")
        print(f"P-value: {p_value_smoke:.4f}")
        print(f"Significance: {'***' if p_value_smoke < 0.05 else 'Not significant'}")

        print(f"\nFire Smoke AQI:")
        print(f"  Mean: {smoke_data['fire_smoke_aqi'].mean():.2f}")
        print(f"  Std: {smoke_data['fire_smoke_aqi'].std():.2f}")

        print(f"\nBaseline Smoke AQI:")
        print(f"  Mean: {smoke_data['baseline_smoke_aqi'].mean():.2f}")
        print(f"  Std: {smoke_data['baseline_smoke_aqi'].std():.2f}")
    else:
        print("Insufficient smoke AQI data")

    # 6. Summary Statistics
    print("\n6. SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"\nLoad Drop %:")
    print(f"  Mean: {analysis_df['load_drop_pct'].mean():.2f}%")
    print(f"  Std: {analysis_df['load_drop_pct'].std():.2f}%")
    print(f"  Min: {analysis_df['load_drop_pct'].min():.2f}%")
    print(f"  Max: {analysis_df['load_drop_pct'].max():.2f}%")

    print(f"\nFire Cloud Cover:")
    print(f"  Mean: {analysis_df['fire_cloudcover'].mean():.2f}%")
    print(f"  Std: {analysis_df['fire_cloudcover'].std():.2f}%")

    print(f"\nBaseline Cloud Cover:")
    print(f"  Mean: {analysis_df['baseline_cloudcover'].mean():.2f}%")
    print(f"  Std: {analysis_df['baseline_cloudcover'].std():.2f}%")

    # 7. Correlation Matrix
    print("\n7. CORRELATION MATRIX:")
    print("-" * 80)

    corr_cols = ['load_drop_pct', 'fire_cloudcover', 'baseline_cloudcover', 'fire_over_base_cover',
                 'fire_temperature', 'baseline_temperature', 'fire_smoke_aqi', 'baseline_smoke_aqi',
                 'fire_solar_irradiance', 'baseline_solar_irradiance']

    corr_cols = [col for col in corr_cols if col in analysis_df.columns]

    corr_matrix = analysis_df[corr_cols].corr()
    print(corr_matrix)

    # 8. Linear Regression
    print("\n8. LINEAR REGRESSION ANALYSIS:")
    print("-" * 80)

    features = ['fire_cloudcover', 'baseline_cloudcover', 'fire_temperature',
                'baseline_temperature', 'fire_smoke_aqi', 'baseline_smoke_aqi',
                'fire_solar_irradiance', 'baseline_solar_irradiance']

    available_features = [f for f in features if f in analysis_df.columns]

    regression_data = analysis_df[available_features + ['load_drop_pct']].dropna()

    if len(regression_data) > 0:
        X = regression_data[available_features].values
        y = regression_data['load_drop_pct'].values

        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)

        print(f"Model: load_drop_pct = {model.intercept_:.4f}")
        for i, feature in enumerate(available_features):
            sign = "+" if model.coef_[i] >= 0 else ""
            print(f"  {sign} {model.coef_[i]:.6f} * {feature}")

        print(f"\nR-squared: {r_squared:.4f}")
        print(f"Explained variance: {r_squared * 100:.2f}%")
        print(f"\nCoefficients:")
        for i, feature in enumerate(available_features):
            print(f"  {feature}: {model.coef_[i]:.6f}")
    else:
        print("Insufficient data for regression")
        model = None

    return analysis_df, model


def prepare_training_data(aligned_data, fire_impact_df, lookback_days=5):
    """
    Prepare training data for the predictive model.
    Creates features from historical data and fire impact patterns.
    """

    # Merge aligned data with fire impact information
    aligned_data['Time'] = pd.to_datetime(aligned_data['Time'], utc=True)
    aligned_data['date'] = aligned_data['Time'].dt.date

    # Create fire indicator (1 if during fire, 0 otherwise)
    fire_dates = set()
    for _, row in fire_impact_df.iterrows():
        fire_start = pd.to_datetime(row['fire_start']).date()
        fire_end = pd.to_datetime(row['fire_end_analyzed']).date()
        current_date = fire_start
        while current_date <= fire_end:
            fire_dates.add(current_date)
            current_date += timedelta(days=1)

    aligned_data['is_fire'] = aligned_data['date'].isin(fire_dates).astype(int)

    # Create lagged features (previous hours' data)
    for lag in [1, 6, 24]:  # 1 hour, 6 hours, 1 day ago
        aligned_data[f'actual_load_lag_{lag}h'] = aligned_data['actual_load_mean'].shift(lag)
        aligned_data[f'forecast_load_lag_{lag}h'] = aligned_data['forecast_load'].shift(lag)
        aligned_data[f'temperature_lag_{lag}h'] = aligned_data['temperature'].shift(lag)
        aligned_data[f'solar_irradiance_lag_{lag}h'] = aligned_data['solar_irradiance'].shift(lag)

    # Create rolling averages
    for window in [6, 24]:  # 6 hours, 1 day
        aligned_data[f'actual_load_rolling_{window}h'] = aligned_data['actual_load_mean'].rolling(window).mean()
        aligned_data[f'forecast_load_rolling_{window}h'] = aligned_data['forecast_load'].rolling(window).mean()

    # Extract time features
    aligned_data['hour'] = aligned_data['Time'].dt.hour
    aligned_data['day_of_week'] = aligned_data['Time'].dt.dayofweek
    aligned_data['month'] = aligned_data['Time'].dt.month

    # Create cyclical features for hour and day
    aligned_data['hour_sin'] = np.sin(2 * np.pi * aligned_data['hour'] / 24)
    aligned_data['hour_cos'] = np.cos(2 * np.pi * aligned_data['hour'] / 24)
    aligned_data['day_sin'] = np.sin(2 * np.pi * aligned_data['day_of_week'] / 7)
    aligned_data['day_cos'] = np.cos(2 * np.pi * aligned_data['day_of_week'] / 7)

    # Drop rows with NaN values
    training_data = aligned_data.dropna()

    print(f"Training data shape: {training_data.shape}")
    print(f"Fire events in training data: {training_data['is_fire'].sum()} hours")
    print(f"Non-fire events in training data: {(1 - training_data['is_fire']).sum()} hours")

    return training_data


def build_wildfire_load_model(training_data, model_type='random_forest'):
    """
    Build a model to predict load during wildfire conditions.
    """

    # Define features
    feature_cols = [
        'forecast_load', 'temperature', 'cloudcover', 'humidity', 'solar_irradiance', 'smoke_aqi_proxy',
        'is_fire', 'hour', 'day_of_week', 'month',
        'actual_load_lag_1h', 'actual_load_lag_6h', 'actual_load_lag_24h',
        'forecast_load_lag_1h', 'forecast_load_lag_6h', 'forecast_load_lag_24h',
        'temperature_lag_1h', 'temperature_lag_6h', 'temperature_lag_24h',
        'solar_irradiance_lag_1h', 'solar_irradiance_lag_6h', 'solar_irradiance_lag_24h',
        'actual_load_rolling_6h', 'actual_load_rolling_24h',
        'forecast_load_rolling_6h', 'forecast_load_rolling_24h',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]

    # Filter to only available columns
    available_features = [col for col in feature_cols if col in training_data.columns]

    X = training_data[available_features]
    y = training_data['actual_load_mean']

    print(f"Features used: {len(available_features)}")
    print(f"Training samples: {len(X)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build model
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = LinearRegression()

    model.fit(X_scaled, y)

    # Calculate R-squared
    train_score = model.score(X_scaled, y)
    print(f"\nModel R-squared: {train_score:.4f}")

    # Feature importance (for Random Forest)
    if model_type == 'random_forest':
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

    return model, scaler, available_features


# Run alignment
aligned_data = align_datasets(actual_load, forecasted_load, weather_load)
fire_impact = quantify_fire_load_impact(actual_load, forecasted_load, weather_load, days=3)
analysis_results, regression_model = analyze_driving_variables(fire_impact)
train_data = prepare_training_data(aligned_data, fire_impact)