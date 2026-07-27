import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime, timedelta

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


def predict_load_during_fire(
        forecast_data,
        actual_load_history,
        model,
        scaler,
        feature_cols,
        fire_start_date,
        prediction_days=5
):
    """
    Predict load for X days during a wildfire.

    Parameters:
    -----------
    forecast_data : DataFrame
        Weather forecast data (temperature, cloudcover, solar_irradiance, etc.)
    actual_load_history : DataFrame
        Recent actual load history (for lagged features)
    model : sklearn model
        Trained predictive model
    scaler : StandardScaler
        Fitted scaler for features
    feature_cols : list
        List of feature column names
    fire_start_date : datetime
        When the fire started
    prediction_days : int
        Number of days to predict

    Returns:
    --------
    predictions_df : DataFrame
        Predicted loads with timestamps
    """

    # Prepare forecast data
    forecast_data['Time'] = pd.to_datetime(forecast_data['time'], utc=True)
    forecast_data['date'] = forecast_data['Time'].dt.date
    forecast_data['hour'] = forecast_data['Time'].dt.hour
    forecast_data['day_of_week'] = forecast_data['Time'].dt.dayofweek
    forecast_data['month'] = forecast_data['Time'].dt.month

    # Create cyclical features
    forecast_data['hour_sin'] = np.sin(2 * np.pi * forecast_data['hour'] / 24)
    forecast_data['hour_cos'] = np.cos(2 * np.pi * forecast_data['hour'] / 24)
    forecast_data['day_sin'] = np.sin(2 * np.pi * forecast_data['day_of_week'] / 7)
    forecast_data['day_cos'] = np.cos(2 * np.pi * forecast_data['day_of_week'] / 7)

    # Create fire indicator
    fire_start = pd.to_datetime(fire_start_date)
    fire_end = fire_start + timedelta(days=prediction_days)
    forecast_data['is_fire'] = ((forecast_data['Time'] >= fire_start) &
                                (forecast_data['Time'] <= fire_end)).astype(int)

    # Get recent history for lagged features
    actual_load_history = actual_load_history.sort_values('Time').tail(24)

    predictions = []

    for idx, row in forecast_data.iterrows():
        # Build feature vector
        feature_dict = {}

        for col in feature_cols:
            if col in row.index:
                feature_dict[col] = row[col]
            elif col.startswith('actual_load_lag_'):
                # Get from history
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = row['Time'] - timedelta(hours=lag_hours)
                matching = actual_load_history[actual_load_history['Time'] == lag_time]
                feature_dict[col] = matching['actual_load_mean'].values[0] if len(matching) > 0 else row.get(
                    'forecast_load', 0)
            elif col.startswith('forecast_load_lag_'):
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = row['Time'] - timedelta(hours=lag_hours)
                matching = forecast_data[forecast_data['Time'] == lag_time]
                feature_dict[col] = matching['forecast_load'].values[0] if len(matching) > 0 else row.get(
                    'forecast_load', 0)
            elif col.startswith('temperature_lag_'):
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = row['Time'] - timedelta(hours=lag_hours)
                matching = forecast_data[forecast_data['Time'] == lag_time]
                feature_dict[col] = matching['temperature'].values[0] if len(matching) > 0 else row.get('temperature',
                                                                                                        0)
            elif col.startswith('solar_irradiance_lag_'):
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = row['Time'] - timedelta(hours=lag_hours)
                matching = forecast_data[forecast_data['Time'] == lag_time]
                feature_dict[col] = matching['solar_irradiance'].values[0] if len(matching) > 0 else row.get(
                    'solar_irradiance', 0)
            elif col.startswith('actual_load_rolling_'):
                window = int(col.split('_')[-1].replace('h', ''))
                window_start = row['Time'] - timedelta(hours=window)
                window_data = forecast_data[(forecast_data['Time'] >= window_start) &
                                            (forecast_data['Time'] <= row['Time'])]
                feature_dict[col] = window_data['forecast_load'].mean() if len(window_data) > 0 else row.get(
                    'forecast_load', 0)
            elif col.startswith('forecast_load_rolling_'):
                window = int(col.split('_')[-1].replace('h', ''))
                window_start = row['Time'] - timedelta(hours=window)
                window_data = forecast_data[(forecast_data['Time'] >= window_start) &
                                            (forecast_data['Time'] <= row['Time'])]
                feature_dict[col] = window_data['forecast_load'].mean() if len(window_data) > 0 else row.get(
                    'forecast_load', 0)
            else:
                feature_dict[col] = 0

        # Create feature array
        X_pred = np.array([feature_dict[col] for col in feature_cols]).reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)

        # Make prediction
        pred_load = model.predict(X_pred_scaled)[0]

        predictions.append({
            'Time': row['Time'],
            'predicted_load': pred_load,
            'forecast_load': row.get('forecast_load', np.nan),
            'temperature': row.get('temperature', np.nan),
            'cloudcover': row.get('cloudcover', np.nan),
            'solar_irradiance': row.get('solar_irradiance', np.nan),
            'is_fire': row['is_fire']
        })

    predictions_df = pd.DataFrame(predictions)
    return predictions_df


