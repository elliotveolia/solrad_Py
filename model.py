import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
    forecast_data['Time'] = pd.to_datetime(forecast_data['time'], )
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


