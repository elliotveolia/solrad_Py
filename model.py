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
        prediction_days=5,
        daily_avg_load=None
):
    """
    Predict load for X days during a wildfire.
    Uses rolling predictions where each prediction feeds into the next.
    """

    # Prepare forecast data
    forecast_data = forecast_data.copy()
    forecast_data['Time'] = pd.to_datetime(forecast_data['time'], utc=True)
    forecast_data = forecast_data.sort_values('Time').reset_index(drop=True)

    forecast_data['hour'] = forecast_data['Time'].dt.hour
    forecast_data['day_of_week'] = forecast_data['Time'].dt.dayofweek
    forecast_data['month'] = forecast_data['Time'].dt.month

    # Create cyclical features
    forecast_data['hour_sin'] = np.sin(2 * np.pi * forecast_data['hour'] / 24)
    forecast_data['hour_cos'] = np.cos(2 * np.pi * forecast_data['hour'] / 24)
    forecast_data['day_sin'] = np.sin(2 * np.pi * forecast_data['day_of_week'] / 7)
    forecast_data['day_cos'] = np.cos(2 * np.pi * forecast_data['day_of_week'] / 7)

    # Create fire indicator
    fire_start = pd.to_datetime(fire_start_date, utc=True)
    fire_end = fire_start + timedelta(days=prediction_days)
    forecast_data['is_fire'] = ((forecast_data['Time'] >= fire_start) &
                                (forecast_data['Time'] <= fire_end)).astype(int)

    # Sort history by time
    actual_load_history = actual_load_history.sort_values('Time').copy()

    # Find the actual load column
    load_col = None
    for col in ['actual_load_mean', 'actual_load', 'Load', 'load']:
        if col in actual_load_history.columns:
            load_col = col
            break

    if load_col is None:
        print(f"ERROR: Could not find load column in history. Available: {actual_load_history.columns.tolist()}")
        return None

    print(f"Using load column: {load_col}")

    # Calculate normalized load for history
    if daily_avg_load is None:
        daily_avg_load = actual_load_history[load_col].mean()

    actual_load_history['date'] = actual_load_history['Time'].dt.date
    daily_avg_history = actual_load_history.groupby('date')[load_col].transform('mean')
    actual_load_history['Load_normalized'] = ((actual_load_history[
                                                   load_col] - daily_avg_history) / daily_avg_history) * 100

    # Initialize prediction history with actual history
    prediction_history = actual_load_history[['Time', 'Load_normalized']].copy()
    prediction_history.rename(columns={'Load_normalized': 'predicted_load_normalized'}, inplace=True)

    print(f"Prediction history initialized with {len(prediction_history)} rows")
    print(f"History time range: {prediction_history['Time'].min()} to {prediction_history['Time'].max()}")

    predictions = []

    for idx, row in forecast_data.iterrows():
        feature_dict = {}
        current_time = row['Time']

        for col in feature_cols:
            if col in row.index and pd.notna(row[col]):
                feature_dict[col] = row[col]

            elif col.startswith('actual_load_lag_'):
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = current_time - timedelta(hours=lag_hours)

                # Look in prediction history
                matching = prediction_history[prediction_history['Time'] == lag_time]
                if len(matching) > 0:
                    feature_dict[col] = matching['predicted_load_normalized'].values[0]
                else:
                    feature_dict[col] = 0

            elif col.startswith('forecast_load_lag_'):
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = current_time - timedelta(hours=lag_hours)
                matching = forecast_data[forecast_data['Time'] == lag_time]
                feature_dict[col] = matching['forecast_load'].values[0] if len(matching) > 0 else row.get(
                    'forecast_load', 0)

            elif col.startswith('temperature_lag_'):
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = current_time - timedelta(hours=lag_hours)
                matching = forecast_data[forecast_data['Time'] == lag_time]
                feature_dict[col] = matching['temperature'].values[0] if len(matching) > 0 else row.get('temperature',
                                                                                                        0)

            elif col.startswith('solar_irradiance_lag_'):
                lag_hours = int(col.split('_')[-1].replace('h', ''))
                lag_time = current_time - timedelta(hours=lag_hours)
                matching = forecast_data[forecast_data['Time'] == lag_time]
                feature_dict[col] = matching['solar_irradiance'].values[0] if len(matching) > 0 else row.get(
                    'solar_irradiance', 0)

            elif col.startswith('actual_load_rolling_'):
                window = int(col.split('_')[-1].replace('h', ''))
                window_start = current_time - timedelta(hours=window)

                # Use prediction history for rolling average
                window_data = prediction_history[
                    (prediction_history['Time'] >= window_start) &
                    (prediction_history['Time'] <= current_time)
                    ]
                feature_dict[col] = window_data['predicted_load_normalized'].mean() if len(window_data) > 0 else 0

            elif col.startswith('forecast_load_rolling_'):
                window = int(col.split('_')[-1].replace('h', ''))
                window_start = current_time - timedelta(hours=window)
                window_data = forecast_data[
                    (forecast_data['Time'] >= window_start) &
                    (forecast_data['Time'] <= current_time)
                    ]
                feature_dict[col] = window_data['forecast_load'].mean() if len(window_data) > 0 else row.get(
                    'forecast_load', 0)

            else:
                feature_dict[col] = 0

        # Create feature array in correct order
        X_pred = np.array([feature_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)

        # Make prediction (returns percentage deviation)
        pred_normalized = model.predict(X_pred_scaled)[0]

        # Add to prediction history for next iteration
        new_history_row = pd.DataFrame({
            'Time': [current_time],
            'predicted_load_normalized': [pred_normalized]
        })
        prediction_history = pd.concat([prediction_history, new_history_row], ignore_index=True)

        predictions.append({
            'Time': current_time,
            'predicted_load_normalized': pred_normalized,
            'forecast_load': row.get('forecast_load', np.nan),
            'temperature': row.get('temperature', np.nan),
            'cloudcover': row.get('cloudcover', np.nan),
            'solar_irradiance': row.get('solar_irradiance', np.nan),
            'is_fire': row['is_fire']
        })

    predictions_df = pd.DataFrame(predictions)
    return predictions_df
