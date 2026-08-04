import sys
from datetime import datetime, timedelta
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from helper import fetch_load_data, get_weather_data, align_datasets
from model import predict_load_during_fire
import plotly.graph_objects as go


# Define scenario
scenario_start = '2026-01-01'
scenario_end = '2026-07-27'
fire_start_date = '2026-07-14 '
fire_duration_days = 5

# Location
lat = 39.9526
lon = -75.1652
location_name = 'Philadelphia'
timezone = 'America/New_York'

print(f"Scenario: {scenario_start} to {scenario_end}")
print(f"Fire: {fire_start_date} for {fire_duration_days} days")
print(f"Location: {location_name} ({lat}, {lon})")




print("\nLoading trained model...")
wildfire_model = joblib.load('data/model/wildfire_load_model.pkl')
scaler = joblib.load('data/model/wildfire_scaler.pkl')
feature_cols = joblib.load('data/model/wildfire_features.pkl')
print("✓ Model loaded!")



# ============================================================================
# LOAD/CREATE DATA FOR SCENARIO
# ============================================================================
print("\n" + "=" * 80)
print("LOADING DATA FOR SCENARIO")
print("=" * 80)

print(f"\nFetching PJM load data...")
actual_load, forecasted_load = fetch_load_data(scenario_start, scenario_end)

if actual_load is None or forecasted_load is None:
    print("ERROR: Failed to fetch load data")
    exit()

print(f"\nFetching weather data...")
weather_load = get_weather_data(
    start_date=scenario_start,
    end_date=scenario_end,
    lat=lat,
    lon=lon,
    location_name=location_name,
    timezone=timezone
)

if weather_load is None:
    print("ERROR: Failed to fetch weather data")
    exit()

print(actual_load, forecasted_load, weather_load)

print(f"\nAligning datasets...")
aligned_data = align_datasets(actual_load, forecasted_load, weather_load)

print("✓ Data loaded and aligned!")

# ============================================================================
# MAKE PREDICTIONS FOR FIRE
# ============================================================================
print("\n" + "=" * 80)
print(f"PREDICTING LOAD FOR {location_name.upper()} FIRE ({fire_duration_days} days)")
print("=" * 80)

# Define fire period
fire_start = pd.to_datetime(fire_start_date)
fire_end = fire_start + timedelta(days=fire_duration_days)

print(f"\nFire period: {fire_start.date()} to {fire_end.date()}")

# Get weather data for fire period
fire_weather = get_weather_data(
    start_date=fire_start,
    end_date=fire_end,
    lat=lat,
    lon=lon,
    location_name=f'{location_name}_fire',
    timezone=timezone
)

if fire_weather is None:
    print("ERROR: Failed to load fire period weather data")
    exit()

print(f"✓ Fire weather data loaded: {len(fire_weather)} rows")

# Filter weather data for fire period
forecast_subset = fire_weather[
    (pd.to_datetime(fire_weather['time']) >= fire_start) &
    (pd.to_datetime(fire_weather['time']) <= fire_end)
].copy()


print(f"✓ Forecast subset: {len(forecast_subset)} rows")

# Merge in forecast load data
forecast_subset['Time'] = pd.to_datetime(forecast_subset['time'])
fire_start_utc = pd.to_datetime(fire_start, utc=True)
fire_end_utc = pd.to_datetime(fire_end, utc=True)

forecast_load_subset = forecasted_load[
    (pd.to_datetime(forecasted_load['Time'], utc=True) >= fire_start_utc) &
    (pd.to_datetime(forecasted_load['Time'], utc=True) <= fire_end_utc)
][['Time', 'Load Forecast']].copy()
forecast_load_subset['Time'] = pd.to_datetime(forecast_load_subset['Time'], utc=True)

# Rename for consistency
forecast_load_subset.rename(columns={'Load Forecast': 'forecast_load'}, inplace=True)

# Convert forecast_subset Time to UTC for merge
forecast_subset['Time'] = pd.to_datetime(forecast_subset['time'], utc=True)

# Merge
forecast_subset = forecast_subset.merge(forecast_load_subset, on='Time', how='left')

print(f"✓ Forecast subset with load: {len(forecast_subset)} rows")


# Define history period (24 hours before fire starts)
history_start = pd.to_datetime(fire_start, utc=True) - timedelta(days=1)
fire_start_utc = pd.to_datetime(fire_start, utc=True)

# Load historical aligned data
history_subset = aligned_data[
    (aligned_data['Time'] >= history_start) &
    (aligned_data['Time'] < fire_start_utc)
].copy()

print(f"✓ History subset: {len(history_subset)} rows")

if len(history_subset) == 0:
    print("WARNING: No historical data found!")
    exit()

print(f"✓ Found {len(history_subset)} hours of historical data")

# Get actual load for fire period
fire_start_utc = pd.to_datetime(fire_start, utc=True)
fire_end_utc = pd.to_datetime(fire_end, utc=True)

actual_load_subset = actual_load[
    (pd.to_datetime(actual_load['Time'], utc=True) >= fire_start_utc) &
    (pd.to_datetime(actual_load['Time'], utc=True) <= fire_end_utc)
][['Time', 'Load']].copy()
actual_load_subset['Time'] = pd.to_datetime(actual_load_subset['Time'], utc=True)
actual_load_subset.rename(columns={'Load': 'actual_load'}, inplace=True)

# Make predictions
predictions = predict_load_during_fire(
    forecast_subset,
    history_subset,
    wildfire_model,
    scaler,
    feature_cols,
    fire_start,
    prediction_days=fire_duration_days,
    daily_avg_load=actual_load['Load'].mean()  # Add this
)

# Merge actual load into predictions - convert to same timezone
predictions['Time'] = pd.to_datetime(predictions['Time'], utc=True)
actual_load_subset['Time'] = actual_load_subset['Time'].dt.tz_convert('UTC')

predictions = predictions.merge(actual_load_subset, on='Time', how='left')


# ====================================================================
# DENORMALIZE PREDICTIONS TO TARGET ISO SCALE
# ====================================================================

# Use the fire period's actual mean for denormalization (more accurate)
daily_avg_target = actual_load_subset['actual_load'].mean()

print(f"\nDenormalizing predictions:")
print(f"  Fire period daily average load: {daily_avg_target:.2f} MW")

# Convert percentage deviation back to absolute load
predictions['predicted_load'] = daily_avg_target * (1 + predictions['predicted_load_normalized'] / 100)


print("✓ Predictions denormalized to target ISO scale")
print(f"\nPredicted load range: {predictions['predicted_load'].min():.2f} - {predictions['predicted_load'].max():.2f} MW")

# ====================================================================
# ALIGN PREDICTIONS TO ACTUAL TIME (shift values, not time)
# ====================================================================

# Shift predicted load values back 3 hours to align with actual
predictions['predicted_load'] = predictions['predicted_load'].shift(-3)
print("✓ Predicted load values shifted back 3 hours to align with actual load times")


# ====================================================================
# DISPLAY RESULTS
# ====================================================================
print("\n" + "=" * 80)
print("PREDICTION RESULTS")
print("=" * 80)

print("\nFirst 24 hours of predictions:")
print(predictions.head(24))

print("\nSummary Statistics:")
avg_predicted = predictions['predicted_load'].mean()
avg_forecast = predictions['forecast_load'].mean()
load_reduction_pct = ((avg_forecast - avg_predicted) / avg_forecast) * 100

print(f"  Average predicted load: {avg_predicted:.2f} MW")
print(f"  Average forecast load: {avg_forecast:.2f} MW")
print(f"  Load reduction: {load_reduction_pct:.2f}%")

# Save predictions
output_file = f'data/{location_name}_fire_predictions.csv'
predictions.to_csv(output_file, index=False)
print(f"\n✓ Predictions saved to '{output_file}'")




# ====================================================================
# PLOT PREDICTIONS
# ====================================================================
print("\n" + "=" * 80)
print("PLOTTING PREDICTIONS")
print("=" * 80)

predictions['Time'] = pd.to_datetime(predictions['Time'])

fig = go.Figure()

# Add actual load
fig.add_trace(go.Scatter(
    x=predictions['Time'],
    y=predictions.get('actual_load', None),
    name='Actual Load',
    mode='lines',
    line=dict(color='green', width=2)
))

# Add forecast load
fig.add_trace(go.Scatter(
    x=predictions['Time'],
    y=predictions['forecast_load'],
    name='Forecast Load',
    mode='lines',
    line=dict(color='blue', width=2)
))

# Add predicted load
fig.add_trace(go.Scatter(
    x=predictions['Time'],
    y=predictions['predicted_load'],
    name='Predicted Load',
    mode='lines',
    line=dict(color='red', width=2)
))

# Update layout
fig.update_layout(
    title=f'Fire Impact on Grid Load - {location_name}',
    xaxis_title='Time',
    yaxis_title='Load (MW)',
    hovermode='x unified',
    template='plotly_white',
    height=600,
    width=1200
)

# Save and show
plot_file = f'figs/{location_name}_fire_predictions.html'
fig.write_html(plot_file)
fig.show()

print(f"\n✓ Plot saved to '{plot_file}'")

