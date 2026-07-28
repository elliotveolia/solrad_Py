from ice_data_py import cds
from datetime import datetime
import pandas as pd
import requests
import os


def fetch_load_data(start_date, end_date):
    """
    Fetch PJM actual and forecasted load data using Ice Data Center
    """

    print("Fetching data from Ice Data Center...")

    os.makedirs('data', exist_ok=True)

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    try:
        # Get ACTUAL load data
        print("\n1. Fetching actual load data...")
        actual_load = cds.fetch_all([
            cds.QuerySpec(
                ms="PJMISO",
                mn="locational_load.rto_combined",
                mp="instantaneous_load"
            )
        ], start_date, end_date)

        # Convert Polars to Pandas
        actual_load_df = pd.DataFrame({col: actual_load[col].to_numpy() for col in actual_load.columns})
        actual_load_df.columns = ['Time', 'Load']

        print(f"✓ Actual load loaded! Shape: {actual_load_df.shape}")
        print(actual_load_df.head())

        # Get FORECASTED load data
        print("\n2. Fetching forecasted load data...")
        forecasted_load = cds.fetch_all([
            cds.QuerySpec(
                ms="PJMISO",
                mn="locational_load.rto_combined",
                mp="originalforecast_load"
            )
        ], start_date, end_date)

        # Convert Polars to Pandas
        forecasted_load_df = pd.DataFrame({col: forecasted_load[col].to_numpy() for col in forecasted_load.columns})
        forecasted_load_df.columns = ['Time', 'Load Forecast']

        print(f"✓ Forecasted load loaded! Shape: {forecasted_load_df.shape}")
        print(forecasted_load_df.head())

        return actual_load_df, forecasted_load_df

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_weather_data(start_date, end_date, lat, lon, location_name="Location",
                     timezone="UTC", save_path=None):
    """
    Fetch weather data for any location using Open-Meteo API.
    """

    try:
        if not isinstance(start_date, str):
            start_date = start_date.strftime('%Y-%m-%d')
        if not isinstance(end_date, str):
            end_date = end_date.strftime('%Y-%m-%d')

        print(f"Fetching weather data for {location_name}")
        print(f"Location: ({lat}, {lon})")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Timezone: {timezone}\n")

        # WEATHER DATA - Open-Meteo API with solar radiation
        weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,cloudcover,precipitation,relative_humidity_2m,shortwave_radiation&timezone={timezone}"

        print(f"Fetching from API...")
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()

        if weather_response.status_code != 200:
            print(f"Weather Error: {weather_response.status_code}")
            return None

        if 'hourly' not in weather_data:
            print("No hourly data in response")
            return None

        # Convert weather to DataFrame
        df = pd.DataFrame({
            'time': pd.to_datetime(weather_data['hourly']['time']),
            'temperature_celsius': weather_data['hourly']['temperature_2m'],
            'cloudcover': weather_data['hourly']['cloudcover'],
            'precipitation': weather_data['hourly']['precipitation'],
            'humidity': weather_data['hourly']['relative_humidity_2m'],
            'solar_irradiance': weather_data['hourly']['shortwave_radiation']
        })

        # Convert temperature to Fahrenheit
        df['temperature'] = (df['temperature_celsius'] * 9 / 5) + 32
        df = df.drop('temperature_celsius', axis=1)

        # Create smoke AQI proxy
        df['smoke_aqi_proxy'] = (df['cloudcover'] * 0.6 + df['humidity'] * 0.4)

        # Extract hour of day
        df['hour'] = df['time'].dt.hour

        # Reorder columns
        df = df[['time', 'hour', 'temperature', 'cloudcover', 'humidity', 'precipitation',
                 'solar_irradiance', 'smoke_aqi_proxy']]

        # Save to CSV if path provided
        if save_path is None:
            os.makedirs('data', exist_ok=True)
            save_path = os.path.join('data', f'{location_name}_weather.csv')

        df.to_csv(save_path, index=False)

        print(f"✓ Retrieved {len(df)} hours of data")
        print(f"✓ Data saved to {save_path}")

        return df

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def align_datasets(actual_load, forecasted_load, weather_load):
    """
    Align actual load, forecasted load, and weather data by hour.
    All datasets converted to hourly.
    """

    # Convert Time columns to datetime (utc=True to handle timezone)
    actual_load['Time'] = pd.to_datetime(actual_load['Time'], utc=True)
    forecasted_load['Time'] = pd.to_datetime(forecasted_load['Time'], utc=True)
    weather_load['time'] = pd.to_datetime(weather_load['time'], utc=True)

    # Convert actual load to hourly (mean of 5-min intervals)
    actual_hourly = actual_load.set_index('Time').resample('1h').agg({
        'Load': ['mean', 'max', 'min']
    }).reset_index()
    actual_hourly.columns = ['Time', 'actual_load_mean', 'actual_load_max', 'actual_load_min']

    # Forecasted load is already hourly
    forecasted_hourly = forecasted_load[['Time', 'Load Forecast']].copy()
    forecasted_hourly.columns = ['Time', 'forecast_load']

    # Weather is already hourly, just rename
    weather_hourly = weather_load.copy()
    weather_hourly.columns = ['Time', 'hour', 'temperature', 'cloudcover', 'humidity', 'precipitation',
                              'solar_irradiance', 'smoke_aqi_proxy']

    # Merge all datasets on Time
    aligned_data = actual_hourly.copy()
    aligned_data = aligned_data.merge(forecasted_hourly, on='Time', how='left')
    aligned_data = aligned_data.merge(weather_hourly, on='Time', how='left')

    # Sort by time
    aligned_data = aligned_data.sort_values('Time').reset_index(drop=True)

    print(f"Aligned dataset shape: {aligned_data.shape}")
    print(f"Time range: {aligned_data['Time'].min()} to {aligned_data['Time'].max()}")
    print(f"\nAligned data:\n{aligned_data.head(10)}")
    print(f"\nMissing values:\n{aligned_data.isnull().sum()}")
    print(f"\nColumns: {aligned_data.columns.tolist()}")

    return aligned_data