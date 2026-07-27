import gridstatus
from datetime import datetime
import pandas as pd
import requests


def fetch_caiso_load_data(start_date, end_date):
    """
    Fetch CAISO actual and forecasted load data

    Args:
        start_date: datetime or str (YYYY-MM-DD)
        end_date: datetime or str (YYYY-MM-DD)

    Returns:
        tuple: (actual_load_df, forecasted_load_df)
    """

    print("Fetching CAISO data...")

    # Convert strings to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Initialize CAISO client
    caiso = gridstatus.CAISO()

    try:
        # Get ACTUAL load data
        print("\n1. Fetching actual load data...")
        actual_load = caiso.get_load(
            start=start_date,
            end=end_date
        )

        print(f"✓ Actual load loaded!")
        print(f"  Shape: {actual_load.shape}")
        print(f"  Columns: {actual_load.columns.tolist()}")
        print(actual_load.head())

        # Save as CSV
        actual_load.to_csv('../data/model/caiso_actual_load.csv', index=False)
        print("  ✓ Saved to caiso_actual_load.csv")

        # Get FORECASTED/PREDICTED load data
        print("\n2. Fetching forecasted load data...")
        forecasted_load = caiso.get_load_forecast(
            start=start_date,
            end=end_date
        )

        print(f"✓ Forecasted load loaded!")
        print(f"  Shape: {forecasted_load.shape}")
        print(f"  Columns: {forecasted_load.columns.tolist()}")
        print(forecasted_load.head())

        # Save as CSV
        forecasted_load.to_csv('../data/model/caiso_forecasted_load.csv', index=False)
        print("  ✓ Saved to caiso_forecasted_load.csv")

        # Compare them
        print("\n3. Comparing actual vs forecasted...")
        print(f"\nActual load stats:")
        print(actual_load.describe())

        print(f"\nForecasted load stats:")
        print(forecasted_load.describe())

        return actual_load, forecasted_load

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_california_weather(start_date, end_date, lat=37.2808, lon=-119.2945,
                           save_path='../data/model/california_weather.csv'):
    """
    Fetch weather data + smoke/AQI proxy + solar irradiance.
    """

    try:
        if not isinstance(start_date, str):
            start_date = start_date.strftime('%Y-%m-%d')
        if not isinstance(end_date, str):
            end_date = end_date.strftime('%Y-%m-%d')

        print(f"Fetching weather data from {start_date} to {end_date}...")

        # WEATHER DATA - Open-Meteo API with solar radiation
        weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,cloudcover,precipitation,relative_humidity_2m,shortwave_radiation&timezone=America/Los_Angeles"

        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()

        if weather_response.status_code != 200:
            print(f"Weather Error")
            return None

        # Convert weather to DataFrame
        df = pd.DataFrame({
            'time': weather_data['hourly']['time'],
            'temperature_celsius': weather_data['hourly']['temperature_2m'],
            'cloudcover': weather_data['hourly']['cloudcover'],
            'precipitation': weather_data['hourly']['precipitation'],
            'humidity': weather_data['hourly']['relative_humidity_2m'],
            'solar_irradiance': weather_data['hourly']['shortwave_radiation']  # NEW
        })

        # Convert temperature to Fahrenheit
        df['temperature'] = (df['temperature_celsius'] * 9 / 5) + 32
        df = df.drop('temperature_celsius', axis=1)

        # Create smoke AQI proxy (combination of cloudcover and humidity)
        df['smoke_aqi_proxy'] = (df['cloudcover'] * 0.6 + df['humidity'] * 0.4)

        # Extract hour of day for time-of-day analysis
        df['hour'] = pd.to_datetime(df['time']).dt.hour

        # Reorder columns
        df = df[['time', 'hour', 'temperature', 'cloudcover', 'humidity', 'precipitation',
                 'solar_irradiance', 'smoke_aqi_proxy']]

        df.to_csv(save_path, index=False)
        print(f"\nRetrieved {len(df)} hours of data")
        print(f"Data saved to {save_path}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"Temperature range: {df['temperature'].min():.1f}°F to {df['temperature'].max():.1f}°F")
        print(f"Solar irradiance range: {df['solar_irradiance'].min():.1f} to {df['solar_irradiance'].max():.1f} W/m²")

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
