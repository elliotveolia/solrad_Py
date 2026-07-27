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
        actual_load.to_csv('../data/caiso_actual_load.csv', index=False)
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
        forecasted_load.to_csv('../data/caiso_forecasted_load.csv', index=False)
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


def get_california_weather(start_date, end_date, lat=37.2808, lon=-119.2945, save_path='../data/california_weather.csv'):
    """
    Fetch historical weather data for central California from Meteostat and save to file.

    Parameters:
    -----------
    start_date : str or datetime
        Start date in format 'YYYY-MM-DD' or datetime object
    end_date : str or datetime
        End date in format 'YYYY-MM-DD' or datetime object
    lat : float
        Latitude (default: Central California)
    lon : float
        Longitude (default: Central California)
    save_path : str
        File path to save data (default: 'california_weather.csv')

    Returns:
    --------
    pd.DataFrame : Historical weather data
    """

    try:
        # Make sure dates are strings in YYYY-MM-DD format
        if not isinstance(start_date, str):
            start_date = start_date.strftime('%Y-%m-%d')
        if not isinstance(end_date, str):
            end_date = end_date.strftime('%Y-%m-%d')

        print(f"Fetching hourly data from {start_date} to {end_date}...")

        # Open-Meteo API - hourly data
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,cloudcover,precipitation&timezone=America/Los_Angeles"

        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            print(f"Error: {data.get('reason', data.get('error', 'Unknown error'))}")
            return None

        if 'hourly' not in data:
            print("No hourly data in response")
            return None

        # Convert to DataFrame
        df = pd.DataFrame({
            'time': data['hourly']['time'],
            'temperature': data['hourly']['temperature_2m'],
            'cloudcover': data['hourly']['cloudcover'],
            'precipitation': data['hourly']['precipitation']
        })

        df.to_csv(save_path, index=False)
        print(f"Retrieved {len(df)} hours of data")
        print(f"Weather data saved to {save_path}")

        return df

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def align_datasets(actual_load, forecasted_load, weather_load, debug = False):
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
    weather_hourly.columns = ['Time', 'temperature', 'cloudcover', 'precipitation']

    # Merge all datasets on Time
    aligned_data = actual_hourly.copy()
    aligned_data = aligned_data.merge(forecasted_hourly, on='Time', how='left')
    aligned_data = aligned_data.merge(weather_hourly, on='Time', how='left')

    # Sort by time
    aligned_data = aligned_data.sort_values('Time').reset_index(drop=True)
    if debug:
        print(f"Aligned dataset shape: {aligned_data.shape}")
        print(f"Time range: {aligned_data['Time'].min()} to {aligned_data['Time'].max()}")
        print(f"\nAligned data:\n{aligned_data.head(10)}")
        print(f"\nMissing values:\n{aligned_data.isnull().sum()}")

    aligned_data.to_csv('../data/aligned_data_hourly.csv', index=False)

    return aligned_data