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

        print(f"Fetching data from {start_date} to {end_date}...")

        # Open-Meteo API (free, no key required)
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,cloudcover_mean,precipitation_sum&timezone=America/Los_Angeles"

        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            print(f"Error: {data.get('reason', data.get('error', 'Unknown error'))}")
            return None

        if 'daily' not in data:
            print("No daily data in response")
            return None

        # Convert to DataFrame
        df = pd.DataFrame({
            'date': data['daily']['time'],
            'temp_max': data['daily']['temperature_2m_max'],
            'temp_min': data['daily']['temperature_2m_min'],
            'cloudcover': data['daily']['cloudcover_mean'],
            'precipitation': data['daily']['precipitation_sum']
        })

        df.to_csv(save_path, index=False)
        print(f"Retrieved {len(df)} days of data")
        print(f"Weather data saved to {save_path}")

        return df

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None