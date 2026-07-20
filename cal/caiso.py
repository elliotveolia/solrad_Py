import gridstatus
from datetime import datetime


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

