import gridstatus
from datetime import datetime
import pandas as pd

# Initialize CAISO client
caiso = gridstatus.CAISO()

t1 = datetime(2020, 1, 1)
t2 = datetime(2025, 12, 31)

print("Fetching CAISO data for 2020...")

try:
    # Get ACTUAL load data
    print("\n1. Fetching actual load data...")
    actual_load = caiso.get_load(
        start=t1,
        end=t2
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
        start=t1,
        end=t2
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

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
