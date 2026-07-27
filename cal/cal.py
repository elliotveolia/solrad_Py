import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import sys
from plotly.subplots import make_subplots
from caiso import fetch_caiso_load_data, get_california_weather, align_datasets

start = datetime(2020, 1, 1)
end = datetime(2025, 12, 31)

#Generate data sets
#actual_data, forcast_data = fetch_caiso_load_data(start, end)
#weather_data = get_california_weather(start, end)

# Load the actual load data
actual_load = pd.read_csv('../data/caiso_actual_load.csv')
forecasted_load = pd.read_csv('../data/caiso_forecasted_load.csv')
weather_load = pd.read_csv('../data/california_weather.csv')


# Run alignment
aligned_data = align_datasets(actual_load, forecasted_load, weather_load)

print(aligned_data)

