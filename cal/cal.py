import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import sys
from plotly.subplots import make_subplots
from caiso import fetch_caiso_load_data, get_california_weather, align_datasets
import numpy as np
from scipy import stats
import pandas as pd

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

# Define fire periods
fire_periods = [
    # 2020 Fires
    ('2020-08-16', '2020-11-30'),  # August Complex
    ('2020-09-04', '2020-12-24'),  # Creek Fire
    ('2020-08-19', '2020-10-16'),  # North Complex
    ('2020-08-16', '2020-10-15'),  # SCU Lightning Complex
    ('2020-08-16', '2020-10-16'),  # LNU Lightning Complex

    # 2021 Fires
    ('2021-07-13', '2021-10-25'),  # Dixie Fire
    ('2021-08-14', '2021-09-21'),  # Caldor Fire
    ('2021-07-30', '2021-09-15'),  # Tamarack Fire

    # 2022 Fires
    ('2022-09-05', '2022-10-05'),  # Mosquito Fire
    ('2022-07-08', '2022-08-15'),  # Fairview Fire

    # 2023 Fires
    ('2023-08-15', '2023-11-17'),  # Park Fire (started late July, major activity Aug)
    ('2023-09-09', '2023-10-12'),  # Coastal Fire

    # 2024 Fires
    ('2024-07-24', '2024-09-26'),  # Park Fire (2024 continuation/major phase)
    ('2024-06-22', '2024-08-10'),  # Borel Fire

    # 2025 Fires
    ('2025-01-07', '2025-01-31'),  # Current/Recent fires
]


def quantify_fire_load_impact(actual_load, forecasted_load, weather_load, fire_periods, days=10):
    """
    Quantify load drop using only PRE-FIRE forecasts.
    Compare actual load during fire to forecasts made BEFORE fire started.
    """

    weather_load['time'] = pd.to_datetime(weather_load['time'], utc=True)
    weather_load['month_day'] = weather_load['time'].dt.strftime('%m-%d')
    weather_load['year'] = weather_load['time'].dt.year

    results = []

    for fire_start, fire_end in fire_periods:
        fire_start = pd.to_datetime(fire_start, utc=True)
        fire_end_limited = fire_start + pd.Timedelta(days=days - 1)

        # Get actual load during fire
        actual_fire = actual_load[(actual_load['Time'] >= fire_start) &
                                  (actual_load['Time'] <= fire_end_limited)].copy()
        actual_fire['Time'] = pd.to_datetime(actual_fire['Time'], utc=True)
        actual_hourly = actual_fire.set_index('Time').resample('1h').agg({'Load': 'mean'}).reset_index()

        # Get forecasts made BEFORE fire started (1-2 days before)
        forecast_cutoff = fire_start - pd.Timedelta(days=1)
        forecast_fire = forecasted_load[(forecasted_load['Time'] >= fire_start) &
                                        (forecasted_load['Time'] <= fire_end_limited) &
                                        (pd.to_datetime(forecasted_load['Publish Time'], utc=True) < forecast_cutoff)]

        # Get fire period weather
        fire_weather = weather_load[(weather_load['time'] >= fire_start) &
                                    (weather_load['time'] <= fire_end_limited)]

        # Get baseline weather - same calendar dates in other years
        fire_month_day_range = set()
        for i in range(days):
            date = fire_start + pd.Timedelta(days=i)
            fire_month_day_range.add(date.strftime('%m-%d'))

        baseline_weather = weather_load[
            (weather_load['month_day'].isin(fire_month_day_range)) &
            (weather_load['year'] != fire_start.year)
            ]

        if len(forecast_fire) == 0:
            print(f"No pre-fire forecasts for {fire_start.date()}")
            continue

        actual_mean = actual_hourly['Load'].mean()
        forecast_mean = forecast_fire['Load Forecast'].mean()
        load_drop_pct = ((forecast_mean - actual_mean) / forecast_mean) * 100

        fire_cloudcover = fire_weather['cloudcover'].mean() if len(fire_weather) > 0 else None
        baseline_cloudcover = baseline_weather['cloudcover'].mean() if len(baseline_weather) > 0 else None

        # Calculate fire over baseline cloudcover ratio
        if fire_cloudcover is not None and baseline_cloudcover is not None and baseline_cloudcover != 0:
            fire_over_base_cover = fire_cloudcover / baseline_cloudcover
        else:
            fire_over_base_cover = None

        results.append({
            'fire_start': fire_start,
            'fire_end_analyzed': fire_end_limited,
            'duration_days': days,
            'actual_load': actual_mean,
            'forecast_load': forecast_mean,
            'load_drop_pct': load_drop_pct,
            'fire_cloudcover': fire_cloudcover,
            'baseline_cloudcover': baseline_cloudcover,
            'fire_over_base_cover': fire_over_base_cover,
            'forecast_data_points': len(forecast_fire)
        })

    results_df = pd.DataFrame(results)
    print(f"FIRE IMPACT ANALYSIS (First {days} days - Pre-fire forecasts only):")
    print(results_df.to_string())
    print(f"\nAverage load drop: {results_df['load_drop_pct'].mean():.2f}%")

    return results_df

fire_impact = quantify_fire_load_impact(actual_load, forecasted_load, weather_load, fire_periods, days=5)

sys.exit()

def analyze_driving_variables(fire_impact_df):
    """
    Analyze correlations between load drop and weather variables.
    """

    # Remove rows with NaN values for analysis
    analysis_df = fire_impact_df.dropna(subset=['load_drop_pct', 'fire_cloudcover', 'baseline_cloudcover'])

    if len(analysis_df) == 0:
        print("No complete data for analysis")
        return

    print("=" * 80)
    print("DRIVING VARIABLES ANALYSIS")
    print("=" * 80)

    # 1. Cloud Cover Analysis
    print("\n1. CLOUD COVER IMPACT:")
    print("-" * 80)

    cloudcover_diff = analysis_df['fire_cloudcover'] - analysis_df['baseline_cloudcover']
    corr_cloudcover, p_value_cc = stats.pearsonr(cloudcover_diff, analysis_df['load_drop_pct'])

    print(f"Cloud cover difference (fire - baseline):")
    print(f"  Mean: {cloudcover_diff.mean():.2f}%")
    print(f"  Std: {cloudcover_diff.std():.2f}%")
    print(f"\nCorrelation with load drop: {corr_cloudcover:.4f}")
    print(f"P-value: {p_value_cc:.4f}")
    print(f"Significance: {'***' if p_value_cc < 0.05 else 'Not significant'}")

    # 2. Cloud Cover Ratio Analysis
    print("\n2. CLOUD COVER RATIO (Fire/Baseline):")
    print("-" * 80)

    ratio_data = analysis_df.dropna(subset=['fire_over_base_cover'])
    if len(ratio_data) > 0:
        corr_ratio, p_value_ratio = stats.pearsonr(ratio_data['fire_over_base_cover'],
                                                   ratio_data['load_drop_pct'])
        print(f"Mean ratio: {ratio_data['fire_over_base_cover'].mean():.4f}")
        print(f"Correlation with load drop: {corr_ratio:.4f}")
        print(f"P-value: {p_value_ratio:.4f}")
        print(f"Significance: {'***' if p_value_ratio < 0.05 else 'Not significant'}")

    # 3. Summary Statistics
    print("\n3. SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"\nLoad Drop %:")
    print(f"  Mean: {analysis_df['load_drop_pct'].mean():.2f}%")
    print(f"  Std: {analysis_df['load_drop_pct'].std():.2f}%")
    print(f"  Min: {analysis_df['load_drop_pct'].min():.2f}%")
    print(f"  Max: {analysis_df['load_drop_pct'].max():.2f}%")

    print(f"\nFire Cloud Cover:")
    print(f"  Mean: {analysis_df['fire_cloudcover'].mean():.2f}%")
    print(f"  Std: {analysis_df['fire_cloudcover'].std():.2f}%")

    print(f"\nBaseline Cloud Cover:")
    print(f"  Mean: {analysis_df['baseline_cloudcover'].mean():.2f}%")
    print(f"  Std: {analysis_df['baseline_cloudcover'].std():.2f}%")

    # 4. Correlation Matrix
    print("\n4. CORRELATION MATRIX:")
    print("-" * 80)

    corr_cols = ['load_drop_pct', 'fire_cloudcover', 'baseline_cloudcover', 'fire_over_base_cover']
    corr_matrix = analysis_df[corr_cols].corr()
    print(corr_matrix)

    # 5. Linear Regression
    print("\n5. LINEAR REGRESSION ANALYSIS:")
    print("-" * 80)

    from sklearn.linear_model import LinearRegression

    X = analysis_df[['fire_cloudcover', 'baseline_cloudcover']].values
    y = analysis_df['load_drop_pct'].values

    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)

    print(
        f"Model: load_drop_pct = {model.intercept_:.4f} + {model.coef_[0]:.4f}*fire_cloudcover + {model.coef_[1]:.4f}*baseline_cloudcover")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Coefficients:")
    print(f"  Fire cloud cover: {model.coef_[0]:.6f}")
    print(f"  Baseline cloud cover: {model.coef_[1]:.6f}")

    return analysis_df, model


# Run analysis
analysis_results, regression_model = analyze_driving_variables(fire_impact)



