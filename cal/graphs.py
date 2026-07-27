import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the actual load data
actual_load = pd.read_csv('../data/caiso_actual_load.csv')
forecasted_load = pd.read_csv('../data/caiso_forecasted_load.csv')

# Convert Time column to datetime with UTC handling
actual_load['Time'] = pd.to_datetime(actual_load['Time'], utc=True)
forecasted_load['Time'] = pd.to_datetime(forecasted_load['Time'], utc=True)

# Convert to a common timezone (e.g., US/Pacific for CAISO)
actual_load['Time'] = actual_load['Time'].dt.tz_convert('US/Pacific')
forecasted_load['Time'] = forecasted_load['Time'].dt.tz_convert('US/Pacific')

# Extract year and month for grouping (without timezone to avoid warning)
actual_load['YearMonth'] = actual_load['Time'].dt.strftime('%Y-%m')
forecasted_load['YearMonth'] = forecasted_load['Time'].dt.strftime('%Y-%m')

# Get unique months and sort them
months = sorted(actual_load['YearMonth'].unique())

# Create figure
fig = go.Figure()

# Add traces for each month (initially all visible, we'll control with buttons)
for month in months:
    actual_month = actual_load[actual_load['YearMonth'] == month]
    forecasted_month = forecasted_load[forecasted_load['YearMonth'] == month]

    # Add actual load trace
    fig.add_trace(
        go.Scatter(
            x=actual_month['Time'],
            y=actual_month['Load'],
            mode='lines',
            name='Actual Load',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Actual Load</b><br>Time: %{x}<br>Load: %{y:.0f} MW<extra></extra>',
            visible=(month == months[0]),  # Only first month visible initially
            legendgroup='actual',
            showlegend=(month == months[0])  # Only show legend for first trace
        )
    )

    # Add forecasted load trace
    fig.add_trace(
        go.Scatter(
            x=forecasted_month['Time'],
            y=forecasted_month['Load Forecast'],
            mode='lines',
            name='Forecasted Load',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Forecasted Load</b><br>Time: %{x}<br>Load: %{y:.0f} MW<extra></extra>',
            visible=(month == months[0]),  # Only first month visible initially
            legendgroup='forecast',
            showlegend=(month == months[0])  # Only show legend for first trace
        )
    )

# Create buttons for month selection
buttons = []
for i, month in enumerate(months):
    # Create visibility list - each month has 2 traces (actual + forecast)
    visibility = [False] * len(fig.data)
    visibility[i * 2] = True  # Actual load for this month
    visibility[i * 2 + 1] = True  # Forecasted load for this month

    buttons.append(
        dict(
            label=month,
            method='update',
            args=[{'visible': visibility},
                  {'title': f'CAISO Actual vs Forecasted Load - {month}'}]
        )
    )

# Update layout with dropdown menu
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    ],
    title=f'CAISO Actual vs Forecasted Load - {months[0]}',
    xaxis_title='Date',
    yaxis_title='Load (MW)',
    hovermode='x unified',
    height=600,
    template='plotly_white',
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1
    ),
    margin=dict(t=100)  # Add top margin for dropdown
)

# Show plot
fig.show()

# Save as HTML
fig.write_html('../figs/caiso_load_comparison.html')
print("✓ Plot saved as caiso_load_comparison.html")
