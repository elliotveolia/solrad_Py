import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the actual load data
actual_load = pd.read_csv('../data/caiso_actual_load.csv')

# Load the forecasted load data
forecasted_load = pd.read_csv('../data/caiso_forecasted_load.csv')

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": False}]])

# Add actual load
fig.add_trace(
    go.Scatter(
        x=actual_load['Time'],
        y=actual_load['Load'],
        mode='lines',
        name='Actual Load',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Actual Load</b><br>Time: %{x}<br>Load: %{y:.0f} MW<extra></extra>'
    )
)

# Add forecasted load
fig.add_trace(
    go.Scatter(
        x=forecasted_load['Time'],
        y=forecasted_load['Load Forecast'],
        mode='lines',
        name='Forecasted Load',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Forecasted Load</b><br>Time: %{x}<br>Load: %{y:.0f} MW<extra></extra>'
    )
)

# Update layout
fig.update_layout(
    title='CAISO Actual vs Forecasted Load - 2020',
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
    )
)

# Show plot
fig.show()

# Save as HTML
fig.write_html('../figs/caiso_load_comparison.html')
print("✓ Plot saved as caiso_load_comparison.html")
