import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")
st.title("🌫️ AQI Anomaly Detection Dashboard (Advanced)")

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

# -------------------------------
# CLEAN DATA
# -------------------------------
df = df.sort_values(['City', 'Date'])
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df = df.dropna(subset=['AQI'])

# -------------------------------
# FILTER CITIES
# -------------------------------
valid_cities = [
    city for city in df['City'].unique()
    if len(df[df['City'] == city]) > 150
]

selected_city = st.sidebar.selectbox("Select City", valid_cities)

city_df = df[df['City'] == selected_city].copy()
city_df['AQI'] = city_df['AQI'].interpolate()

# -------------------------------
# AQI TREND
# -------------------------------
st.subheader(f"📈 AQI Trend - {selected_city}")

trend_df = city_df.copy()
trend_df['rolling'] = trend_df['AQI'].rolling(30, min_periods=10).mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['AQI'], name='AQI'))
fig1.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['rolling'], name='30-Day Avg'))

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
detect_df = city_df.copy()

detect_df['mean'] = detect_df['AQI'].rolling(30, min_periods=10).mean()
detect_df['std'] = detect_df['AQI'].rolling(30, min_periods=10).std()

detect_df = detect_df.dropna(subset=['mean', 'std'])

detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']
detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3

# Isolation Forest
scaler = StandardScaler()
scaled = scaler.fit_transform(detect_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
detect_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# SEASONAL DECOMPOSITION (INTERACTIVE)
# -------------------------------
st.subheader("📊 Seasonal Decomposition (Interactive)")

ts_df = city_df.set_index('Date')

if len(ts_df) > 365:

    result = seasonal_decompose(ts_df['AQI'], model='additive', period=365)

    decomp_df = pd.DataFrame({
        'Date': ts_df.index,
        'Observed': result.observed,
        'Trend': result.trend,
        'Seasonal': result.seasonal,
        'Residual': result.resid
    }).dropna()

    # -------------------------------
    # PLOTLY SUBPLOTS
    # -------------------------------
    from plotly.subplots import make_subplots

    fig2 = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed AQI", "Trend", "Seasonal", "Residual")
    )

    # Observed
    fig2.add_trace(go.Scatter(
        x=decomp_df['Date'], y=decomp_df['Observed'],
        name='Observed'
    ), row=1, col=1)

    # Highlight anomalies
    anomaly_dates = detect_df[detect_df['iso_anomaly']]['Date']

    fig2.add_trace(go.Scatter(
        x=anomaly_dates,
        y=decomp_df.set_index('Date').loc[anomaly_dates]['Observed'],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Anomaly'
    ), row=1, col=1)

    # Trend
    fig2.add_trace(go.Scatter(
        x=decomp_df['Date'], y=decomp_df['Trend'],
        name='Trend'
    ), row=2, col=1)

    # Seasonal
    fig2.add_trace(go.Scatter(
        x=decomp_df['Date'], y=decomp_df['Seasonal'],
        name='Seasonal'
    ), row=3, col=1)

    # Residual
    fig2.add_trace(go.Scatter(
        x=decomp_df['Date'], y=decomp_df['Residual'],
        name='Residual'
    ), row=4, col=1)

    fig2.update_layout(height=900, showlegend=False)

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("Need at least 1 year data for decomposition")

# -------------------------------
# ANOMALY VISUALIZATION
# -------------------------------
st.subheader("🔍 Anomaly Detection Overview")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=detect_df['Date'], y=detect_df['AQI'],
    name='AQI'
))

fig3.add_trace(go.Scatter(
    x=detect_df[detect_df['iso_anomaly']]['Date'],
    y=detect_df[detect_df['iso_anomaly']]['AQI'],
    mode='markers',
    marker=dict(color='red', size=7),
    name='Isolation Forest'
))

fig3.add_trace(go.Scatter(
    x=detect_df[detect_df['z_anomaly']]['Date'],
    y=detect_df[detect_df['z_anomaly']]['AQI'],
    mode='markers',
    marker=dict(color='orange', size=7),
    name='Z-score'
))

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("💡 Insights")

st.markdown(f"""
- Selected City: **{selected_city}**
- Total Data Points: {len(city_df)}
- Isolation Forest Anomalies: {detect_df['iso_anomaly'].sum()}
- Z-score Anomalies: {detect_df['z_anomaly'].sum()}

### Interpretation:
- Trend shows long-term pollution change  
- Seasonal shows recurring patterns (winter pollution spikes)  
- Residual shows unexpected events  
- Red dots = critical anomalies  
""")
