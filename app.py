import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots

st.set_page_config(page_title="AQI Dashboard", layout="wide")
st.title("🌫️ AQI Anomaly Detection Dashboard")

# -------------------------------
# LOAD DATA (CACHED)
# -------------------------------
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file, parse_dates=['Date'])
    else:
        return pd.read_csv("city_day.csv", parse_dates=['Date'])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded_file)

# -------------------------------
# CLEAN
# -------------------------------
df = df.sort_values(['City', 'Date'])
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

valid_cities = [c for c in df['City'].unique() if len(df[df['City']==c]) > 150]
selected_city = st.sidebar.selectbox("Select City", valid_cities)

city_df = df[df['City']==selected_city].copy()
city_df['AQI'] = city_df['AQI'].interpolate()

# -------------------------------
# AQI CATEGORY FUNCTION
# -------------------------------
def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

city_df['category'] = city_df['AQI'].apply(aqi_category)

# -------------------------------
# TREND GRAPH
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
detect_df = detect_df.dropna()

detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']
detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3

scaler = StandardScaler()
scaled = scaler.fit_transform(detect_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
detect_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# SEVERITY + REASON
# -------------------------------
def severity(aqi):
    if aqi > 300: return "High"
    elif aqi > 200: return "Medium"
    else: return "Low"

def reason(aqi, z):
    if aqi > 300: return "Severe pollution spike"
    elif aqi > 200: return "High pollution level"
    elif z < -3: return "Sudden drop"
    else: return "Unusual variation"

detect_df['severity'] = detect_df['AQI'].apply(severity)
detect_df['reason'] = detect_df.apply(lambda r: reason(r['AQI'], r['z']), axis=1)

# -------------------------------
# DECOMPOSITION
# -------------------------------
st.subheader("📊 Seasonal Decomposition")

ts_df = city_df.set_index('Date')[['AQI']]
ts_df['AQI'] = ts_df['AQI'].ffill().bfill()

if len(ts_df) > 365:

    result = seasonal_decompose(ts_df['AQI'], model='additive', period=365)

    decomp_df = pd.DataFrame({
        'Date': ts_df.index,
        'Observed': result.observed,
        'Trend': result.trend,
        'Seasonal': result.seasonal,
        'Residual': result.resid
    }).dropna()

    merged = pd.merge(
        decomp_df,
        detect_df,
        on='Date',
        how='inner'
    )

    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True)

    fig2.add_trace(go.Scatter(x=merged['Date'], y=merged['Observed']), row=1, col=1)

    anomalies = merged[merged['iso_anomaly']]

    fig2.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['Observed'],
        mode='markers',
        marker=dict(color='red', size=7),
        text=anomalies['reason'],
        customdata=anomalies['severity'],
        hovertemplate="Date:%{x}<br>AQI:%{y}<br>Severity:%{customdata}<br>%{text}<extra></extra>"
    ), row=1, col=1)

    fig2.add_trace(go.Scatter(x=merged['Date'], y=merged['Trend']), row=2, col=1)
    fig2.add_trace(go.Scatter(x=merged['Date'], y=merged['Seasonal']), row=3, col=1)
    fig2.add_trace(go.Scatter(x=merged['Date'], y=merged['Residual']), row=4, col=1)

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# ANOMALY GRAPH
# -------------------------------
st.subheader("🔍 Anomaly Detection")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=detect_df['Date'], y=detect_df['AQI']))

fig3.add_trace(go.Scatter(
    x=detect_df[detect_df['iso_anomaly']]['Date'],
    y=detect_df[detect_df['iso_anomaly']]['AQI'],
    mode='markers',
    marker=dict(color='red', size=8),
    text=detect_df[detect_df['iso_anomaly']]['reason'],
    hovertemplate="Date:%{x}<br>AQI:%{y}<br>%{text}<extra></extra>"
))

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# INSIGHTS (CORRECTED)
# -------------------------------
st.subheader("💡 Smart Insights")

avg_aqi = city_df['AQI'].mean()
max_aqi = city_df['AQI'].max()
most_common = city_df['category'].mode()[0]

st.write(f"📍 City: {selected_city}")
st.write(f"📊 Average AQI: {avg_aqi:.2f}")
st.write(f"🚨 Max AQI: {max_aqi}")
st.write(f"📌 Most Frequent Category: {most_common}")
st.write(f"🚨 Total Anomalies: {detect_df['iso_anomaly'].sum()}")

if max_aqi > 300:
    st.error("Severe pollution spikes detected 🚨")
elif avg_aqi > 200:
    st.error("Unhealthy air quality")
elif avg_aqi > 100:
    st.warning("Moderate pollution")
else:
    st.success("Good air quality")
