import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")
st.title("🌫️ AQI Anomaly Detection Dashboard")

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

# -------------------------------
# SELECT CITY
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", df['City'].unique())

city_df = df[df['City'] == selected_city].copy()

# Fill missing values safely
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# GRAPH 1: AQI TREND
# -------------------------------
st.subheader(f"📈 AQI Trend - {selected_city}")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=city_df['Date'],
    y=city_df['AQI'],
    name='AQI'
))

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
detect_df = city_df.copy()

# Rolling stats (SAFE)
detect_df['mean'] = detect_df['AQI'].rolling(30, min_periods=10).mean()
detect_df['std'] = detect_df['AQI'].rolling(30, min_periods=10).std()

detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']
detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3

# Isolation Forest
scaler = StandardScaler()
scaled = scaler.fit_transform(detect_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
detect_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# GRAPH 2: ANOMALIES
# -------------------------------
st.subheader("🔍 Anomaly Detection")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=detect_df['Date'],
    y=detect_df['AQI'],
    name='AQI'
))

# Isolation Forest anomalies
fig2.add_trace(go.Scatter(
    x=detect_df[detect_df['iso_anomaly']]['Date'],
    y=detect_df[detect_df['iso_anomaly']]['AQI'],
    mode='markers',
    marker=dict(color='red', size=8),
    name='Isolation Forest'
))

# Z-score anomalies
fig2.add_trace(go.Scatter(
    x=detect_df[detect_df['z_anomaly']]['Date'],
    y=detect_df[detect_df['z_anomaly']]['AQI'],
    mode='markers',
    marker=dict(color='orange', size=6),
    name='Z-score'
))

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("💡 Insights")

avg_aqi = city_df['AQI'].mean()
max_aqi = city_df['AQI'].max()

st.write(f"📍 City: {selected_city}")
st.write(f"📊 Average AQI: {avg_aqi:.2f}")
st.write(f"🚨 Max AQI: {max_aqi}")
st.write(f"🔴 Isolation Forest Anomalies: {detect_df['iso_anomaly'].sum()}")
st.write(f"🟠 Z-score Anomalies: {detect_df['z_anomaly'].sum()}")

# Smart interpretation
if max_aqi > 300:
    st.error("Severe pollution spikes detected 🚨")
elif avg_aqi > 200:
    st.error("Unhealthy air quality")
elif avg_aqi > 100:
    st.warning("Moderate pollution")
else:
    st.success("Good air quality")
