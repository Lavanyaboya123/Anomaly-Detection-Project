import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ AQI Anomaly Detection (Final Corrected Version)")

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

# -------------------------------
# CLEAN DATA (IMPORTANT FIX)
# -------------------------------
df = df.sort_values(['City', 'Date'])
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

# Remove only invalid AQI rows
df = df.dropna(subset=['AQI'])

# -------------------------------
# FILTER VALID CITIES
# -------------------------------
valid_cities = []

for city in df['City'].unique():
    temp = df[df['City'] == city]

    if len(temp) > 150:
        valid_cities.append(city)

if len(valid_cities) == 0:
    st.error("No valid cities available")
    st.stop()

# -------------------------------
# SELECT CITY
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", valid_cities)

city_df = df[df['City'] == selected_city].copy()
city_df = city_df.sort_values('Date')

# Light interpolation (DO NOT overfill)
city_df['AQI'] = city_df['AQI'].interpolate()

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend", "🔍 Detection", "🤖 KNN", "📊 Compare"
])

# ===============================
# 📈 TREND
# ===============================
with tab1:
    trend_df = city_df.copy()

    trend_df['rolling'] = trend_df['AQI'].rolling(
        window=30, min_periods=10
    ).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df['Date'], y=trend_df['AQI'],
        name='AQI', line=dict(color='cyan')
    ))

    fig.add_trace(go.Scatter(
        x=trend_df['Date'], y=trend_df['rolling'],
        name='30-Day Avg', line=dict(color='yellow')
    ))

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 🔍 DETECTION (Z + ISO)
# ===============================
with tab2:
    detect_df = city_df.copy()

    detect_df['mean'] = detect_df['AQI'].rolling(
        30, min_periods=10
    ).mean()

    detect_df['std'] = detect_df['AQI'].rolling(
        30, min_periods=10
    ).std()

    # Keep only valid rows for detection
    detect_df = detect_df.dropna(subset=['mean', 'std'])

    detect_df['z'] = (
        (detect_df['AQI'] - detect_df['mean']) /
        detect_df['std']
    )

    detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3

    # Isolation Forest
    scaler = StandardScaler()
    scaled = scaler.fit_transform(detect_df[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
    detect_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=detect_df['Date'], y=detect_df['AQI'],
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=detect_df[detect_df['iso_anomaly']]['Date'],
        y=detect_df[detect_df['iso_anomaly']]['AQI'],
        mode='markers',
        marker=dict(color='red', size=7),
        name='Isolation Forest'
    ))

    fig.add_trace(go.Scatter(
        x=detect_df[detect_df['z_anomaly']]['Date'],
        y=detect_df[detect_df['z_anomaly']]['AQI'],
        mode='markers',
        marker=dict(color='orange', size=7),
        name='Z-score'
    ))

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 🤖 KNN
# ===============================
with tab3:
    knn_df = city_df.copy()
    knn_df = knn_df.dropna(subset=['AQI'])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(knn_df[['AQI']])

    knn = KNN(contamination=0.05)
    knn_df['knn_anomaly'] = knn.fit_predict(scaled) == 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=knn_df['Date'], y=knn_df['AQI'],
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=knn_df[knn_df['knn_anomaly']]['Date'],
        y=knn_df[knn_df['knn_anomaly']]['AQI'],
        mode='markers',
        marker=dict(color='green', size=7),
        name='KNN'
    ))

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 📊 COMPARE
# ===============================
with tab4:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=city_df['Date'], y=city_df['AQI'],
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=detect_df[detect_df['iso_anomaly']]['Date'],
        y=detect_df[detect_df['iso_anomaly']]['AQI'],
        mode='markers',
        marker=dict(color='red'),
        name='ISO'
    ))

    fig.add_trace(go.Scatter(
        x=detect_df[detect_df['z_anomaly']]['Date'],
        y=detect_df[detect_df['z_anomaly']]['AQI'],
        mode='markers',
        marker=dict(color='orange'),
        name='Z'
    ))

    fig.add_trace(go.Scatter(
        x=knn_df[knn_df['knn_anomaly']]['Date'],
        y=knn_df[knn_df['knn_anomaly']]['AQI'],
        mode='markers',
        marker=dict(color='green'),
        name='KNN'
    ))

    st.plotly_chart(fig, use_container_width=True)
