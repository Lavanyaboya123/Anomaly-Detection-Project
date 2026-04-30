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
st.title("🌫️ AQI Anomaly Detection (Fixed Version)")

# -------------------------------
# HELPER FUNCTION (IMPORTANT)
# -------------------------------
def show_plot(df, fig, name):
    if df is None or len(df) == 0:
        st.warning(f"⚠️ No data available for {name}")
    else:
        st.plotly_chart(fig, use_container_width=True)

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
# FILTER GOOD CITIES (STRONG FIX)
# -------------------------------
valid_cities = []

for city in df['City'].unique():
    temp = df[df['City'] == city]
    
    # keep only cities with enough good data
    if len(temp) > 200 and temp['AQI'].notna().sum() > 150:
        valid_cities.append(city)

if len(valid_cities) == 0:
    st.error("No cities have enough valid data")
    st.stop()

# -------------------------------
# SELECT CITY
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", valid_cities)

city_df = df[df['City'] == selected_city].copy()
city_df['AQI'] = city_df['AQI'].ffill().bfill()

if len(city_df) < 50:
    st.warning("Not enough data for this city")
    st.stop()

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
    trend_df['rolling'] = trend_df['AQI'].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['AQI'], name='AQI'))
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['rolling'], name='Avg'))

    show_plot(trend_df, fig, "Trend")

# ===============================
# 🔍 DETECTION (Z + ISO)
# ===============================
with tab2:
    detect_df = city_df.copy()

    detect_df['mean'] = detect_df['AQI'].rolling(30).mean()
    detect_df['std'] = detect_df['AQI'].rolling(30).std()

    detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']
    detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3

    # clean
    detect_df = detect_df.replace([np.inf, -np.inf], np.nan)
    detect_df = detect_df.dropna()

    if len(detect_df) > 0:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(detect_df[['AQI']])

        iso = IsolationForest(contamination=0.05, random_state=42)
        detect_df['iso_anomaly'] = iso.fit_predict(scaled) == -1
    else:
        detect_df['iso_anomaly'] = False

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=detect_df['Date'], y=detect_df['AQI'], name='AQI'))

    fig.add_trace(go.Scatter(
        x=detect_df[detect_df['iso_anomaly']]['Date'],
        y=detect_df[detect_df['iso_anomaly']]['AQI'],
        mode='markers', marker=dict(color='red'), name='ISO'
    ))

    fig.add_trace(go.Scatter(
        x=detect_df[detect_df['z_anomaly']]['Date'],
        y=detect_df[detect_df['z_anomaly']]['AQI'],
        mode='markers', marker=dict(color='orange'), name='Z'
    ))

    show_plot(detect_df, fig, "Detection")

# ===============================
# 🤖 KNN
# ===============================
with tab3:
    knn_df = city_df.copy()
    knn_df = knn_df.dropna(subset=['AQI'])

    if len(knn_df) > 0:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(knn_df[['AQI']])

        knn = KNN(contamination=0.05)
        knn_df['knn_anomaly'] = knn.fit_predict(scaled) == 1
    else:
        knn_df['knn_anomaly'] = False

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=knn_df['Date'], y=knn_df['AQI']))

    fig.add_trace(go.Scatter(
        x=knn_df[knn_df['knn_anomaly']]['Date'],
        y=knn_df[knn_df['knn_anomaly']]['AQI'],
        mode='markers', marker=dict(color='green'), name='KNN'
    ))

    show_plot(knn_df, fig, "KNN")

# ===============================
# 📊 COMPARE
# ===============================
with tab4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name='AQI'))

    show_plot(city_df, fig, "Compare")
