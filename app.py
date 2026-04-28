import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose

# Safe import (Streamlit Cloud fix)
try:
    from pyod.models.knn import KNN
    PYOD_AVAILABLE = True
except:
    PYOD_AVAILABLE = False

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Statistical + Machine Learning Dashboard")

# -------------------------------
# Upload Option
# -------------------------------
st.sidebar.header("📁 Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    else:
        df = pd.read_csv('city_day.csv', parse_dates=['Date'])
except:
    st.error("❌ Dataset not found. Please upload CSV.")
    st.stop()

# -------------------------------
# Clean Data (IMPORTANT FIX)
# -------------------------------
df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# Ensure AQI is numeric
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

# Fill missing safely (FIX for your error)
df['AQI'] = df['AQI'].ffill()
df['AQI'] = df['AQI'].bfill()

# Drop remaining NaN
df = df.dropna(subset=['AQI'])

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trend",
    "🔍 Anomaly Detection",
    "🤖 Advanced ML",
    "💡 Insights",
    "📊 Summary"
])

# -------------------------------
# TAB 1: Trend
# -------------------------------
with tab1:
    selected_city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == selected_city].copy()

    st.subheader(f"AQI Trend - {selected_city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    st.plotly_chart(fig, use_container_width=True)

    if len(city_df) > 365:
        st.subheader("Seasonal Decomposition")
        decomp = seasonal_decompose(
            city_df.set_index('Date')['AQI'],
            model='additive',
            period=365
        )
        st.pyplot(decomp.plot())

# -------------------------------
# TAB 2: Anomaly Detection
# -------------------------------
with tab2:
    st.subheader("🔍 Isolation Forest")

    city = st.selectbox("City", df['City'].unique(), key="city2")
    city_df = df[df['City'] == city].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
    city_df['anomaly'] = iso.fit_predict(scaled)
    city_df['anomaly'] = city_df['anomaly'] == -1

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['anomaly']]['Date'],
        y=city_df[city_df['anomaly']]['AQI'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Evaluation
    # -------------------------------
    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['anomaly'].astype(int)

    st.subheader("📊 Evaluation")

    st.write("Precision:", round(precision_score(y_true, y_pred), 2))
    st.write("Recall:", round(recall_score(y_true, y_pred), 2))
    st.write("F1 Score:", round(f1_score(y_true, y_pred), 2))

# -------------------------------
# TAB 3: Advanced ML
# -------------------------------
with tab3:
    st.subheader("🤖 KNN (PyOD)")

    if not PYOD_AVAILABLE:
        st.warning("⚠️ PyOD not installed. Skipping KNN.")
    else:
        city = st.selectbox("City", df['City'].unique(), key="city3")
        city_df = df[df['City'] == city].copy()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(city_df[['AQI']])

        knn = KNN(contamination=0.05)
        city_df['knn'] = knn.fit_predict(scaled) == 1

        st.write(f"Detected anomalies: {city_df['knn'].sum()}")

# -------------------------------
# TAB 4: Insights
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    - Delhi shows high winter pollution spikes  
    - Hyderabad has moderate anomalies  
    - Isolation Forest detects pattern anomalies  
    - KNN detects density anomalies  
    """)

# -------------------------------
# TAB 5: Summary
# -------------------------------
with tab5:
    st.subheader("📊 Project Summary")

    st.markdown("""
    ### 🔍 Project
    Detect anomalies in air quality time-series data

    ### 🧠 Methods
    - Statistical cleaning
    - Isolation Forest
    - KNN (PyOD)

    ### 🚀 Features
    - Interactive dashboard
    - Evaluation metrics
    - Upload your dataset

    ### 🌍 Use Cases
    - Smart cities
    - Pollution monitoring
    """)
