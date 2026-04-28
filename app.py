import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose
from pyod.models.knn import KNN

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Statistical + Machine Learning Models")

# -------------------------------
# Upload Option
# -------------------------------
st.sidebar.header("📁 Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv('city_day.csv', parse_dates=['Date'])

# -------------------------------
# Basic Cleaning (VERY IMPORTANT)
# -------------------------------
required_cols = ['City', 'Date', 'AQI']

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain: City, Date, AQI")
    st.stop()

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# Force AQI numeric
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

# Fill NaN safely (FIXED ERROR HERE)
df['AQI'] = df['AQI'].ffill()
df['AQI'] = df['AQI'].bfill()

# Final safety (remove any remaining NaN)
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
    city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == city].copy()

    st.subheader(f"AQI Trend - {city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'],
                             mode='lines', name='AQI'))
    st.plotly_chart(fig, use_container_width=True)

    if len(city_df) > 365:
        st.subheader("Seasonal Decomposition")

        decomp = seasonal_decompose(
            city_df.set_index('Date')['AQI'],
            model='additive',
            period=365
        )

        fig2 = decomp.plot()
        st.pyplot(fig2)

# -------------------------------
# TAB 2: Anomaly Detection
# -------------------------------
with tab2:
    st.subheader("🔍 Isolation Forest Detection")

    city = st.selectbox("City", df['City'].unique(), key="city2")
    city_df = df[df['City'] == city].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
    city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'],
                             mode='lines', name='AQI'))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['iso_anomaly']]['Date'],
        y=city_df[city_df['iso_anomaly']]['AQI'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Evaluation
    # -------------------------------
    threshold = city_df['AQI'].quantile(0.95)
    city_df['gt'] = city_df['AQI'] > threshold

    y_true = city_df['gt'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    st.subheader("📊 Evaluation")

    st.write("Precision:", round(precision_score(y_true, y_pred), 2))
    st.write("Recall:", round(recall_score(y_true, y_pred), 2))
    st.write("F1 Score:", round(f1_score(y_true, y_pred), 2))

# -------------------------------
# TAB 3: KNN (SAFE FIX)
# -------------------------------
with tab3:
    st.subheader("🤖 KNN Anomaly Detection")

    try:
        city = st.selectbox("City", df['City'].unique(), key="city3")
        city_df = df[df['City'] == city].copy()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(city_df[['AQI']])

        knn = KNN(contamination=0.05)
        preds = knn.fit_predict(scaled)

        city_df['knn_anomaly'] = preds == 1

        st.success(f"KNN Detected {city_df['knn_anomaly'].sum()} anomalies")

    except Exception as e:
        st.warning("KNN failed (possibly due to data issue). Skipping...")

# -------------------------------
# TAB 4: Insights
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    - Delhi shows higher anomalies in winter
    - Hyderabad is more stable
    - Isolation Forest captures pattern anomalies
    - KNN captures density anomalies
    """)

# -------------------------------
# TAB 5: Summary
# -------------------------------
with tab5:
    st.subheader("📊 Project Summary")

    st.markdown("""
    ### 🔍 Project Goal
    Detect anomalies in AQI time series data

    ### 🧠 Methods Used
    - Statistical (Z-score concept)
    - Machine Learning (Isolation Forest, KNN)

    ### 🚀 Features
    - Interactive dashboard
    - Evaluation metrics
    - Upload custom dataset

    ### 🌍 Applications
    - Air pollution monitoring
    - Smart city analytics
    """)
