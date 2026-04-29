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
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Statistical + Machine Learning (Production Ready Version)")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# SAFE DATA CLEANING FUNCTION
# -------------------------------
def clean_aqi(data):
    data = pd.to_numeric(data, errors='coerce')
    data = data.ffill()
    data = data.bfill()
    return data

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "🤖 Advanced ML",
    "💡 Insights",
    "📊 Summary"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == city].copy()

    city_df['AQI'] = clean_aqi(city_df['AQI'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI',
        line=dict(color='blue')
    ))

    fig.update_layout(title=f"AQI Trend - {city}")
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
# TAB 2: ANOMALY DETECTION
# -------------------------------
with tab2:
    st.subheader("🔍 Isolation Forest Detection")

    city = st.selectbox("City", df['City'].unique(), key="iso_city")
    city_df = df[df['City'] == city].copy()

    city_df['AQI'] = clean_aqi(city_df['AQI'])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
    city_df['anomaly'] = iso.fit_predict(scaled) == -1

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
        name='Anomaly',
        marker=dict(color='red', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Evaluation
    # -----------------------
    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['anomaly'].astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.subheader("📊 Evaluation")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

# -------------------------------
# TAB 3: KNN (PYOD)
# -------------------------------
with tab3:
    st.subheader("🤖 KNN Anomaly Detection")

    city = st.selectbox("City", df['City'].unique(), key="knn_city")
    city_df = df[df['City'] == city].copy()

    city_df['AQI'] = clean_aqi(city_df['AQI'])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    # REMOVE ANY REMAINING NaN
    scaled = np.nan_to_num(scaled)

    knn = KNN(contamination=0.05)
    preds = knn.fit_predict(scaled)

    city_df['knn_anomaly'] = preds == 1

    st.write(f"Detected: {city_df['knn_anomaly'].sum()} anomalies")

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['knn_anomaly']]['Date'],
        y=city_df[city_df['knn_anomaly']]['AQI'],
        mode='markers',
        name='KNN Anomaly',
        marker=dict(color='orange', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 4: INSIGHTS
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    - Delhi shows strong seasonal spikes (winter pollution)
    - Hyderabad has fewer anomalies
    - Isolation Forest detects pattern anomalies
    - KNN detects density anomalies
    """)

# -------------------------------
# TAB 5: SUMMARY
# -------------------------------
with tab5:
    st.subheader("📊 Project Summary")

    st.markdown("""
    ### 🔍 What this project does:
    Detects anomalies in time-series AQI data

    ### 🧠 Techniques used:
    - Statistical cleaning
    - Isolation Forest
    - KNN (PyOD)

    ### 🚀 Features:
    - Interactive dashboard
    - Evaluation metrics
    - Upload your dataset

    ### 🌍 Use Cases:
    - Pollution monitoring
    - Smart city analytics
    - Risk detection
    """)
