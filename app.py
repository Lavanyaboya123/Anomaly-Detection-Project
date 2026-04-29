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
st.markdown("Statistical + Machine Learning + Deep Learning Concepts")

# -------------------------------
# Upload Option
# -------------------------------
st.sidebar.header("📁 Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv('city_day.csv', parse_dates=['Date'])

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trend & EDA",
    "🔍 Anomaly Detection",
    "🤖 Advanced ML",
    "💡 Insights",
    "📊 Summary"
])

# -------------------------------
# Common function to clean AQI
# -------------------------------
def clean_aqi(data):
    data['AQI'] = pd.to_numeric(data['AQI'], errors='coerce')
    data['AQI'] = data['AQI'].ffill().bfill()
    data = data.dropna(subset=['AQI'])
    return data

# -------------------------------
# TAB 1: Trend
# -------------------------------
with tab1:
    selected_city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == selected_city].copy()

    city_df = clean_aqi(city_df)

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
        fig2 = decomp.plot()
        fig2.set_size_inches(12, 8)
        st.pyplot(fig2)

# -------------------------------
# TAB 2: Anomaly Detection
# -------------------------------
with tab2:
    st.subheader("🔍 Anomaly Detection")

    city = st.selectbox("City", df['City'].unique(), key="city2")
    city_df = df[df['City'] == city].copy()

    city_df = clean_aqi(city_df)

    # Z-score
    city_df['mean'] = city_df['AQI'].rolling(30).mean()
    city_df['std'] = city_df['AQI'].rolling(30).std()
    city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
    city_df['z_anomaly'] = np.abs(city_df['z']) > 3

    # Isolation Forest
    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
    city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['iso_anomaly']]['Date'],
        y=city_df[city_df['iso_anomaly']]['AQI'],
        mode='markers',
        name='Anomaly',
        marker=dict(size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Evaluation
    threshold_gt = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold_gt

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    st.subheader("📊 Model Evaluation")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

# -------------------------------
# TAB 3: Advanced ML (KNN)
# -------------------------------
with tab3:
    st.subheader("🤖 Advanced ML - KNN")

    city = st.selectbox("City", df['City'].unique(), key="city3")
    city_df = df[df['City'] == city].copy()

    city_df = clean_aqi(city_df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    if len(city_df) > 10:
        knn = KNN(contamination=0.05)
        city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1

        st.write(f"Detected: {city_df['knn_anomaly'].sum()} anomalies")

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
            name='KNN Anomaly'
        ))

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough data for KNN")

# -------------------------------
# TAB 4: Insights
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    - Delhi shows strong seasonal pollution spikes
    - Winter months have highest anomalies
    - Isolation Forest detects global anomalies
    - KNN detects local density anomalies
    """)

# -------------------------------
# TAB 5: Summary
# -------------------------------
with tab5:
    st.subheader("📊 Project Summary")

    st.markdown("""
    ### 🔍 What this project does:
    Detects anomalies in time-series AQI data

    ### 🧠 Techniques used:
    - Z-score
    - Isolation Forest
    - KNN (PyOD)

    ### 🚀 Features:
    - Interactive dashboard
    - Evaluation metrics
    - Real-world dataset
    """)

st.caption("🚀 Final Year Project | Advanced Anomaly Detection")
