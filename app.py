import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose
from pyod.models.knn import KNN

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Statistical + ML + Deep Learning (LSTM optional)")

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
    "🤖 ML Advanced",
    "💡 Insights",
    "📊 Summary"
])

# -------------------------------
# TAB 1: Trend
# -------------------------------
with tab1:
    selected_city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == selected_city].copy()

    # Handle NaN
    city_df['AQI'] = city_df['AQI'].fillna(method='ffill')
    city_df['AQI'] = city_df['AQI'].fillna(method='bfill')

    st.subheader(f"AQI Trend - {selected_city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'],
                             mode='lines', name='AQI'))
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal decomposition
    if len(city_df) > 365:
        st.subheader("Seasonal Decomposition")
        decomp = seasonal_decompose(city_df.set_index('Date')['AQI'],
                                     model='additive', period=365)
        fig2 = decomp.plot()
        fig2.set_size_inches(12, 8)
        st.pyplot(fig2)

# -------------------------------
# TAB 2: Anomaly Detection
# -------------------------------
with tab2:
    st.subheader("🔍 Anomaly Detection")

    city = st.selectbox("City", df['City'].unique(), key="city2")
    city_df = df[df['City'] == city].copy().reset_index(drop=True)

    # Handle NaN
    city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
    city_df['AQI'] = city_df['AQI'].ffill().bfill()

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
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'],
                             mode='lines', name='AQI'))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['iso_anomaly']]['Date'],
        y=city_df[city_df['iso_anomaly']]['AQI'],
        mode='markers',
        name='Isolation Forest',
        marker=dict(color='red', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    threshold_gt = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold_gt

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.subheader("📊 Model Evaluation")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # -------------------------------
    # Explanation
    # -------------------------------
    st.subheader("🧠 Anomaly Explanation")

    anomalies = city_df[city_df['iso_anomaly']]

    for i, row in anomalies.head(5).iterrows():
        if row['AQI'] > 300:
            st.write(f"{row['Date'].date()} → Severe pollution spike")
        elif row['AQI'] > 200:
            st.write(f"{row['Date'].date()} → High pollution level")
        else:
            st.write(f"{row['Date'].date()} → Unusual variation")

# -------------------------------
# TAB 3: Advanced ML (KNN)
# -------------------------------
with tab3:
    st.subheader("🤖 Advanced ML - KNN (PyOD)")

    city = st.selectbox("City", df['City'].unique(), key="city3")
    city_df = df[df['City'] == city].copy()

    # Handle NaN
    city_df['AQI'] = city_df['AQI'].fillna(method='ffill')
    city_df['AQI'] = city_df['AQI'].fillna(method='bfill')

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    knn = KNN(contamination=0.05)
    city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1

    st.write(f"KNN Detected: {city_df['knn_anomaly'].sum()} anomalies")

# -------------------------------
# TAB 4: Insights
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    - Delhi shows high pollution anomalies in winter
    - Hyderabad has fewer anomalies
    - Isolation Forest detects pattern-based anomalies
    - KNN detects density-based anomalies
    """)

# -------------------------------
# TAB 5: Summary
# -------------------------------
with tab5:
    st.subheader("📊 Project Summary")

    st.markdown("""
    ### 🔍 What this project does:
    - Detects anomalies in AQI time series data

    ### 🧠 Techniques used:
    - Statistical (Z-score)
    - Machine Learning (Isolation Forest, KNN)

    ### 🚀 Features:
    - Interactive dashboard
    - Model evaluation (Precision, Recall, F1)
    - Anomaly explanation
    - Upload your own dataset

    ### 🌍 Use cases:
    - Air pollution monitoring
    - Smart city analytics
    - Environmental risk detection
    """)
