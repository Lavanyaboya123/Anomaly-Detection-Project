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

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

df = df.sort_values(['City', 'Date'])

# FIX DATA
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

# -------------------------------
# GLOBAL CITY SELECT (IMPORTANT)
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", df['City'].unique())

city_df = df[df['City'] == selected_city].copy()
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "🤖 ML Models",
    "💡 Insights"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {selected_city}")

    city_df['rolling'] = city_df['AQI'].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name='AQI'))
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['rolling'], name='30-day Avg'))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2: ANOMALY DETECTION
# -------------------------------
with tab2:
    st.subheader("🔍 Anomaly Detection")

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

    # GRAPH
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name='AQI'))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['iso_anomaly']]['Date'],
        y=city_df[city_df['iso_anomaly']]['AQI'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # METRICS
    # -------------------------------
    threshold = city_df['AQI'].quantile(0.95)
    city_df['gt'] = city_df['AQI'] > threshold

    precision = precision_score(city_df['gt'], city_df['iso_anomaly'])
    recall = recall_score(city_df['gt'], city_df['iso_anomaly'])
    f1 = f1_score(city_df['gt'], city_df['iso_anomaly'])

    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

# -------------------------------
# TAB 3: ML MODELS
# -------------------------------
with tab3:
    st.subheader("🤖 KNN Anomaly Detection")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    knn = KNN(contamination=0.05)
    city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1

    st.write(f"KNN detected {city_df['knn_anomaly'].sum()} anomalies")

# -------------------------------
# TAB 4: SMART INSIGHTS
# -------------------------------
with tab4:
    st.subheader("💡 Smart Insights")

    anomalies = city_df[city_df['iso_anomaly']]

    for _, row in anomalies.head(10).iterrows():
        if row['AQI'] > 300:
            st.write(f"{row['Date'].date()} → 🚨 Severe pollution spike")
        elif row['AQI'] > 200:
            st.write(f"{row['Date'].date()} → ⚠️ High pollution")
        elif row['AQI'] < 50:
            st.write(f"{row['Date'].date()} → 🌿 Clean air drop")
        else:
            st.write(f"{row['Date'].date()} → 🔍 Unusual variation")
