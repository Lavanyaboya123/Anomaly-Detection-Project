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
st.markdown("Statistical + ML + Advanced ML (KNN)")

# -------------------------------
# Upload Option
# -------------------------------
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv('city_day.csv', parse_dates=['Date'])

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "🤖 Advanced",
    "📊 Summary"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == city].copy()

    # FIXED NaN handling
    city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
    city_df['AQI'] = city_df['AQI'].ffill().bfill()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Decomposition
    if len(city_df) > 365:
        decomp = seasonal_decompose(
            city_df.set_index('Date')['AQI'],
            model='additive',
            period=365
        )
        st.pyplot(decomp.plot())

# -------------------------------
# TAB 2: DETECTION
# -------------------------------
with tab2:
    city = st.selectbox("City", df['City'].unique(), key="tab2")
    city_df = df[df['City'] == city].copy()

    # FIXED NaN
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
        marker=dict(color='red', size=8),
        name='Anomaly'
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    threshold = city_df['AQI'].quantile(0.95)
    city_df['gt'] = city_df['AQI'] > threshold

    precision = precision_score(city_df['gt'], city_df['iso_anomaly'])
    recall = recall_score(city_df['gt'], city_df['iso_anomaly'])
    f1 = f1_score(city_df['gt'], city_df['iso_anomaly'])

    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

# -------------------------------
# TAB 3: KNN
# -------------------------------
with tab3:
    city = st.selectbox("City", df['City'].unique(), key="tab3")
    city_df = df[df['City'] == city].copy()

    # FIXED NaN
    city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
    city_df['AQI'] = city_df['AQI'].ffill().bfill()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    knn = KNN(contamination=0.05)
    city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1

    st.write(f"KNN detected: {city_df['knn_anomaly'].sum()} anomalies")

# -------------------------------
# TAB 4: SUMMARY
# -------------------------------
with tab4:
    st.markdown("""
    ### 📌 Summary
    - Detects AQI anomalies
    - Uses Statistical + ML + KNN
    - Handles missing values properly
    - Interactive dashboard
    """)
