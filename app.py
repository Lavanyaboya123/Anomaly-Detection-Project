import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose

# Optional (safe import)
try:
    from pyod.models.knn import KNN
    KNN_AVAILABLE = True
except:
    KNN_AVAILABLE = False

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")

st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Statistical + Machine Learning Dashboard")

# -------------------------------
# Upload Data
# -------------------------------
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv('city_day.csv', parse_dates=['Date'])

# -------------------------------
# Data Validation
# -------------------------------
required_cols = ['City', 'Date', 'AQI']

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain columns: City, Date, AQI")
    st.stop()

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# Fix AQI values (IMPORTANT)
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df['AQI'] = df['AQI'].ffill().bfill()
df = df.dropna(subset=['AQI'])

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
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

    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(city_df))
    col2.metric("Avg AQI", int(city_df['AQI'].mean()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    fig.update_layout(
        title=f"AQI Trend - {city}",
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_dark"
    )

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
# TAB 2: Isolation Forest
# -------------------------------
with tab2:
    st.subheader("🔍 Isolation Forest Detection")

    city = st.selectbox("City", df['City'].unique(), key="city2")
    city_df = df[df['City'] == city].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
    city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

    st.metric("Detected Anomalies", city_df['iso_anomaly'].sum())

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
        marker=dict(color='red', size=8)
    ))

    fig.update_layout(
        title="AQI with Anomalies",
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Evaluation
    threshold = city_df['AQI'].quantile(0.95)
    city_df['gt'] = city_df['AQI'] > threshold

    y_true = city_df['gt'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    st.subheader("📊 Model Evaluation")
    st.write("Precision:", round(precision_score(y_true, y_pred), 2))
    st.write("Recall:", round(recall_score(y_true, y_pred), 2))
    st.write("F1 Score:", round(f1_score(y_true, y_pred), 2))

    # Explanation
    st.subheader("🧠 Why Anomalies?")
    for _, row in city_df[city_df['iso_anomaly']].head(5).iterrows():
        if row['AQI'] > 300:
            st.write(f"{row['Date'].date()} → Severe pollution spike")
        elif row['AQI'] > 200:
            st.write(f"{row['Date'].date()} → High pollution level")
        else:
            st.write(f"{row['Date'].date()} → Sudden variation")

# -------------------------------
# TAB 3: KNN (Advanced ML)
# -------------------------------
with tab3:
    st.subheader("🤖 KNN Detection")

    if not KNN_AVAILABLE:
        st.warning("KNN not available (PyOD not installed)")
    else:
        city = st.selectbox("City", df['City'].unique(), key="city3")
        city_df = df[df['City'] == city].copy()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(city_df[['AQI']])

        knn = KNN(contamination=0.05)
        preds = knn.fit_predict(scaled)

        city_df['knn_anomaly'] = preds == 1

        st.metric("KNN Anomalies", city_df['knn_anomaly'].sum())

        # Comparison
        st.subheader("📊 Model Comparison")

        comparison = pd.DataFrame({
            "Model": ["Isolation Forest", "KNN"],
            "Anomalies": [
                city_df['iso_anomaly'].sum() if 'iso_anomaly' in city_df else 0,
                city_df['knn_anomaly'].sum()
            ]
        })

        st.dataframe(comparison)

# -------------------------------
# TAB 4: Insights
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    - Delhi shows higher pollution spikes in winter
    - Hyderabad shows more stable AQI trends
    - Isolation Forest detects pattern anomalies
    - KNN detects density-based anomalies
    """)

    st.subheader("🌍 Real World Impact")
    st.markdown("""
    - Early pollution spike detection
    - Smart city monitoring
    - Environmental risk alerts
    - Government policy insights
    """)

# -------------------------------
# TAB 5: Summary
# -------------------------------
with tab5:
    st.subheader("📊 Project Summary")

    st.markdown("""
    ### 🔍 Objective
    Detect anomalies in air quality data

    ### 🧠 Techniques
    - Isolation Forest
    - KNN (PyOD)

    ### 🚀 Features
    - Interactive dashboard
    - Model evaluation
    - Upload custom dataset

    ### 🌍 Use Cases
    - Air pollution monitoring
    - Smart cities
    - Environmental analytics
    """)
