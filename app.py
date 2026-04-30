import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose

# Try PyOD (safe import for cloud)
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
# GLOBAL CITY SELECTION (FIXED)
# -------------------------------
selected_city = st.sidebar.selectbox("🏙️ Select City", df['City'].unique())

city_df = df[df['City'] == selected_city].copy().reset_index(drop=True)

# -------------------------------
# DATA CLEANING (FIXED ERRORS)
# -------------------------------
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()
city_df = city_df.dropna(subset=['AQI'])

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Anomaly Detection",
    "🤖 Advanced ML",
    "💡 Insights"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {selected_city}")

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
        marker=dict(color='red', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Evaluation
    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    st.subheader("📊 Evaluation")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

# -------------------------------
# TAB 3: ADVANCED ML
# -------------------------------
with tab3:
    st.subheader("🤖 Advanced ML")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    if PYOD_AVAILABLE:
        knn = KNN(contamination=0.05)
        city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1

        st.success(f"KNN detected {city_df['knn_anomaly'].sum()} anomalies")

    else:
        st.warning("PyOD not installed on cloud. KNN disabled.")

# -------------------------------
# TAB 4: INSIGHTS
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    avg_aqi = city_df['AQI'].mean()
    max_aqi = city_df['AQI'].max()

    st.write(f"Average AQI: {avg_aqi:.2f}")
    st.write(f"Max AQI: {max_aqi}")

    st.markdown("""
    ### Observations:
    - High spikes indicate pollution events
    - Isolation Forest detects unusual patterns
    - Seasonal trends affect AQI levels
    """)
