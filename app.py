import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.tsa.seasonal import seasonal_decompose

# Safe import
try:
    from pyod.models.knn import KNN
    pyod_available = True
except:
    pyod_available = False

st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv('city_day.csv', parse_dates=['Date'])

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# GLOBAL CITY (IMPORTANT FIX)
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", df['City'].unique())
city_df = df[df['City'] == selected_city].copy()

# Clean AQI
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# COMMON ANOMALY LOGIC (SHARED)
# -------------------------------
city_df['mean'] = city_df['AQI'].rolling(30, min_periods=10).mean()
city_df['std'] = city_df['AQI'].rolling(30, min_periods=10).std()

city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
city_df['z_anomaly'] = np.abs(city_df['z']) > 3

scaler = StandardScaler()
scaled = scaler.fit_transform(city_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# Severity + reason
def severity(aqi):
    if aqi > 300: return "High"
    elif aqi > 200: return "Medium"
    else: return "Low"

def reason(aqi, z):
    if aqi > 300: return "Severe pollution spike"
    elif aqi > 200: return "High pollution"
    elif z < -3: return "Sudden drop"
    else: return "Unusual variation"

city_df['severity'] = city_df['AQI'].apply(severity)
city_df['reason'] = city_df.apply(lambda r: reason(r['AQI'], r['z']), axis=1)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trend",
    "🔍 Anomalies",
    "🤖 ML",
    "💡 Insights",
    "📊 Summary"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {selected_city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name='AQI'))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2: ANOMALY
# -------------------------------
with tab2:
    st.subheader("🔍 Anomaly Detection")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=city_df['Date'], y=city_df['AQI'],
        name='AQI'
    ))

    anomalies = city_df[city_df['iso_anomaly']]

    fig.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['AQI'],
        mode='markers',
        marker=dict(color='red', size=8),
        text=anomalies['reason'],
        hovertemplate="Date:%{x}<br>AQI:%{y}<br>%{text}<extra></extra>"
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Explanation")
    for _, row in anomalies.head(5).iterrows():
        st.write(f"{row['Date'].date()} → {row['reason']} ({row['severity']})")

# -------------------------------
# TAB 3: ML
# -------------------------------
with tab3:
    st.subheader("🤖 ML Models")

    st.write(f"Isolation Forest Anomalies: {city_df['iso_anomaly'].sum()}")
    st.write(f"Z-score Anomalies: {city_df['z_anomaly'].sum()}")

    if pyod_available:
        knn = KNN(contamination=0.05)
        city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1
        st.write(f"KNN Anomalies: {city_df['knn_anomaly'].sum()}")
    else:
        st.warning("Install PyOD for KNN")

# -------------------------------
# TAB 4: INSIGHTS (FIXED)
# -------------------------------
with tab4:
    st.subheader("💡 Smart Insights")

    avg = city_df['AQI'].mean()
    max_val = city_df['AQI'].max()

    st.write(f"Average AQI: {avg:.2f}")
    st.write(f"Max AQI: {max_val}")
    st.write(f"Total anomalies: {city_df['iso_anomaly'].sum()}")

    if max_val > 300:
        st.error("Severe spikes detected 🚨")
    elif avg > 200:
        st.error("Unhealthy air quality")
    elif avg > 100:
        st.warning("Moderate pollution")
    else:
        st.success("Good air quality")

# -------------------------------
# TAB 5: SUMMARY
# -------------------------------
with tab5:
    st.subheader("📊 Summary")

    st.write("""
    - Time series AQI analysis
    - Anomaly detection using ML
    - Explainable anomalies
    - Real-time insights
    """)
