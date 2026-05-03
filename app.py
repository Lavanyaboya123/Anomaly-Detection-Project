import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detection", layout="wide")
st.title("🌫️ AQI Anomaly Detection System")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file, parse_dates=['Date'])
    return pd.read_csv("city_day.csv", parse_dates=['Date'])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded_file)

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# CITY SELECTOR (TOP)
# -------------------------------
st.markdown("## 📍 Select City")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("Choose City", sorted(df['City'].unique()))

city_df = df[df['City'] == city].copy()

with col2:
    st.metric("Records", len(city_df))

with col3:
    st.metric("Years", city_df['Date'].dt.year.nunique())

# -------------------------------
# CLEAN
# -------------------------------
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# FEATURES
# -------------------------------
city_df['mean'] = city_df['AQI'].rolling(30, min_periods=10).mean()
city_df['std'] = city_df['AQI'].rolling(30, min_periods=10).std()

city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
city_df['z_anomaly'] = np.abs(city_df['z']) > 3

# -------------------------------
# MODELS
# -------------------------------
scaler = StandardScaler()
scaled = scaler.fit_transform(city_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

knn = KNN(contamination=0.05)
lof = LOF(contamination=0.05)

city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1
city_df['lof_anomaly'] = lof.fit_predict(scaled) == 1

# -------------------------------
# EXPLANATION
# -------------------------------
def get_reason(row):
    if pd.isna(row['mean']) or pd.isna(row['std']):
        return "Insufficient data"
    elif row['AQI'] > row['mean'] + 2 * row['std']:
        return "High pollution spike"
    elif row['AQI'] < row['mean'] - 2 * row['std']:
        return "Sudden drop"
    else:
        return "Normal variation"

def get_severity(row):
    if pd.isna(row['z']):
        return "Low"
    elif abs(row['z']) > 4:
        return "High"
    elif abs(row['z']) > 3:
        return "Medium"
    else:
        return "Low"

city_df['reason'] = city_df.apply(get_reason, axis=1)
city_df['severity'] = city_df.apply(get_severity, axis=1)

# -------------------------------
# AQI CATEGORY
# -------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Anomalies",
    "🤖 Models",
    "💡 Insights"
])

# -------------------------------
# TREND
# -------------------------------
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name='AQI'))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# ANOMALIES
# -------------------------------
with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name='AQI'))

    def add_points(col, color, name):
        subset = city_df[city_df[col]]
        fig2.add_trace(go.Scatter(
            x=subset['Date'],
            y=subset['AQI'],
            mode='markers',
            marker=dict(color=color, size=8),
            name=name,
            text=subset['reason'],
            customdata=subset['severity'],
            hovertemplate=
            "<b>Date:</b> %{x}<br>" +
            "<b>AQI:</b> %{y}<br>" +
            "<b>Reason:</b> %{text}<br>" +
            "<b>Severity:</b> %{customdata}<extra></extra>"
        ))

    add_points('iso_anomaly', 'red', 'Isolation Forest')
    add_points('z_anomaly', 'orange', 'Z-score')

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# MODELS
# -------------------------------
with tab3:
    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    models = ['z_anomaly','iso_anomaly','knn_anomaly','lof_anomaly']
    scores = {}

    for m in models:
        y_true = city_df['ground_truth'].astype(int)
        y_pred = city_df[m].astype(int)
        scores[m] = f1_score(y_true, y_pred, zero_division=0)
        st.write(f"{m} → F1 Score: {scores[m]:.2f}")

    st.success(f"Best Model: {max(scores, key=scores.get)}")

# -------------------------------
# 💡 INSIGHTS (STRONG ALERT FIX)
# -------------------------------
with tab4:
    latest = city_df['AQI'].iloc[-1]
    avg = city_df['AQI'].mean()
    max_val = city_df['AQI'].max()

    anomaly_count = city_df['iso_anomaly'].sum()
    danger_days = len(city_df[city_df['AQI'] > 150])

    st.metric("Latest AQI", int(latest))
    st.metric("Worst AQI", int(max_val))
    st.metric("Danger Days", danger_days)

    st.write(f"Current: **{aqi_category(latest)}**")
    st.write(f"Worst: **{aqi_category(max_val)}**")

    # 🔴 FINAL ALERT LOGIC
    if (
        latest > 200 or
        max_val > 300 or
        anomaly_count > 20 or
        danger_days > 30
    ):
        st.error("🚨 RED ALERT: Pollution risk detected!")

    elif latest > 150 or max_val > 200:
        st.warning("⚠️ Pollution is concerning")

    else:
        st.success("✅ Air quality is relatively safe")

    st.markdown("### Recent Anomalies")
    recent = city_df[city_df['iso_anomaly']].tail(5)

    for _, row in recent.iterrows():
        st.write(
            f"{row['Date'].date()} → AQI {row['AQI']} "
            f"({row['severity']}) - {row['reason']}"
        )
