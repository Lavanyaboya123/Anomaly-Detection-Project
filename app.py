import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Advanced App", layout="wide")
st.title("🌫️ Advanced AQI Anomaly Detection")

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
# SINGLE PIPELINE (IMPORTANT)
# -------------------------------
city = st.sidebar.selectbox("Select City", df['City'].unique())

city_df = df[df['City'] == city].copy()

# CLEAN
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
# ML MODELS
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
# GRAPH 1: TREND
# -------------------------------
st.subheader("📈 AQI Trend")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name='AQI'))

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# GRAPH 2: ANOMALIES
# -------------------------------
st.subheader("🔍 Model Comparison")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=city_df['Date'],
    y=city_df['AQI'],
    name='AQI'
))

def add_points(col, color, name):
    fig2.add_trace(go.Scatter(
        x=city_df[city_df[col]]['Date'],
        y=city_df[city_df[col]]['AQI'],
        mode='markers',
        marker=dict(color=color, size=7),
        name=name
    ))

add_points('z_anomaly', 'orange', 'Z-score')
add_points('iso_anomaly', 'red', 'Isolation Forest')
add_points('knn_anomaly', 'green', 'KNN')
add_points('lof_anomaly', 'purple', 'LOF')

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# EVALUATION
# -------------------------------
st.subheader("📊 Model Evaluation")

threshold_gt = city_df['AQI'].quantile(0.95)
city_df['ground_truth'] = city_df['AQI'] > threshold_gt

models = ['z_anomaly','iso_anomaly','knn_anomaly','lof_anomaly']

for m in models:
    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df[m].astype(int)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    st.write(f"{m} → Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("💡 Insights")

avg = city_df['AQI'].mean()
max_val = city_df['AQI'].max()

st.write(f"City: {city}")
st.write(f"Average AQI: {avg:.2f}")
st.write(f"Max AQI: {max_val}")
st.write(f"Total anomalies: {city_df['iso_anomaly'].sum()}")

if max_val > 300:
    st.error("Severe pollution spikes 🚨")
elif avg > 200:
    st.error("Unhealthy air quality")
elif avg > 100:
    st.warning("Moderate pollution")
else:
    st.success("Good air quality")
