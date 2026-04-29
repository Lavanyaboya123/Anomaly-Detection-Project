import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Smart Analyzer", layout="wide")
st.title("🌫️ AQI Smart Anomaly Analyzer")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("city_day.csv", parse_dates=["Date"])

df = load_data()
df = df.sort_values(["City", "Date"])

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Controls")

city = st.sidebar.selectbox("Select City", df["City"].unique())

city_df = df[df["City"] == city].copy()

# Clean data
city_df["AQI"] = pd.to_numeric(city_df["AQI"], errors="coerce")
city_df["AQI"] = city_df["AQI"].ffill().bfill()

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
# Z-score
city_df["mean"] = city_df["AQI"].rolling(30).mean()
city_df["std"] = city_df["AQI"].rolling(30).std()
city_df["z"] = (city_df["AQI"] - city_df["mean"]) / city_df["std"]
city_df["z_anomaly"] = np.abs(city_df["z"]) > 3

# Isolation Forest
scaler = StandardScaler()
scaled = scaler.fit_transform(city_df[["AQI"]])

iso = IsolationForest(contamination=0.05, random_state=42)
city_df["iso_anomaly"] = iso.fit_predict(scaled) == -1

# KNN
knn = KNN(contamination=0.05)
city_df["knn_anomaly"] = knn.fit_predict(scaled) == 1

# -------------------------------
# SELECT POINT
# -------------------------------
selected_date = st.sidebar.selectbox(
    "Select Date (Focus Analysis)",
    city_df["Date"]
)

selected_row = city_df[city_df["Date"] == selected_date].iloc[0]

# -------------------------------
# KPI
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("AQI Value", int(selected_row["AQI"]))
col2.metric("Is Anomaly (ISO)", "Yes" if selected_row["iso_anomaly"] else "No")
col3.metric("Z-Score", round(selected_row["z"], 2) if not np.isnan(selected_row["z"]) else "NA")

# -------------------------------
# MAIN GRAPH
# -------------------------------
st.subheader(f"AQI Trend - {city}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=city_df["Date"],
    y=city_df["AQI"],
    mode="lines",
    name="AQI"
))

# Isolation anomalies
fig.add_trace(go.Scatter(
    x=city_df[city_df["iso_anomaly"]]["Date"],
    y=city_df[city_df["iso_anomaly"]]["AQI"],
    mode="markers",
    marker=dict(color="red", size=8),
    name="Isolation Forest"
))

# Selected point highlight
fig.add_trace(go.Scatter(
    x=[selected_row["Date"]],
    y=[selected_row["AQI"]],
    mode="markers",
    marker=dict(color="yellow", size=12),
    name="Selected Point"
))

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# COMPARISON GRAPH
# -------------------------------
st.subheader("Model Comparison")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=city_df["Date"],
    y=city_df["AQI"],
    mode="lines",
    name="AQI"
))

fig2.add_trace(go.Scatter(
    x=city_df[city_df["knn_anomaly"]]["Date"],
    y=city_df[city_df["knn_anomaly"]]["AQI"],
    mode="markers",
    marker=dict(color="blue", size=7),
    name="KNN"
))

fig2.add_trace(go.Scatter(
    x=city_df[city_df["z_anomaly"]]["Date"],
    y=city_df[city_df["z_anomaly"]]["AQI"],
    mode="markers",
    marker=dict(color="orange", size=7),
    name="Z-Score"
))

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# EXPLANATION ENGINE
# -------------------------------
st.subheader("📖 What happened on this date?")

aqi = selected_row["AQI"]

if aqi <= 50:
    explanation = "Air quality is GOOD. No health risk."
elif aqi <= 100:
    explanation = "Satisfactory air quality. Minor issues for sensitive people."
elif aqi <= 200:
    explanation = "Moderate pollution. Breathing discomfort possible."
elif aqi <= 300:
    explanation = "Poor air quality. Health effects likely."
else:
    explanation = "Severe pollution spike 🚨 Likely due to traffic, industry, or seasonal factors."

st.info(f"Date: {selected_row['Date'].date()}")
st.write(f"AQI: {int(aqi)}")
st.write(f"Explanation: {explanation}")

# -------------------------------
# ANOMALY SUMMARY
# -------------------------------
st.subheader("📊 Summary")

col1, col2, col3 = st.columns(3)

col1.metric("ISO Anomalies", city_df["iso_anomaly"].sum())
col2.metric("KNN Anomalies", city_df["knn_anomaly"].sum())
col3.metric("Z-score Anomalies", city_df["z_anomaly"].sum())
