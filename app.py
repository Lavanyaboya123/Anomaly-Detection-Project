import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.knn import KNN

# -------------------------------
# TensorFlow Safe Import
# -------------------------------
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ AQI Anomaly Detection (Final Stable Version)")

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

# -------------------------------
# CLEAN BASE DATA
# -------------------------------
df = df.sort_values(['City', 'Date'])
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df = df.dropna(subset=['AQI'])

# -------------------------------
# FILTER VALID CITIES (CRITICAL FIX)
# -------------------------------
valid_cities = []
for city in df['City'].unique():
    temp = df[df['City'] == city]
    if temp['AQI'].dropna().shape[0] > 100:
        valid_cities.append(city)

if len(valid_cities) == 0:
    st.error("No valid cities with enough data")
    st.stop()

# -------------------------------
# CITY SELECT
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", valid_cities)

base_df = df[df['City'] == selected_city].copy()
base_df = base_df.sort_values('Date')
base_df['AQI'] = base_df['AQI'].ffill().bfill()

if len(base_df) < 100:
    st.warning("Not enough data for analysis")
    st.stop()

# -------------------------------
# TREND DATA
# -------------------------------
trend_df = base_df.copy()
trend_df['rolling'] = trend_df['AQI'].rolling(30).mean()
trend_df = trend_df.dropna()

# -------------------------------
# DETECTION DATA
# -------------------------------
detect_df = base_df.copy()

detect_df['mean'] = detect_df['AQI'].rolling(30).mean()
detect_df['std'] = detect_df['AQI'].rolling(30).std()

detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']

detect_df = detect_df.replace([np.inf, -np.inf], np.nan)
detect_df = detect_df.dropna(subset=['AQI', 'mean', 'std', 'z'])

detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3

# -------------------------------
# ISOLATION FOREST
# -------------------------------
model_df = detect_df.copy()

safe_df = model_df[['AQI']].replace([np.inf, -np.inf], np.nan).dropna()

if len(safe_df) == 0:
    st.error("No valid data for model")
    st.stop()

scaler = StandardScaler()
scaled = scaler.fit_transform(safe_df)

model_df = model_df.loc[safe_df.index].reset_index(drop=True)

iso = IsolationForest(contamination=0.05, random_state=42)
model_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# KNN
# -------------------------------
df3 = base_df.copy()
safe_knn = df3[['AQI']].replace([np.inf, -np.inf], np.nan).dropna()

scaled_knn = scaler.fit_transform(safe_knn)
df3 = df3.loc[safe_knn.index]

knn = KNN(contamination=0.05)
df3['knn_anomaly'] = knn.fit_predict(scaled_knn) == 1

# -------------------------------
# LSTM
# -------------------------------
df_lstm = None

if TF_AVAILABLE:
    data = base_df[['AQI']].values
    scaler_lstm = MinMaxScaler()
    data_scaled = scaler_lstm.fit_transform(data)

    def create_sequences(data, seq_len=30):
        return np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)])

    X = create_sequences(data_scaled, 30)

    if len(X) > 50:
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(30,1)),
            RepeatVector(30),
            LSTM(32, activation='relu', return_sequences=True),
            TimeDistributed(Dense(1))
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, X, epochs=2, verbose=0)

        X_pred = model.predict(X)
        mse = np.mean((X - X_pred)**2, axis=(1,2))

        threshold = np.percentile(mse, 95)

        df_lstm = base_df.iloc[30:].copy()
        df_lstm['lstm_anomaly'] = mse > threshold

# -------------------------------
# ALIGN DATES
# -------------------------------
common_dates = model_df['Date']
df3 = df3[df3['Date'].isin(common_dates)]

if df_lstm is not None:
    df_lstm = df_lstm[df_lstm['Date'].isin(common_dates)]

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trend", "🔍 Detection", "🤖 KNN", "🧠 LSTM", "📊 Compare"
])

# -------------------------------
# TREND
# -------------------------------
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['AQI'], name='AQI'))
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['rolling'], name='Avg'))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DETECTION
# -------------------------------
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_df['Date'], y=model_df['AQI']))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['iso_anomaly']]['Date'],
        y=model_df[model_df['iso_anomaly']]['AQI'],
        mode='markers', marker=dict(color='red'), name='ISO'
    ))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['z_anomaly']]['Date'],
        y=model_df[model_df['z_anomaly']]['AQI'],
        mode='markers', marker=dict(color='orange'), name='Z'
    ))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# KNN
# -------------------------------
with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df3['Date'], y=df3['AQI']))
    fig.add_trace(go.Scatter(
        x=df3[df3['knn_anomaly']]['Date'],
        y=df3[df3['knn_anomaly']]['AQI'],
        mode='markers', marker=dict(color='green')
    ))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# LSTM
# -------------------------------
with tab4:
    if df_lstm is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_lstm['Date'], y=df_lstm['AQI']))
        fig.add_trace(go.Scatter(
            x=df_lstm[df_lstm['lstm_anomaly']]['Date'],
            y=df_lstm[df_lstm['lstm_anomaly']]['AQI'],
            mode='markers', marker=dict(color='purple')
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("LSTM not available")

# -------------------------------
# COMPARE
# -------------------------------
with tab5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_df['Date'], y=base_df['AQI']))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['iso_anomaly']]['Date'],
        y=model_df[model_df['iso_anomaly']]['AQI'],
        mode='markers', marker=dict(color='red'), name='ISO'
    ))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['z_anomaly']]['Date'],
        y=model_df[model_df['z_anomaly']]['AQI'],
        mode='markers', marker=dict(color='orange'), name='Z'
    ))

    fig.add_trace(go.Scatter(
        x=df3[df3['knn_anomaly']]['Date'],
        y=df3[df3['knn_anomaly']]['AQI'],
        mode='markers', marker=dict(color='green'), name='KNN'
    ))

    if df_lstm is not None:
        fig.add_trace(go.Scatter(
            x=df_lstm[df_lstm['lstm_anomaly']]['Date'],
            y=df_lstm[df_lstm['lstm_anomaly']]['AQI'],
            mode='markers', marker=dict(color='purple'), name='LSTM'
        ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame({
        "Model": ["Z-score", "Isolation Forest", "KNN", "LSTM"],
        "Anomalies": [
            model_df['z_anomaly'].sum(),
            model_df['iso_anomaly'].sum(),
            df3['knn_anomaly'].sum(),
            df_lstm['lstm_anomaly'].sum() if df_lstm is not None else 0
        ]
    }))

    st.download_button("Download Report", model_df.to_csv(index=False), "report.csv")
