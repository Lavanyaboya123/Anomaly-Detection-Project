import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

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
st.title("🌫️ Advanced Air Quality Anomaly Detection")

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

# -------------------------------
# CLEAN DATA
# -------------------------------
df = df.sort_values(['City', 'Date'])
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df = df.dropna(subset=['AQI'])

# -------------------------------
# CITY SELECT
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", df['City'].unique())

base_df = df[df['City'] == selected_city].copy()
base_df = base_df.sort_values('Date')
base_df['AQI'] = base_df['AQI'].ffill().bfill()

# -------------------------------
# TREND DATA
# -------------------------------
trend_df = base_df.copy()
trend_df['rolling'] = trend_df['AQI'].rolling(30).mean()
trend_df = trend_df.dropna().reset_index(drop=True)

# -------------------------------
# DETECTION DATA
# -------------------------------
detect_df = base_df.copy()
detect_df['mean'] = detect_df['AQI'].rolling(30).mean()
detect_df['std'] = detect_df['AQI'].rolling(30).std()

detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']
detect_df['z'] = detect_df['z'].replace([np.inf, -np.inf], np.nan)
detect_df['z'] = detect_df['z'].fillna(0)

detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3
detect_df = detect_df.dropna().reset_index(drop=True)

# -------------------------------
# ISOLATION FOREST
# -------------------------------
model_df = detect_df.copy()

scaler = StandardScaler()
scaled = scaler.fit_transform(model_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
model_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# KNN
# -------------------------------
df3 = base_df.copy()
scaled_knn = scaler.fit_transform(df3[['AQI']])

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

    seq_len = 30
    X = create_sequences(data_scaled, seq_len)

    if len(X) > 50:
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_len,1)),
            RepeatVector(seq_len),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(1))
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, X, epochs=3, batch_size=32, verbose=0)

        X_pred = model.predict(X)
        mse = np.mean((X - X_pred)**2, axis=(1,2))

        threshold = np.percentile(mse, 95)
        anomalies = mse > threshold

        df_lstm = base_df.iloc[seq_len:].copy()
        df_lstm['lstm_anomaly'] = anomalies

# -------------------------------
# ALIGN DATA (IMPORTANT)
# -------------------------------
common_dates = model_df['Date']
df3 = df3[df3['Date'].isin(common_dates)]

if df_lstm is not None:
    df_lstm = df_lstm[df_lstm['Date'].isin(common_dates)]

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "🤖 KNN",
    "🧠 LSTM",
    "📊 Compare",
    "💡 Insights"
])

# -------------------------------
# TREND
# -------------------------------
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['AQI'], name="AQI"))
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['rolling'], name="30-day Avg"))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DETECTION
# -------------------------------
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_df['Date'], y=model_df['AQI'], name="AQI"))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['iso_anomaly']]['Date'],
        y=model_df[model_df['iso_anomaly']]['AQI'],
        mode='markers', name="Isolation Forest", marker=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['z_anomaly']]['Date'],
        y=model_df[model_df['z_anomaly']]['AQI'],
        mode='markers', name="Z-score", marker=dict(color='orange')
    ))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# KNN
# -------------------------------
with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df3['Date'], y=df3['AQI'], name="AQI"))
    fig.add_trace(go.Scatter(
        x=df3[df3['knn_anomaly']]['Date'],
        y=df3[df3['knn_anomaly']]['AQI'],
        mode='markers', name="KNN", marker=dict(color='green')
    ))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# LSTM
# -------------------------------
with tab4:
    if df_lstm is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_lstm['Date'], y=df_lstm['AQI'], name="AQI"))
        fig.add_trace(go.Scatter(
            x=df_lstm[df_lstm['lstm_anomaly']]['Date'],
            y=df_lstm[df_lstm['lstm_anomaly']]['AQI'],
            mode='markers', name="LSTM", marker=dict(color='purple')
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("LSTM not available")

# -------------------------------
# COMPARE
# -------------------------------
with tab5:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=base_df['Date'], y=base_df['AQI'], name="AQI"))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['iso_anomaly']]['Date'],
        y=model_df[model_df['iso_anomaly']]['AQI'],
        mode='markers', name="Isolation Forest", marker=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['z_anomaly']]['Date'],
        y=model_df[model_df['z_anomaly']]['AQI'],
        mode='markers', name="Z-score", marker=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df3[df3['knn_anomaly']]['Date'],
        y=df3[df3['knn_anomaly']]['AQI'],
        mode='markers', name="KNN", marker=dict(color='green')
    ))

    if df_lstm is not None:
        fig.add_trace(go.Scatter(
            x=df_lstm[df_lstm['lstm_anomaly']]['Date'],
            y=df_lstm[df_lstm['lstm_anomaly']]['AQI'],
            mode='markers', name="LSTM", marker=dict(color='purple')
        ))

    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("📊 Model Comparison")
    st.dataframe(pd.DataFrame({
        "Model": ["Z-score", "Isolation Forest", "KNN", "LSTM"],
        "Anomalies": [
            model_df['z_anomaly'].sum(),
            model_df['iso_anomaly'].sum(),
            df3['knn_anomaly'].sum(),
            df_lstm['lstm_anomaly'].sum() if df_lstm is not None else 0
        ]
    }))

    # Download
    csv = model_df.to_csv(index=False)
    st.download_button("Download Report", csv, file_name="report.csv")

# -------------------------------
# INSIGHTS
# -------------------------------
with tab6:
    avg = base_df['AQI'].tail(30).mean()

    if avg > 200:
        st.error("Severe pollution")
    elif avg > 100:
        st.warning("Moderate pollution")
    else:
        st.success("Good air quality")

    st.markdown("""
    ### 📘 Project Summary
    - Multi-model anomaly detection
    - Statistical + ML + Deep Learning
    - Real-world AQI dataset
    """)
