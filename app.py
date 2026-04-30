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

# -------------------------------
# CLEAN DATA
# -------------------------------
df = df.sort_values(['City', 'Date'])
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df = df.dropna(subset=['AQI'])

# -------------------------------
# GLOBAL CITY SELECT
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", df['City'].unique())

base_df = df[df['City'] == selected_city].copy()
base_df = base_df.sort_values('Date')
base_df['AQI'] = base_df['AQI'].ffill().bfill()

# -------------------------------
# CREATE SEPARATE DATAFRAMES
# -------------------------------

# Trend
trend_df = base_df.copy()
trend_df['rolling'] = trend_df['AQI'].rolling(30).mean()
trend_df = trend_df.dropna().reset_index(drop=True)

# Detection
detect_df = base_df.copy()
detect_df['mean'] = detect_df['AQI'].rolling(30).mean()
detect_df['std'] = detect_df['AQI'].rolling(30).std()

detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']
detect_df['z'] = detect_df['z'].replace([np.inf, -np.inf], np.nan)
detect_df['z'] = detect_df['z'].fillna(0)

detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3
detect_df = detect_df.dropna().reset_index(drop=True)

# Isolation Forest
model_df = detect_df.copy()
scaler = StandardScaler()
scaled = scaler.fit_transform(model_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
model_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# KNN
df3 = base_df.copy()
scaled_knn = scaler.fit_transform(df3[['AQI']])
knn = KNN(contamination=0.05)
df3['knn_anomaly'] = knn.fit_predict(scaled_knn) == 1

# -------------------------------
# LSTM PREP
# -------------------------------
df_lstm = None

if TF_AVAILABLE:
    data = base_df[['AQI']].values
    scaler_lstm = MinMaxScaler()
    data_scaled = scaler_lstm.fit_transform(data)

    def create_sequences(data, seq_len=30):
        sequences = []
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
        return np.array(sequences)

    seq_len = 30
    X = create_sequences(data_scaled, seq_len)

    if len(X) > 50:
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_len, 1)),
            RepeatVector(seq_len),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(1))
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, X, epochs=3, batch_size=32, verbose=0)

        X_pred = model.predict(X)
        mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))

        threshold = np.percentile(mse, 95)
        anomalies = mse > threshold

        df_lstm = base_df.iloc[seq_len:].copy()
        df_lstm['lstm_anomaly'] = anomalies

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "🤖 KNN",
    "🧠 LSTM",
    "📊 Compare Models",
    "💡 Insights"
])

# -------------------------------
# TAB 1
# -------------------------------
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['AQI'], name='AQI'))
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['rolling'], name='30-day Avg'))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2
# -------------------------------
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_df['Date'], y=model_df['AQI'], name='AQI'))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['iso_anomaly']]['Date'],
        y=model_df[model_df['iso_anomaly']]['AQI'],
        mode='markers',
        name='Isolation Forest',
        marker=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['z_anomaly']]['Date'],
        y=model_df[model_df['z_anomaly']]['AQI'],
        mode='markers',
        name='Z-score',
        marker=dict(color='orange')
    ))

    st.plotly_chart(fig, use_container_width=True)

    threshold = model_df['AQI'].quantile(0.95)
    model_df['gt'] = model_df['AQI'] > threshold

    st.write("Precision:", precision_score(model_df['gt'], model_df['iso_anomaly']))
    st.write("Recall:", recall_score(model_df['gt'], model_df['iso_anomaly']))
    st.write("F1:", f1_score(model_df['gt'], model_df['iso_anomaly']))

# -------------------------------
# TAB 3
# -------------------------------
with tab3:
    st.write("KNN anomalies:", df3['knn_anomaly'].sum())

# -------------------------------
# TAB 4
# -------------------------------
with tab4:
    if df_lstm is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_lstm['Date'], y=df_lstm['AQI'], name='AQI'))

        fig.add_trace(go.Scatter(
            x=df_lstm[df_lstm['lstm_anomaly']]['Date'],
            y=df_lstm[df_lstm['lstm_anomaly']]['AQI'],
            mode='markers',
            name='LSTM',
            marker=dict(color='purple')
        ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("LSTM not available")

# -------------------------------
# TAB 5 (🔥 BEST PART)
# -------------------------------
with tab5:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=base_df['Date'],
        y=base_df['AQI'],
        name='AQI',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['iso_anomaly']]['Date'],
        y=model_df[model_df['iso_anomaly']]['AQI'],
        mode='markers',
        name='Isolation Forest',
        marker=dict(color='red', size=8)
    ))

    fig.add_trace(go.Scatter(
        x=model_df[model_df['z_anomaly']]['Date'],
        y=model_df[model_df['z_anomaly']]['AQI'],
        mode='markers',
        name='Z-score',
        marker=dict(color='orange', size=6)
    ))

    fig.add_trace(go.Scatter(
        x=df3[df3['knn_anomaly']]['Date'],
        y=df3[df3['knn_anomaly']]['AQI'],
        mode='markers',
        name='KNN',
        marker=dict(color='green', size=7)
    ))

    if df_lstm is not None:
        fig.add_trace(go.Scatter(
            x=df_lstm[df_lstm['lstm_anomaly']]['Date'],
            y=df_lstm[df_lstm['lstm_anomaly']]['AQI'],
            mode='markers',
            name='LSTM',
            marker=dict(color='purple', size=9)
        ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 🧠 Model Insights
    - Isolation Forest → global anomalies  
    - Z-score → statistical spikes  
    - KNN → density-based anomalies  
    - LSTM → time pattern anomalies  
    """)

# -------------------------------
# TAB 6
# -------------------------------
with tab6:
    recent = base_df.tail(30)
    avg = recent['AQI'].mean()

    if avg > 200:
        st.error("Severe pollution")
    elif avg > 100:
        st.warning("Moderate pollution")
    else:
        st.success("Good air quality")

    st.write("Average AQI:", avg)
