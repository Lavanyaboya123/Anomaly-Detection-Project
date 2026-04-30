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
# CLEAN DATA (CRITICAL FIX)
# -------------------------------
df = df.sort_values(['City', 'Date'])
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df = df.dropna(subset=['AQI'])

# -------------------------------
# GLOBAL CITY SELECT (SYNC ALL TABS)
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", df['City'].unique())

city_df = df[df['City'] == selected_city].copy()
city_df = city_df.sort_values('Date')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "🤖 ML Models",
    "🧠 LSTM",
    "💡 Insights"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {selected_city}")

    city_df['rolling'] = city_df['AQI'].rolling(30).mean()
    trend_df = city_df.dropna().reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['AQI'], name='AQI'))
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['rolling'], name='30-day Avg'))

    fig.update_layout(
        title=f"AQI Trend with Moving Average - {selected_city}",
        xaxis_title="Date",
        yaxis_title="AQI"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2: ANOMALY DETECTION
# -------------------------------
with tab2:
    st.subheader("🔍 Anomaly Detection")

    df2 = city_df.copy()

    # Z-score
    df2['mean'] = df2['AQI'].rolling(30).mean()
    df2['std'] = df2['AQI'].rolling(30).std()
    df2['z'] = (df2['AQI'] - df2['mean']) / df2['std']

    df2['z'] = df2['z'].replace([np.inf, -np.inf], np.nan)
    df2['z'] = df2['z'].fillna(0)

    df2['z_anomaly'] = np.abs(df2['z']) > 3

    df2 = df2.dropna().reset_index(drop=True)

    # Isolation Forest
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df2[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
    df2['iso_anomaly'] = iso.fit_predict(scaled) == -1

    # Graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df2['Date'],
        y=df2['AQI'],
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=df2[df2['iso_anomaly']]['Date'],
        y=df2[df2['iso_anomaly']]['AQI'],
        mode='markers',
        name='Isolation Forest',
        marker=dict(color='red', size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df2[df2['z_anomaly']]['Date'],
        y=df2[df2['z_anomaly']]['AQI'],
        mode='markers',
        name='Z-score',
        marker=dict(color='orange', size=6)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Evaluation
    threshold = df2['AQI'].quantile(0.95)
    df2['gt'] = df2['AQI'] > threshold

    precision = precision_score(df2['gt'], df2['iso_anomaly'])
    recall = recall_score(df2['gt'], df2['iso_anomaly'])
    f1 = f1_score(df2['gt'], df2['iso_anomaly'])

    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Debug
    st.write("Total Data:", len(df2))
    st.write("Anomalies:", df2['iso_anomaly'].sum())

# -------------------------------
# TAB 3: KNN
# -------------------------------
with tab3:
    st.subheader("🤖 KNN Anomaly Detection")

    df3 = city_df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df3[['AQI']])

    knn = KNN(contamination=0.05)
    df3['knn_anomaly'] = knn.fit_predict(scaled) == 1

    st.write(f"KNN detected {df3['knn_anomaly'].sum()} anomalies")

# -------------------------------
# TAB 4: LSTM
# -------------------------------
with tab4:
    st.subheader("🧠 LSTM Autoencoder")

    if not TF_AVAILABLE:
        st.error("TensorFlow not available")
    else:
        data = city_df[['AQI']].values

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        def create_sequences(data, seq_len=30):
            sequences = []
            for i in range(len(data) - seq_len):
                sequences.append(data[i:i+seq_len])
            return np.array(sequences)

        seq_len = 30
        X = create_sequences(data_scaled, seq_len)

        if len(X) < 50:
            st.warning("Not enough data for LSTM")
        else:
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

            df_lstm = city_df.iloc[seq_len:].copy()
            df_lstm['lstm_anomaly'] = anomalies

            st.write(f"LSTM detected {df_lstm['lstm_anomaly'].sum()} anomalies")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_lstm['Date'],
                y=df_lstm['AQI'],
                name='AQI'
            ))

            fig.add_trace(go.Scatter(
                x=df_lstm[df_lstm['lstm_anomaly']]['Date'],
                y=df_lstm[df_lstm['lstm_anomaly']]['AQI'],
                mode='markers',
                name='LSTM',
                marker=dict(color='purple', size=8)
            ))

            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 5: INSIGHTS
# -------------------------------
with tab5:
    st.subheader("💡 Smart Insights")

    recent = city_df.tail(30)

    avg = recent['AQI'].mean()
    max_val = recent['AQI'].max()

    if avg > 200:
        st.error("⚠️ Severe pollution trend detected")
    elif avg > 100:
        st.warning("⚠️ Moderate pollution levels")
    else:
        st.success("✅ Air quality is good")

    st.write(f"Recent Avg AQI: {avg:.2f}")
    st.write(f"Peak AQI: {max_val}")
