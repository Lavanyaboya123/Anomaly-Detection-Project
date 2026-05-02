import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Advanced Anomaly Detection", layout="wide")
st.title("🚀 Advanced Time Series Anomaly Detection")

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

df = df.sort_values(['City', 'Date'])

# -------------------------------
# SELECT CITY
# -------------------------------
city = st.sidebar.selectbox("Select City", df['City'].unique())
city_df = df[df['City'] == city].copy()

city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
scaled = scaler.fit_transform(city_df[['AQI']])

# -------------------------------
# 1. STATISTICAL MODEL (Z-SCORE)
# -------------------------------
city_df['mean'] = city_df['AQI'].rolling(30, min_periods=10).mean()
city_df['std'] = city_df['AQI'].rolling(30, min_periods=10).std()

city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
city_df['z_anomaly'] = np.abs(city_df['z']) > 3

# -------------------------------
# 2. ML MODEL (Isolation Forest)
# -------------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# 3. PYOD MODELS (KNN + LOF)
# -------------------------------
knn = KNN(contamination=0.05)
lof = LOF(contamination=0.05)

city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1
city_df['lof_anomaly'] = lof.fit_predict(scaled) == 1

# -------------------------------
# 4. LSTM AUTOENCODER
# -------------------------------
def create_sequences(data, window=10):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
    return np.array(X)

seq_data = create_sequences(scaled)

if len(seq_data) > 50:
    inputs = Input(shape=(seq_data.shape[1], seq_data.shape[2]))
    encoded = LSTM(32, activation='relu')(inputs)
    decoded = RepeatVector(seq_data.shape[1])(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)

    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')

    model.fit(seq_data, seq_data, epochs=5, batch_size=32, verbose=0)

    recon = model.predict(seq_data)
    loss = np.mean(np.abs(recon - seq_data), axis=(1,2))

    threshold = np.percentile(loss, 95)
    lstm_anomaly = loss > threshold

    # align with original data
    city_df['lstm_anomaly'] = False
    city_df.iloc[10:, city_df.columns.get_loc('lstm_anomaly')] = lstm_anomaly

else:
    city_df['lstm_anomaly'] = False

# -------------------------------
# GRAPH: ALL MODELS
# -------------------------------
st.subheader("📊 Model Comparison")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=city_df['Date'],
    y=city_df['AQI'],
    name='AQI'
))

def add_points(col, color, name):
    fig.add_trace(go.Scatter(
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
add_points('lstm_anomaly', 'black', 'LSTM')

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# EVALUATION
# -------------------------------
threshold_gt = city_df['AQI'].quantile(0.95)
city_df['ground_truth'] = city_df['AQI'] > threshold_gt

models = ['z_anomaly','iso_anomaly','knn_anomaly','lof_anomaly','lstm_anomaly']

st.subheader("📊 Model Evaluation")

for m in models:
    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df[m].astype(int)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    st.write(f"{m}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")
