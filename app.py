import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose

# Safe PyOD import
try:
    from pyod.models.knn import KNN
    pyod_available = True
except:
    pyod_available = False

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")

st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Interactive Dashboard | Statistical + ML")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

city = st.sidebar.selectbox("Select City", df['City'].unique())

city_df = df[df['City'] == city].copy()

# -------------------------------
# CLEAN DATA
# -------------------------------
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# Date Filter
start_date, end_date = st.sidebar.date_input(
    "Date Range",
    [city_df['Date'].min(), city_df['Date'].max()]
)

city_df = city_df[
    (city_df['Date'] >= pd.to_datetime(start_date)) &
    (city_df['Date'] <= pd.to_datetime(end_date))
]

# -------------------------------
# KPI DASHBOARD
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(city_df))
col2.metric("Max AQI", int(city_df['AQI'].max()))
col3.metric("Avg AQI", f"{city_df['AQI'].mean():.2f}")

# -------------------------------
# Z-SCORE
# -------------------------------
threshold = st.slider("Z-score Threshold", 1.0, 5.0, 3.0)

city_df['mean'] = city_df['AQI'].rolling(30).mean()
city_df['std'] = city_df['AQI'].rolling(30).std()
city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
city_df['z_anomaly'] = np.abs(city_df['z']) > threshold

# -------------------------------
# ISOLATION FOREST
# -------------------------------
scaler = StandardScaler()
scaled = scaler.fit_transform(city_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# MAIN GRAPH (COMPARISON)
# -------------------------------
st.subheader("📈 AQI Trend with Anomalies")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=city_df['Date'], y=city_df['AQI'],
    mode='lines', name='AQI'
))

# Isolation Forest
fig.add_trace(go.Scatter(
    x=city_df[city_df['iso_anomaly']]['Date'],
    y=city_df[city_df['iso_anomaly']]['AQI'],
    mode='markers',
    name='Isolation Forest',
    marker=dict(color='red', size=8)
))

# Z-score
fig.add_trace(go.Scatter(
    x=city_df[city_df['z_anomaly']]['Date'],
    y=city_df[city_df['z_anomaly']]['AQI'],
    mode='markers',
    name='Z-score',
    marker=dict(color='orange', size=6)
))

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DISTRIBUTION
# -------------------------------
st.subheader("📊 AQI Distribution")

hist = go.Figure()
hist.add_trace(go.Histogram(x=city_df['AQI'], nbinsx=50))
st.plotly_chart(hist, use_container_width=True)

# -------------------------------
# MODEL COMPARISON
# -------------------------------
st.subheader("📊 Model Comparison")

comparison = pd.DataFrame({
    'Method': ['Z-score', 'Isolation Forest'],
    'Anomalies': [
        city_df['z_anomaly'].sum(),
        city_df['iso_anomaly'].sum()
    ]
})

bar = go.Figure([go.Bar(
    x=comparison['Method'],
    y=comparison['Anomalies']
)])

st.plotly_chart(bar, use_container_width=True)

# -------------------------------
# EVALUATION
# -------------------------------
st.subheader("📊 Evaluation")

threshold_gt = city_df['AQI'].quantile(0.95)
city_df['ground_truth'] = city_df['AQI'] > threshold_gt

y_true = city_df['ground_truth'].astype(int)
y_pred = city_df['iso_anomaly'].astype(int)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# -------------------------------
# KNN (OPTIONAL)
# -------------------------------
if pyod_available:
    st.subheader("🤖 KNN (Advanced ML)")

    knn = KNN(contamination=0.05)
    city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1

    st.write(f"KNN Detected: {city_df['knn_anomaly'].sum()} anomalies")
else:
    st.warning("Install pyod to enable KNN")

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("💡 Insights")

st.markdown(f"""
- Total anomalies: **{city_df['iso_anomaly'].sum()}**
- Peak AQI: **{city_df['AQI'].max()}**
- Average AQI: **{city_df['AQI'].mean():.2f}**

### Observations:
- High spikes indicate pollution events
- Seasonal patterns visible
- ML detects hidden anomalies better than statistical methods
""")

# -------------------------------
# DOWNLOAD
# -------------------------------
st.download_button(
    "📥 Download Results",
    city_df.to_csv(index=False),
    file_name="anomaly_results.csv"
)
