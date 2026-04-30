import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.tsa.seasonal import seasonal_decompose

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
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file, parse_dates=['Date'])
    return pd.read_csv("city_day.csv", parse_dates=['Date'])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded_file)

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# SINGLE CITY PIPELINE (🔥 FIX)
# -------------------------------
selected_city = st.sidebar.selectbox("Select City", df['City'].unique())
city_df = df[df['City'] == selected_city].copy()

city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# rolling safely
city_df['mean'] = city_df['AQI'].rolling(30, min_periods=10).mean()
city_df['std'] = city_df['AQI'].rolling(30, min_periods=10).std()

city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
city_df['z_anomaly'] = np.abs(city_df['z']) > 3

# ML
scaler = StandardScaler()
scaled = scaler.fit_transform(city_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# severity + reason
def severity(aqi):
    if aqi > 300: return "High"
    elif aqi > 200: return "Medium"
    else: return "Low"

def reason(aqi, z):
    if aqi > 300: return "Severe spike"
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
    "🔍 Anomaly",
    "🤖 ML",
    "💡 Insights",
    "📊 Summary"
])

# -------------------------------
# TAB 1
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {selected_city}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI']))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2
# -------------------------------
with tab2:
    st.subheader("Anomaly Detection")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI']))

    anomalies = city_df[city_df['iso_anomaly']]

    fig.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['AQI'],
        mode='markers',
        marker=dict(color='red', size=8),
        text=anomalies['reason'],
        hovertemplate="Date:%{x}<br>%{text}<extra></extra>"
    ))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 3
# -------------------------------
with tab3:
    st.subheader("ML Results")

    st.write("Isolation Forest:", city_df['iso_anomaly'].sum())
    st.write("Z-score:", city_df['z_anomaly'].sum())

    if pyod_available:
        knn = KNN(contamination=0.05)
        city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1
        st.write("KNN:", city_df['knn_anomaly'].sum())

# -------------------------------
# TAB 4
# -------------------------------
with tab4:
    st.subheader("Insights")

    avg = city_df['AQI'].mean()
    max_val = city_df['AQI'].max()

    st.write(f"Avg AQI: {avg:.2f}")
    st.write(f"Max AQI: {max_val}")
    st.write(f"Anomalies: {city_df['iso_anomaly'].sum()}")

    if max_val > 300:
        st.error("Severe spikes 🚨")
    elif avg > 200:
        st.error("Unhealthy")
    elif avg > 100:
        st.warning("Moderate")
    else:
        st.success("Good")

# -------------------------------
# TAB 5
# -------------------------------
with tab5:
    st.write("Project summary: anomaly detection using ML + stats")
