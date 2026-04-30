import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")
st.title("🌫️ AQI Anomaly Detection Dashboard (Final Smart Version)")

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

# -------------------------------
# CITY FILTER
# -------------------------------
valid_cities = [
    c for c in df['City'].unique()
    if len(df[df['City'] == c]) > 150
]

selected_city = st.sidebar.selectbox("Select City", valid_cities)

city_df = df[df['City'] == selected_city].copy()
city_df['AQI'] = city_df['AQI'].interpolate()

# -------------------------------
# TREND
# -------------------------------
st.subheader(f"📈 AQI Trend - {selected_city}")

trend_df = city_df.copy()
trend_df['rolling'] = trend_df['AQI'].rolling(30, min_periods=10).mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['AQI'], name='AQI'))
fig1.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['rolling'], name='30-Day Avg'))

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
detect_df = city_df.copy()

detect_df['mean'] = detect_df['AQI'].rolling(30, min_periods=10).mean()
detect_df['std'] = detect_df['AQI'].rolling(30, min_periods=10).std()
detect_df = detect_df.dropna(subset=['mean', 'std'])

detect_df['z'] = (detect_df['AQI'] - detect_df['mean']) / detect_df['std']
detect_df['z_anomaly'] = np.abs(detect_df['z']) > 3

# Isolation Forest
scaler = StandardScaler()
scaled = scaler.fit_transform(detect_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
detect_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

# -------------------------------
# SEVERITY + EXPLANATION
# -------------------------------
def get_severity(aqi):
    if aqi > 300:
        return "High"
    elif aqi > 200:
        return "Medium"
    else:
        return "Low"

def get_reason(aqi, z):
    if aqi > 300:
        return "Severe pollution spike"
    elif aqi > 200:
        return "High pollution level"
    elif z < -3:
        return "Sudden drop"
    else:
        return "Unusual variation"

detect_df['severity'] = detect_df['AQI'].apply(get_severity)
detect_df['reason'] = detect_df.apply(lambda r: get_reason(r['AQI'], r['z']), axis=1)

# -------------------------------
# DECOMPOSITION
# -------------------------------
st.subheader("📊 Seasonal Decomposition + Anomalies")

ts_df = city_df.set_index('Date')

if len(ts_df) > 365:

    result = seasonal_decompose(ts_df['AQI'], model='additive', period=365)

    decomp_df = pd.DataFrame({
        'Date': ts_df.index,
        'Observed': result.observed,
        'Trend': result.trend,
        'Seasonal': result.seasonal,
        'Residual': result.resid
    }).dropna()

    # SAFE MERGE (NO ERROR)
    merged = pd.merge(
        decomp_df,
        detect_df[['Date', 'iso_anomaly', 'severity', 'reason']],
        on='Date',
        how='left'
    )

    merged['iso_anomaly'] = merged['iso_anomaly'].fillna(False)

    fig2 = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed AQI", "Trend", "Seasonal", "Residual")
    )

    # Observed
    fig2.add_trace(go.Scatter(
        x=merged['Date'],
        y=merged['Observed'],
        name='AQI'
    ), row=1, col=1)

    # Anomalies with hover
    anomalies = merged[merged['iso_anomaly']]

    fig2.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['Observed'],
        mode='markers',
        marker=dict(color='red', size=7),
        text=anomalies['reason'],
        customdata=anomalies['severity'],
        hovertemplate=
        "<b>Date:</b> %{x}<br>" +
        "<b>AQI:</b> %{y}<br>" +
        "<b>Severity:</b> %{customdata}<br>" +
        "<b>Reason:</b> %{text}<extra></extra>",
        name='Anomaly'
    ), row=1, col=1)

    # Trend
    fig2.add_trace(go.Scatter(x=merged['Date'], y=merged['Trend']), row=2, col=1)

    # Seasonal
    fig2.add_trace(go.Scatter(x=merged['Date'], y=merged['Seasonal']), row=3, col=1)

    # Residual
    fig2.add_trace(go.Scatter(x=merged['Date'], y=merged['Residual']), row=4, col=1)

    fig2.update_layout(height=900)

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# ANOMALY GRAPH
# -------------------------------
st.subheader("🔍 Anomaly Detection Overview")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=detect_df['Date'],
    y=detect_df['AQI'],
    name='AQI'
))

fig3.add_trace(go.Scatter(
    x=detect_df[detect_df['iso_anomaly']]['Date'],
    y=detect_df[detect_df['iso_anomaly']]['AQI'],
    mode='markers',
    marker=dict(color='red', size=8),
    text=detect_df[detect_df['iso_anomaly']]['reason'],
    hovertemplate=
    "<b>Date:</b> %{x}<br>" +
    "<b>AQI:</b> %{y}<br>" +
    "<b>Reason:</b> %{text}<extra></extra>",
    name='Isolation Forest'
))

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# TEXT EXPLANATION
# -------------------------------
st.subheader("🧠 Anomaly Explanation")

top_anomalies = detect_df[detect_df['iso_anomaly']].head(5)

for _, row in top_anomalies.iterrows():
    st.write(f"📅 {row['Date'].date()} → {row['reason']} (Severity: {row['severity']})")

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("💡 Insights")

st.markdown(f"""
- City: **{selected_city}**
- Total Records: {len(city_df)}
- Isolation Forest Anomalies: {detect_df['iso_anomaly'].sum()}
- Z-score Anomalies: {detect_df['z_anomaly'].sum()}

### Meaning:
- 🔴 High = Dangerous pollution
- 🟠 Medium = Moderate risk
- 🟢 Low = Mild variation
""")
