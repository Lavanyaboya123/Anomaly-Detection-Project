import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from pyod.models.lof import LOF

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detection", layout="wide")
st.title("🌫️ AQI Anomaly Detection System")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file, parse_dates=['Date'])
    else:
        df = pd.read_csv("city_day.csv", parse_dates=['Date'])
    return df.sort_values(['City', 'Date']).reset_index(drop=True)

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
df = load_data(uploaded_file)

# -------------------------------
# CITY SELECTION
# -------------------------------
st.markdown("## 📍 Select City")
city = st.selectbox("Choose City", sorted(df['City'].unique()))
city_df = df[df['City'] == city].copy()

# -------------------------------
# DATA CLEANING
# -------------------------------
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].interpolate(method='linear').ffill().bfill()

# -------------------------------
# CACHED ANOMALY MODELS
# -------------------------------
@st.cache_resource
def get_models():
    return (
        StandardScaler(),
        IsolationForest(contamination=0.05, random_state=42),
        KNN(contamination=0.05),
        LOF(contamination=0.05)
    )

scaler, iso, knn, lof = get_models()

@st.cache_data
def run_anomaly_detection(data):
    data = data.copy()
    # Features
    window = 30
    data['mean'] = data['AQI'].rolling(window, min_periods=10).mean()
    data['std'] = data['AQI'].rolling(window, min_periods=10).std()
    data['z'] = (data['AQI'] - data['mean']) / data['std']
    data['z_anomaly'] = np.abs(data['z']) > 3

    # Scaled data for ML models
    scaled = scaler.fit_transform(data[['AQI']])

    # Isolation Forest
    data['iso_anomaly'] = iso.fit_predict(scaled) == -1

    # KNN & LOF
    data['knn_anomaly'] = knn.fit_predict(scaled) == 1
    data['lof_anomaly'] = lof.fit_predict(scaled) == 1

    # Reason & Severity
    def get_reason(row):
        if pd.isna(row['mean']) or pd.isna(row['std']):
            return "Insufficient data"
        elif row['AQI'] > row['mean'] + 2 * row['std']:
            return "High pollution spike"
        elif row['AQI'] < row['mean'] - 2 * row['std']:
            return "Sudden improvement (rain/wind)"
        else:
            return "Normal variation"

    def get_severity(row):
        if pd.isna(row['z']):
            return "Low"
        elif abs(row['z']) > 4:
            return "High"
        elif abs(row['z']) > 3:
            return "Medium"
        else:
            return "Low"

    data['reason'] = data.apply(get_reason, axis=1)
    data['severity'] = data.apply(get_severity, axis=1)

    return data

# Run anomaly detection
city_df = run_anomaly_detection(city_df)

# -------------------------------
# AQI CATEGORY
# -------------------------------
def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "🔍 Anomalies", "🤖 Models", "💡 Insights"])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {city}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], 
                           mode='lines', name='AQI', line=dict(color='blue')))
    fig.update_layout(xaxis_title="Date", yaxis_title="AQI", height=500)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2: ANOMALIES
# -------------------------------
with tab2:
    st.subheader("Detected Anomalies")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], 
                            mode='lines', name='AQI', line=dict(color='blue')))

    def add_anomalies(col, color, name):
        subset = city_df[city_df[col]]
        if not subset.empty:
            fig2.add_trace(go.Scatter(
                x=subset['Date'], y=subset['AQI'],
                mode='markers',
                marker=dict(color=color, size=9, line=dict(width=1)),
                name=name,
                text=subset['reason'],
                customdata=subset['severity'],
                hovertemplate=
                "<b>Date:</b> %{x}<br>" +
                "<b>AQI:</b> %{y}<br>" +
                "<b>Reason:</b> %{text}<br>" +
                "<b>Severity:</b> %{customdata}<extra></extra>"
            ))

    add_anomalies('z_anomaly', 'orange', 'Z-Score')
    add_anomalies('iso_anomaly', 'red', 'Isolation Forest')
    add_anomalies('knn_anomaly', 'green', 'KNN')
    add_anomalies('lof_anomaly', 'purple', 'LOF')

    fig2.update_layout(height=550)
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# TAB 3: MODEL COMPARISON
# -------------------------------
with tab3:
    st.subheader("Model Evaluation")
    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    models = ['z_anomaly', 'iso_anomaly', 'knn_anomaly', 'lof_anomaly']
    scores = {}

    for m in models:
        y_true = city_df['ground_truth'].astype(int)
        y_pred = city_df[m].astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, zero_division=0)
        scores[m] = f1
        st.write(f"**{m.replace('_anomaly','')}** → F1 Score: **{f1:.3f}**")

    best_model = max(scores, key=scores.get)
    st.success(f"🏆 Best Performing Model: **{best_model.replace('_anomaly','')}**")

    df_scores = pd.DataFrame({
        "Model": [m.replace('_anomaly','') for m in scores.keys()],
        "F1 Score": list(scores.values())
    })
    fig_bar = px.bar(df_scores, x="Model", y="F1 Score", color="Model", text_auto=True)
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------
# TAB 4: INSIGHTS
# -------------------------------
with tab4:
    st.subheader("💡 Smart Insights")
    latest = city_df['AQI'].iloc[-1]
    avg = city_df['AQI'].mean()
    max_val = city_df['AQI'].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest AQI", f"{int(latest)}")
    col2.metric("Average AQI", f"{int(avg)}")
    col3.metric("Highest AQI", f"{int(max_val)}")

    st.write(f"**Current Condition:** {aqi_category(latest)}")

    if latest > 200:
        st.error("🚨 Very Poor Air Quality - Avoid outdoor activities")
    elif latest > 150:
        st.warning("⚠️ Poor Air Quality")
    elif latest > 100:
        st.warning("Moderate Pollution")
    else:
        st.success("✅ Air quality is Acceptable")

    danger_days = len(city_df[city_df['AQI'] > 150])
    anomaly_count = city_df['iso_anomaly'].sum()

    st.write(f"**Unhealthy days (AQI > 150):** {danger_days}")
    st.write(f"**Total Anomalies Detected:** {anomaly_count}")
