import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Production-Level Dashboard | Statistical + Machine Learning")

# -------------------------------
# LOAD DATA (FAST)
# -------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file, parse_dates=['Date'])

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data("city_day.csv")

# Validate columns
if not all(col in df.columns for col in ['City', 'Date', 'AQI']):
    st.error("Dataset must contain City, Date, AQI columns")
    st.stop()

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙️ Controls")

city = st.sidebar.selectbox("Select City", df['City'].unique())

contamination = st.sidebar.slider("Isolation Forest Sensitivity", 0.01, 0.2, 0.05)
z_threshold = st.sidebar.slider("Z-score Threshold", 2.0, 5.0, 3.0)

min_date = df['Date'].min()
max_date = df['Date'].max()

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# -------------------------------
# FILTER DATA
# -------------------------------
city_df = df[df['City'] == city].copy()

city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

city_df = city_df[
    (city_df['Date'] >= pd.to_datetime(date_range[0])) &
    (city_df['Date'] <= pd.to_datetime(date_range[1]))
]

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "📊 Evaluation",
    "💡 Insights"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'],
                             mode='lines', name='AQI'))

    fig.update_layout(title="AQI Over Time", height=450)
    st.plotly_chart(fig, use_container_width=True)

    if len(city_df) > 365:
        st.subheader("Seasonal Decomposition")
        decomp = seasonal_decompose(
            city_df.set_index('Date')['AQI'],
            model='additive',
            period=365
        )
        st.pyplot(decomp.plot())

# -------------------------------
# TAB 2: DETECTION
# -------------------------------
with tab2:
    st.subheader("🔍 Anomaly Detection")

    # Z-score
    city_df['mean'] = city_df['AQI'].rolling(30).mean()
    city_df['std'] = city_df['AQI'].rolling(30).std()
    city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
    city_df['z_anomaly'] = np.abs(city_df['z']) > z_threshold

    # Isolation Forest
    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    iso = IsolationForest(contamination=contamination, random_state=42)
    city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Avg AQI", int(city_df['AQI'].mean()))
    col2.metric("🔥 Max AQI", int(city_df['AQI'].max()))
    col3.metric("⚠️ Anomalies", int(city_df['iso_anomaly'].sum()))
    col4.metric("📉 Min AQI", int(city_df['AQI'].min()))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'],
                             mode='lines', name='AQI'))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['iso_anomaly']]['Date'],
        y=city_df[city_df['iso_anomaly']]['AQI'],
        mode='markers',
        name='Isolation Forest',
        marker=dict(color='red', size=8)
    ))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['z_anomaly']]['Date'],
        y=city_df[city_df['z_anomaly']]['AQI'],
        mode='markers',
        name='Z-score',
        marker=dict(color='orange', size=7)
    ))

    fig.update_layout(title="Detected Anomalies", height=500)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 3: EVALUATION
# -------------------------------
with tab3:
    st.subheader("📊 Model Evaluation")

    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.2f}")
    col2.metric("Recall", f"{recall:.2f}")
    col3.metric("F1 Score", f"{f1:.2f}")

# -------------------------------
# TAB 4: INSIGHTS
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    - High AQI spikes indicate pollution events
    - Winter months show strong anomalies
    - Isolation Forest detects pattern anomalies
    - Z-score detects sudden spikes
    """)

    # Top 5 days
    st.subheader("🚨 Top 5 Pollution Days")
    top5 = city_df.sort_values(by='AQI', ascending=False).head(5)
    st.dataframe(top5[['Date', 'AQI']])

    # Model comparison
    st.subheader("⚖️ Model Comparison")
    st.write(f"Z-score detected: {city_df['z_anomaly'].sum()}")
    st.write(f"Isolation Forest detected: {city_df['iso_anomaly'].sum()}")

    # Download report
    report = city_df[city_df['iso_anomaly'] == True][['Date', 'AQI']]
    csv = report.to_csv(index=False).encode('utf-8')

    st.download_button(
        "📥 Download Anomaly Report",
        csv,
        file_name=f"{city}_anomalies.csv",
        mime='text/csv'
    )

    st.success("🚀 Project Ready for Resume & Interviews")
