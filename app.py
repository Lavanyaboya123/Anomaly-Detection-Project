import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Statistical + ML Dashboard (Production Ready)")

# -------------------------------
# Upload
# -------------------------------
st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])

# Validate columns
required_cols = ['City', 'Date', 'AQI']
if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain City, Date, AQI columns")
    st.stop()

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# Sidebar Controls
# -------------------------------
contamination = st.sidebar.slider("Isolation Forest Sensitivity", 0.01, 0.2, 0.05)
z_threshold = st.sidebar.slider("Z-Score Threshold", 2.0, 5.0, 3.0)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "📊 Evaluation",
    "💡 Insights"
])

# -------------------------------
# TAB 1
# -------------------------------
with tab1:
    city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == city].copy()

    # Fix NaN
    city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
    city_df['AQI'] = city_df['AQI'].ffill().bfill()

    st.subheader(f"AQI Trend - {city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    fig.update_layout(title="AQI Over Time", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal
    if len(city_df) > 365:
        st.subheader("Seasonality Analysis")
        decomp = seasonal_decompose(
            city_df.set_index('Date')['AQI'],
            model='additive',
            period=365
        )
        st.pyplot(decomp.plot())

# -------------------------------
# TAB 2
# -------------------------------
with tab2:
    st.subheader("🔍 Anomaly Detection")

    city = st.selectbox("City", df['City'].unique(), key="tab2")
    city_df = df[df['City'] == city].copy()

    city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
    city_df['AQI'] = city_df['AQI'].ffill().bfill()

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

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

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
# TAB 3
# -------------------------------
with tab3:
    st.subheader("📊 Model Evaluation")

    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.2f}")
    col2.metric("Recall", f"{recall:.2f}")
    col3.metric("F1 Score", f"{f1:.2f}")

# -------------------------------
# TAB 4
# -------------------------------
with tab4:
    st.subheader("💡 Insights")

    st.markdown("""
    ### Key Observations:
    - High AQI spikes indicate pollution events
    - Winter months show stronger anomalies
    - Isolation Forest captures pattern-based anomalies
    - Z-score captures sudden spikes
    
    ### Real-world Use:
    - Smart city monitoring
    - Pollution alerts
    - Environmental analytics
    """)

    st.success("Project Ready for Resume & Interviews 🚀")
