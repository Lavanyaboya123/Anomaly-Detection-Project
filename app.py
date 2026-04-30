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

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv", parse_dates=['Date'])
    df = df.sort_values(['City', 'Date']).reset_index(drop=True)
    return df

df = load_data()

# -------------------------------
# GLOBAL CITY SELECTION (IMPORTANT FIX)
# -------------------------------
selected_city = st.sidebar.selectbox("🌍 Select City", df['City'].unique())

city_df = df[df['City'] == selected_city].copy()

# -------------------------------
# CLEAN DATA (FIX ALL ERRORS)
# -------------------------------
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Anomalies",
    "📊 Evaluation",
    "💡 Insights"
])

# ===============================
# TAB 1: TREND
# ===============================
with tab1:
    st.subheader(f"AQI Trend - {selected_city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Seasonal Decomposition
    if len(city_df) > 365:
        st.subheader("Seasonality Analysis")

        decomp = seasonal_decompose(
            city_df.set_index('Date')['AQI'],
            model='additive',
            period=365
        )

        fig2 = decomp.plot()
        st.pyplot(fig2)

# ===============================
# TAB 2: ANOMALIES
# ===============================
with tab2:
    st.subheader("Anomaly Detection")

    # Z-score
    city_df['mean'] = city_df['AQI'].rolling(30).mean()
    city_df['std'] = city_df['AQI'].rolling(30).std()

    city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
    city_df['z_anomaly'] = np.abs(city_df['z']) > 3

    # Isolation Forest
    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    iso = IsolationForest(contamination=0.05, random_state=42)
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
        name='Anomalies',
        marker=dict(color='red', size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    st.subheader("🧠 What happened?")

    anomalies = city_df[city_df['iso_anomaly']]

    if anomalies.empty:
        st.info("No anomalies detected.")
    else:
        for _, row in anomalies.head(5).iterrows():
            if row['AQI'] > 300:
                st.write(f"{row['Date'].date()} → Severe pollution spike")
            elif row['AQI'] > 200:
                st.write(f"{row['Date'].date()} → High pollution")
            else:
                st.write(f"{row['Date'].date()} → Unusual variation")

# ===============================
# TAB 3: EVALUATION
# ===============================
with tab3:
    st.subheader("Model Evaluation")

    threshold = city_df['AQI'].quantile(0.95)
    city_df['ground_truth'] = city_df['AQI'] > threshold

    y_true = city_df['ground_truth'].astype(int)
    y_pred = city_df['iso_anomaly'].astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.metric("Precision", round(precision, 2))
    st.metric("Recall", round(recall, 2))
    st.metric("F1 Score", round(f1, 2))

# ===============================
# TAB 4: INSIGHTS
# ===============================
with tab4:
    st.subheader("Insights for Selected City")

    avg_aqi = city_df['AQI'].mean()
    max_aqi = city_df['AQI'].max()
    anomaly_count = city_df['iso_anomaly'].sum()

    st.write(f"Average AQI: {avg_aqi:.2f}")
    st.write(f"Maximum AQI: {max_aqi:.2f}")
    st.write(f"Total anomalies detected: {anomaly_count}")

    if avg_aqi > 200:
        st.warning("City has consistently high pollution levels.")
    elif avg_aqi > 100:
        st.info("Moderate pollution levels.")
    else:
        st.success("Good air quality overall.")
