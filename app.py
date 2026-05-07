import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AQI Anomaly Detection System",
    layout="wide"
)

st.title("🌫️ AQI Anomaly Detection System")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file, parse_dates=['Date'])
    return pd.read_csv("city_day.csv", parse_dates=['Date'])

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

df = load_data(uploaded_file)

# ---------------------------------------------------
# SORT DATA
# ---------------------------------------------------
df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# ---------------------------------------------------
# CITY SELECTION
# ---------------------------------------------------
st.markdown("## 📍 Select City")

city = st.selectbox(
    "Choose City",
    sorted(df['City'].dropna().unique())
)

city_df = df[df['City'] == city].copy()

# ---------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------
features = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO']

for col in features:
    city_df[col] = pd.to_numeric(
        city_df[col],
        errors='coerce'
    )

city_df[features] = city_df[features].ffill().bfill()

# ---------------------------------------------------
# ROLLING FEATURES
# ---------------------------------------------------
city_df['mean'] = city_df['AQI'].rolling(
    30,
    min_periods=10
).mean()

city_df['std'] = city_df['AQI'].rolling(
    30,
    min_periods=10
).std()

# ---------------------------------------------------
# Z-SCORE
# ---------------------------------------------------
city_df['z'] = (
    city_df['AQI'] - city_df['mean']
) / city_df['std']

city_df['z_anomaly'] = np.abs(city_df['z']) > 3

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------
scaler = StandardScaler()

scaled = scaler.fit_transform(
    city_df[features]
)

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

city_df['iso_anomaly'] = (
    iso.fit_predict(scaled) == -1
)

knn = KNN(contamination=0.05)

city_df['knn_anomaly'] = (
    knn.fit_predict(scaled) == 1
)

lof = LOF(contamination=0.05)

city_df['lof_anomaly'] = (
    lof.fit_predict(scaled) == 1
)

# ---------------------------------------------------
# ANOMALY EXPLANATION
# ---------------------------------------------------
def get_reason(row):

    if pd.isna(row['mean']) or pd.isna(row['std']):
        return "Insufficient data"

    elif row['AQI'] > row['mean'] + 2 * row['std']:
        return "Sudden pollution spike due to traffic or industrial activity"

    elif row['AQI'] < row['mean'] - 2 * row['std']:
        return "Sudden AQI drop due to rain or cleaner atmosphere"

    else:
        return "Normal AQI variation"


def get_severity(row):

    if pd.isna(row['z']):
        return "Low"

    elif abs(row['z']) > 4:
        return "High"

    elif abs(row['z']) > 3:
        return "Medium"

    else:
        return "Low"


city_df['reason'] = city_df.apply(
    get_reason,
    axis=1
)

city_df['severity'] = city_df.apply(
    get_severity,
    axis=1
)

# ---------------------------------------------------
# AQI CATEGORY
# ---------------------------------------------------
def aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 AQI Trend",
    "🔍 Detected Anomalies",
    "🤖 Model Comparison",
    "💡 Smart Insights"
])

# ---------------------------------------------------
# TAB 1 : AQI TREND
# ---------------------------------------------------
with tab1:

    st.subheader(f"AQI Trend Analysis - {city}")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=city_df['Date'],
            y=city_df['AQI'],
            mode='lines',
            name='AQI'
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# TAB 2 : ANOMALIES
# ---------------------------------------------------
with tab2:

    st.subheader("Detected Anomalies")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=city_df['Date'],
            y=city_df['AQI'],
            mode='lines',
            name='AQI'
        )
    )

    def add_points(col, color, name):

        subset = city_df[city_df[col]]

        fig2.add_trace(
            go.Scatter(
                x=subset['Date'],
                y=subset['AQI'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8
                ),
                name=name,
                text=subset['reason'],
                customdata=subset['severity'],
                hovertemplate=
                "<b>Date:</b> %{x}<br>" +
                "<b>AQI:</b> %{y}<br>" +
                "<b>Reason:</b> %{text}<br>" +
                "<b>Severity:</b> %{customdata}<extra></extra>"
            )
        )

    add_points('z_anomaly', 'orange', 'Z-Score')
    add_points('iso_anomaly', 'red', 'Isolation Forest')
    add_points('knn_anomaly', 'green', 'KNN')
    add_points('lof_anomaly', 'purple', 'LOF')

    fig2.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="AQI"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ---------------------------------------------------
# TAB 3 : MODEL COMPARISON
# ---------------------------------------------------
with tab3:

    st.subheader("Model Evaluation")

    threshold = city_df['AQI'].quantile(0.95)

    city_df['ground_truth'] = (
        city_df['AQI'] > threshold
    )

    models = [
        'z_anomaly',
        'iso_anomaly',
        'knn_anomaly',
        'lof_anomaly'
    ]

    scores = {}

    for m in models:

        y_true = city_df['ground_truth'].astype(int)
        y_pred = city_df[m].astype(int)

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        scores[m] = f1

        st.write(f"{m} → F1 Score: {f1:.2f}")

    best_model = max(
        scores,
        key=scores.get
    )

    st.success(f"🏆 Best Model: {best_model}")

    df_scores = pd.DataFrame({
        "Model": list(scores.keys()),
        "F1 Score": list(scores.values())
    })

    fig_bar = px.bar(
        df_scores,
        x="Model",
        y="F1 Score",
        color="Model",
        title="Model Performance Comparison"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

# ---------------------------------------------------
# TAB 4 : SMART INSIGHTS
# ---------------------------------------------------
with tab4:

    st.subheader("💡 Smart Insights")

    latest = city_df['AQI'].iloc[-1]
    avg = city_df['AQI'].mean()
    max_val = city_df['AQI'].max()

    st.metric(
        "Latest AQI",
        int(latest)
    )

    st.metric(
        "Average AQI",
        int(avg)
    )

    st.metric(
        "Worst AQI",
        int(max_val)
    )

    st.write(
        f"Current Condition: **{aqi_category(latest)}**"
    )

    st.write(
        f"Worst Recorded: **{aqi_category(max_val)}**"
    )

    # CURRENT AIR QUALITY STATUS
    if latest <= 50:
        st.success("✅ Current air quality is good")

    elif latest <= 100:
        st.info("ℹ️ Air quality is satisfactory")

    elif latest <= 200:
        st.warning("⚠️ Moderate pollution detected")

    elif latest <= 300:
        st.warning("⚠️ Poor air quality")

    else:
        st.error("🚨 Severe air pollution")

    # HISTORICAL POLLUTION STATUS
    if max_val > 300:
        st.error("🚨 Severe pollution occurred in the past")

    elif max_val > 200:
        st.warning("⚠️ High pollution spikes occurred in the past")

    # CITY INSIGHT
    if avg > 200:
        st.error(
            f"{city} shows consistently high pollution levels."
        )

    elif avg > 100:
        st.warning(
            f"{city} shows moderate pollution trends."
        )

    else:
        st.success(
            f"{city} has relatively cleaner air quality overall."
        )

    # UNHEALTHY DAYS
    danger_days = len(
        city_df[city_df['AQI'] > 150]
    )

    st.write(
        f"Days with unhealthy AQI (>150): {danger_days}"
    )

    # TOTAL ANOMALIES
    anomaly_count = city_df['iso_anomaly'].sum()

    st.write(
        f"Total anomalies detected: {anomaly_count}"
    )
