import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")
st.title("🌫️ AQI Anomaly Detection Dashboard")

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
# SELECT CITY
# -------------------------------
city = st.sidebar.selectbox("Select City", df['City'].unique())
city_df = df[df['City'] == city].copy()

# -------------------------------
# CLEAN DATA
# -------------------------------
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df['AQI'] = city_df['AQI'].ffill().bfill()

# -------------------------------
# FEATURES
# -------------------------------
city_df['mean'] = city_df['AQI'].rolling(30, min_periods=10).mean()
city_df['std'] = city_df['AQI'].rolling(30, min_periods=10).std()

city_df['z'] = (city_df['AQI'] - city_df['mean']) / city_df['std']
city_df['z_anomaly'] = np.abs(city_df['z']) > 3

# -------------------------------
# MODELS
# -------------------------------
scaler = StandardScaler()
scaled = scaler.fit_transform(city_df[['AQI']])

iso = IsolationForest(contamination=0.05, random_state=42)
city_df['iso_anomaly'] = iso.fit_predict(scaled) == -1

knn = KNN(contamination=0.05)
lof = LOF(contamination=0.05)

city_df['knn_anomaly'] = knn.fit_predict(scaled) == 1
city_df['lof_anomaly'] = lof.fit_predict(scaled) == 1

# -------------------------------
# EXPLANATION
# -------------------------------
def get_reason(row):
    if pd.isna(row['mean']):
        return "Insufficient data"
    elif row['AQI'] > row['mean'] + 2*row['std']:
        return "High pollution spike (traffic / weather / industry)"
    elif row['AQI'] < row['mean'] - 2*row['std']:
        return "Sudden drop (rain / low activity)"
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

city_df['reason'] = city_df.apply(get_reason, axis=1)
city_df['severity'] = city_df.apply(get_severity, axis=1)

# -------------------------------
# TABS (NEW STRUCTURE)
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🤖 Models",
    "💡 Insights",
    "📊 Conclusion"
])

# -------------------------------
# TAB 1: TREND
# -------------------------------
with tab1:
    st.subheader(f"AQI Trend - {city}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name="AQI"))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2: MODELS
# -------------------------------
with tab2:
    st.subheader("Model Comparison")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['Date'], y=city_df['AQI'], name="AQI"))

    def add_points(col, color, name):
        subset = city_df[city_df[col]]
        fig.add_trace(go.Scatter(
            x=subset['Date'],
            y=subset['AQI'],
            mode='markers',
            marker=dict(color=color, size=7),
            name=name,
            text=subset['reason'],
            customdata=subset['severity'],
            hovertemplate=
            "AQI: %{y}<br>" +
            "Reason: %{text}<br>" +
            "Severity: %{customdata}<extra></extra>"
        ))

    add_points('z_anomaly', 'orange', 'Z-score')
    add_points('iso_anomaly', 'red', 'Isolation Forest')
    add_points('knn_anomaly', 'green', 'KNN')
    add_points('lof_anomaly', 'purple', 'LOF')

    st.plotly_chart(fig, use_container_width=True)

    # Evaluation
    threshold = city_df['AQI'].quantile(0.95)
    city_df['gt'] = city_df['AQI'] > threshold

    scores = {}
    for m in ['z_anomaly','iso_anomaly','knn_anomaly','lof_anomaly']:
        p = precision_score(city_df['gt'], city_df[m], zero_division=0)
        r = recall_score(city_df['gt'], city_df[m], zero_division=0)
        f = f1_score(city_df['gt'], city_df[m], zero_division=0)

        scores[m] = f
        st.write(f"{m}: F1 = {f:.2f}")

    best_model = max(scores, key=scores.get)

    st.success(f"🏆 Best Model: {best_model}")

    # Bar chart
    df_score = pd.DataFrame({
        "Model": scores.keys(),
        "F1 Score": scores.values()
    })

    st.plotly_chart(px.bar(df_score, x="Model", y="F1 Score"),
                    use_container_width=True)

# -------------------------------
# TAB 3: INSIGHTS
# -------------------------------
with tab3:
    st.subheader("Why anomalies happened")

    anomalies = city_df[city_df['iso_anomaly']].tail(5)

    for _, row in anomalies.iterrows():
        st.write(
            f"{row['Date'].date()} → AQI {row['AQI']} "
            f"({row['severity']}) → {row['reason']}"
        )

    st.subheader("Real-world impact")
    st.markdown("""
    - Detects pollution spikes early  
    - Useful for smart city monitoring  
    - Helps environmental decision-making  
    """)

# -------------------------------
# TAB 4: CONCLUSION
# -------------------------------
with tab4:
    st.subheader("Final Conclusion")

    st.write(f"Best model: {best_model}")

    st.markdown("""
    - Isolation Forest captures complex patterns  
    - Z-score detects extreme spikes  
    - KNN & LOF identify density anomalies  

    👉 Combining models improves accuracy
    """)
