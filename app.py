import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from statsmodels.tsa.seasonal import seasonal_decompose
from pyod.models.knn import KNN

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AQI Anomaly Detector", layout="wide")
st.title("🌫️ Advanced Air Quality Anomaly Detection")
st.markdown("Detecting unusual pollution patterns using ML models")

# -------------------------------
# Load Data
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    df = pd.read_csv('city_day.csv', parse_dates=['Date'])

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend",
    "🔍 Detection",
    "🤖 Model Compare",
    "💡 Insights"
])

# -------------------------------
# TAB 1: Trend
# -------------------------------
with tab1:
    city = st.selectbox("Select City", df['City'].unique())
    city_df = df[df['City'] == city].copy()

    # FIX: Safe NaN handling
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

    fig.update_layout(title="Air Quality Index Over Time")
    st.plotly_chart(fig, use_container_width=True)

    if len(city_df) > 365:
        st.subheader("Seasonal Pattern")
        decomp = seasonal_decompose(
            city_df.set_index('Date')['AQI'],
            model='additive',
            period=365
        )
        st.pyplot(decomp.plot())

# -------------------------------
# TAB 2: Detection
# -------------------------------
with tab2:
    st.subheader("Anomaly Detection")

    city = st.selectbox("City", df['City'].unique(), key="tab2")
    city_df = df[df['City'] == city].copy()

    city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
    city_df['AQI'] = city_df['AQI'].ffill().bfill()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_df[['AQI']])

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    city_df['iso'] = iso.fit_predict(scaled) == -1

    # KNN
    knn = KNN(contamination=0.05)
    city_df['knn'] = knn.fit_predict(scaled) == 1

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=city_df['Date'],
        y=city_df['AQI'],
        mode='lines',
        name='AQI'
    ))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['iso']]['Date'],
        y=city_df[city_df['iso']]['AQI'],
        mode='markers',
        name='Isolation Forest',
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=city_df[city_df['knn']]['Date'],
        y=city_df[city_df['knn']]['AQI'],
        mode='markers',
        name='KNN',
        marker=dict(size=8)
    ))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 3: Model Comparison
# -------------------------------
with tab3:
    st.subheader("Model Performance Comparison")

    threshold = city_df['AQI'].quantile(0.95)
    y_true = (city_df['AQI'] > threshold).astype(int)

    iso_pred = city_df['iso'].astype(int)
    knn_pred = city_df['knn'].astype(int)

    results = pd.DataFrame({
        "Model": ["Isolation Forest", "KNN"],
        "Precision": [
            precision_score(y_true, iso_pred),
            precision_score(y_true, knn_pred)
        ],
        "Recall": [
            recall_score(y_true, iso_pred),
            recall_score(y_true, knn_pred)
        ],
        "F1 Score": [
            f1_score(y_true, iso_pred),
            f1_score(y_true, knn_pred)
        ]
    })

    st.dataframe(results)

# -------------------------------
# TAB 4: Insights
# -------------------------------
with tab4:
    st.markdown("""
    ### Key Insights

    - Pollution spikes detected clearly using ML models  
    - Isolation Forest captures pattern anomalies  
    - KNN captures density anomalies  
    - Winter months show high AQI spikes  

    ### Use Cases
    - Smart cities  
    - Environmental monitoring  
    - Public health alerts  
    """)
