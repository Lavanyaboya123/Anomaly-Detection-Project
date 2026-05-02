# -------------------------------
# ANOMALY EXPLANATION
# -------------------------------
def get_reason(row):
    if row['AQI'] > row['mean'] + 2*row['std']:
        return "High pollution spike"
    elif row['AQI'] < row['mean'] - 2*row['std']:
        return "Sudden drop"
    else:
        return "Normal variation"

def get_severity(row):
    if abs(row['z']) > 4:
        return "High"
    elif abs(row['z']) > 3:
        return "Medium"
    else:
        return "Low"

city_df['reason'] = city_df.apply(get_reason, axis=1)
city_df['severity'] = city_df.apply(get_severity, axis=1)
# -------------------------------
# ANOMALY EXPLANATION (FIXED)
# -------------------------------
def get_reason(row):
    if pd.isna(row['mean']) or pd.isna(row['std']):
        return "Insufficient data"
    elif row['AQI'] > row['mean'] + 2*row['std']:
        return "High pollution spike"
    elif row['AQI'] < row['mean'] - 2*row['std']:
        return "Sudden drop"
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

# APPLY (IMPORTANT → AFTER city_df exists)
city_df['reason'] = city_df.apply(get_reason, axis=1)
city_df['severity'] = city_df.apply(get_severity, axis=1)
