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
