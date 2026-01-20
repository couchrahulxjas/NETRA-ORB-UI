import pandas as pd

INPUT_CSV = "fleet_summary.csv"
OUTPUT_CSV = "fleet_summary_with_behavior.csv"


df = pd.read_csv(INPUT_CSV)


# These names must match your CSV
MEAN_COL = "mean_error"
P95_COL = "p95_error"
ANOM_COL = "anomaly_count"


def classify_satellite(row):
    mean_err = row[MEAN_COL]
    p95_err = row[P95_COL]
    anomalies = row[ANOM_COL]

    
    if anomalies > 30 or p95_err > 6:
        return "UNSTABLE"

    
    elif anomalies > 5 or p95_err > 3:
        return "MODERATELY_ACTIVE"

    
    else:
        return "STABLE"

# Apply classification
df["behavior_class"] = df.apply(classify_satellite, axis=1)


df.to_csv(OUTPUT_CSV, index=False)


print("✅ Behavior classification done.")
print("📄 Saved to:", OUTPUT_CSV)
print()
print("📊 Class distribution:")
print(df["behavior_class"].value_counts())
