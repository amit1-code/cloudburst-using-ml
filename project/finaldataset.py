import pandas as pd

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/preprocessed_data.csv")
df.columns = df.columns.str.strip()

# -------------------------------
# Create Storm_actual (DATA-DRIVEN)
# -------------------------------

# Dynamic thresholds
temp_thresh = df['TMPC'].quantile(0.75)
relh_thresh = df['RELH'].quantile(0.75)
dwpc_thresh = df['DWPC'].quantile(0.75)

# Create column
df['Storm_actual'] = (
    (df['TMPC'] > temp_thresh) |
    (df['RELH'] > relh_thresh) |
    (df['DWPC'] > dwpc_thresh)
).astype(int)

# -------------------------------
# Check distribution
# -------------------------------
print("Storm distribution:\n", df['Storm_actual'].value_counts())

# -------------------------------
# Save new dataset
# -------------------------------
df.to_csv("data/final_dataset.csv", index=False)

print("✅ New dataset saved as final_dataset.csv")