import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("./data/cvd_data.csv")  # Replace with actual file path

# Display basic info and check for missing values
print(df.info())
print(df.isnull().sum())

# Drop duplicate rows if any
df = df.drop_duplicates()

# Convert height and weight to numeric if not already
df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")


# Handle outliers in blood pressure (ap_hi and ap_lo)
def clean_bp(df):
    df = df[(df["ap_hi"] >= 90) & (df["ap_hi"] <= 200)]  # Normal systolic range
    df = df[(df["ap_lo"] >= 60) & (df["ap_lo"] <= 140)]  # Normal diastolic range
    return df


df = clean_bp(df)

# Recalculate BMI to ensure consistency
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

# Save the cleaned data
df.to_csv("./data/processed_cvd_data.csv", index=False)

print("Data cleaning completed. Cleaned data saved as 'data/processed_cvd_data.csv'.")
