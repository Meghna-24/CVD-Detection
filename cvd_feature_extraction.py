import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("./data/processed_cvd_data.csv")

# Define features and target variable
X = df.drop(
    columns=["cvd", "id", "bp_category", "bp_category_encoded"]
)  # Dropping non-numeric and redundant columns
y = df["cvd"]

# Compute mutual information for feature selection
mi_scores = mutual_info_classif(X, y, discrete_features="auto")
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 5))
mi_scores.plot(kind="bar")
plt.title("Feature Importance using Mutual Information")
plt.xlabel("Features")
plt.ylabel("Mutual Information Score")
# plt.show()
plt.savefig("./data/feature_extraction/feature_importance.png")

# Select top features based on MI score
selected_features = mi_scores[mi_scores > 0.01].index.tolist()
X_selected = X[selected_features]

# Save the dataset with selected features
X_selected.to_csv("./data/feature_extraction/selected_features.csv", index=False)
y.to_csv("./data/feature_extraction/target.csv", index=False)

print(
    "Feature selection completed. Selected features saved as 'selected_features.csv'."
)
