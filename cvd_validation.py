import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
X = pd.read_csv("data/feature_extraction/selected_features.csv")
y = pd.read_csv("data/feature_extraction/target.csv")

print(f"Loaded features shape: {X.shape}")
print(f"Loaded target shape: {y.shape}")

# 2. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Train a fresh model (to match the current feature count)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train.values.ravel())

# 5. Save the model (optional)
joblib.dump(model, "models/random_forest_retrained.pkl")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
