import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# === Load and Check Data ===
X = pd.read_csv("./data/feature_extraction/selected_features.csv")
y = pd.read_csv("./data/feature_extraction/target.csv")

# Combine to handle missing values together
data = pd.concat([X, y], axis=1).dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# === Sanity Check for Invalid Values ===
assert X.isnull().sum().sum() == 0, "Missing values in features"
assert y.isnull().sum() == 0, "Missing values in target"
assert np.isfinite(X.values).all(), "Non-finite (inf/NaN) values in features"
assert np.isfinite(y.values).all(), "Non-finite (inf/NaN) values in target"

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# === Debug Prints ===
print("\u2705 Data check passed.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_train dtype:", X_train.dtype)

# === Define Models and Parameter Grid ===
param_grid = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000),
        "params": {"C": [0.1, 1, 10]},
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_jobs=-1),
        "params": {
            "n_estimators": [100],
            "max_depth": [10, None],
        },
    },
    "Support Vector Machine": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        },
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7]},
    },
}

# === Train and Save Models ===
results = {}
best_model_name = ""
best_model_instance = None
best_accuracy = 0

for name, config in param_grid.items():
    print(f"\n\U0001f504 Training {name}...")

    if name == "Support Vector Machine":
        search = RandomizedSearchCV(
            config["model"],
            param_distributions=config["params"],
            n_iter=5,
            cv=3,
            scoring="accuracy",
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
    else:
        search = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring="accuracy",
            verbose=1,
            n_jobs=-1
        )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"\u2705 {name} Best Params: {search.best_params_}")
    print(f"\u2705 {name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    model_filename = f"models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(best_model, model_filename)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_instance = best_model
        best_model_name = name

# === Save Best Model ===
if best_model_instance:
    joblib.dump(best_model_instance, "models/best_model.pkl")
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f}) saved as models/best_model.pkl")