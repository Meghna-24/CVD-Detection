import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('./models/random_forest.pkl')  # or random_forest.pkl

# Load training data for feature names and scaling
X_train = pd.read_csv('./data/feature_extraction/selected_features.csv')
feature_names = X_train.columns.tolist()

# Fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Streamlit UI
st.title("Cardiovascular Disease Risk Prediction")

st.write("Please enter the following information:")

# Collect 7 inputs dynamically
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", step=1.0)
    user_inputs.append(value)

if st.button("Predict Risk"):
    # Convert to DataFrame with column names
    input_df = pd.DataFrame([user_inputs], columns=feature_names)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Output result
    if prediction[0] == 1:
        st.error("High risk of cardiovascular disease.")
    else:
        st.success("Low risk of cardiovascular disease.")
