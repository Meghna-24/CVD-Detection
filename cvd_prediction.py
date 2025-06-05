import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# --- Page config ---
st.set_page_config(page_title="CVD Risk Predictor", page_icon="🫀", layout="centered")

# --- Load model and scaler ---
model = joblib.load('./models/random_forest.pkl')
X_train = pd.read_csv('./data/feature_extraction/selected_features.csv')
feature_names = X_train.columns.tolist()
scaler = StandardScaler()
scaler.fit(X_train)

# --- Sidebar Info ---
st.sidebar.title("🩺 About")
st.sidebar.markdown("""
This app predicts the **risk of cardiovascular disease** based on health metrics.

- Model: Random Forest
- Input: 7 health features
- Output: Low or High risk
""")

# --- Main Title ---
st.title("🫀 Cardiovascular Disease Risk Predictor")
st.markdown("Enter your health information below to assess risk.")

# --- Input Fields ---
user_inputs = []
cols = st.columns(2)  # 2-column layout for inputs

for i, feature in enumerate(feature_names):
    col = cols[i % 2]  # Alternate between two columns
    with col:
        value = st.number_input(f"{feature}", step=1.0)
        user_inputs.append(value)

# --- Prediction Button ---
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame([user_inputs], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    # --- Result Display ---
    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ High risk of cardiovascular disease. Please consult a doctor.")
    else:
        st.success("✅ Low risk of cardiovascular disease. Keep maintaining a healthy lifestyle!")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ❤️ using Streamlit.")
