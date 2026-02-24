import streamlit as st
import pandas as pd
from src.model_training import load_model

# Load trained model
model = load_model("model/fraud_detection_model.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details:")

# Get exact feature names used during training
feature_names = model.feature_names_in_

user_input = {}

# Generate input fields dynamically
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Predict button
if st.button("Predict"):

    input_df = pd.DataFrame([user_input])

    # 🔥 Get fraud probability
    proba = model.predict_proba(input_df)[0][1]

    st.write(f"Fraud Probability: {proba:.4f}")

    # 🔥 Apply your business threshold (0.3)
    if proba >= 0.3:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")