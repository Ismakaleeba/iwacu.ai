# 🏡 IwacuAI Dashboard
# Author: Ismael Kaleeba

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("iwacuAI_model.pkl")

st.set_page_config(page_title="IwacuAI Dashboard", page_icon="🏡", layout="centered")

# --- Header
st.title("🏡 IwacuAI")
st.subheader("AI for Predicting and Preventing Homelessness")
st.markdown("This tool predicts the likelihood of a person or family being at risk of homelessness using socioeconomic indicators.")

# --- User Inputs
st.header("Enter Household Information")

income = st.number_input("Monthly Income ($)", min_value=0, max_value=2000, value=500)
rent = st.number_input("Monthly Rent ($)", min_value=0, max_value=2000, value=300)
education_level = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Informal"])
household_size = st.slider("Household Size", 1, 10, 3)
health_index = st.slider("Health Index (0=poor, 1=excellent)", 0.0, 1.0, 0.5)
region = st.selectbox("Region", ["Urban", "Rural"])

# --- Create dataframe for prediction
input_data = pd.DataFrame({
    "income": [income],
    "rent": [rent],
    "education_level": [education_level],
    "employment_status": [employment_status],
    "household_size": [household_size],
    "health_index": [health_index],
    "region": [region]
})

# One-hot encode to match training data
input_encoded = pd.get_dummies(input_data)
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_features]

# --- Predict
if st.button("Predict Risk"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ This household is **AT RISK** of homelessness.\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ This household is **STABLE**.\nProbability of risk: {probability:.2%}")

    st.markdown("---")
    st.caption("Model developed by Ismael Kaleeba | IwacuAI (AI for Humanity)")

