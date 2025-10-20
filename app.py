import streamlit as st
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IwacuAI Homelessness Predictor", layout="centered")
st.title("IwacuAI Homelessness Predictor")
st.markdown("Predict the likelihood of homelessness based on personal and household data.")

MODEL_PATH = "iwacuAI_model.pkl"

# Function to train a simple fallback model
def train_fallback_model():
    st.warning("No trained model found. Training a fallback model...")
    # Dummy dataset — replace with real dataset if available
    data = pd.DataFrame({
        "age": [18, 25, 40, 60, 30, 50],
        "income": [100, 500, 1000, 200, 800, 300],
        "household_size": [1, 2, 4, 1, 3, 2],
        "homeless": [1, 0, 0, 1, 0, 1]
    })
    X = data[["age", "income", "household_size"]]
    y = data["homeless"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    st.success("Fallback model trained and saved as iwacuAI_model.pkl")
    return model

# Load the model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_fallback_model()

# User inputs
st.subheader("Enter your details:")
age = st.number_input("Age", min_value=0, max_value=120, value=25)
income = st.number_input("Monthly Income (USD)", min_value=0, value=500)
household_size = st.number_input("Household Size", min_value=1, value=2)

# Predict button
if st.button("Predict"):
    input_features = [[age, income, household_size]]
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][prediction]
    st.write(f"Prediction: **{'Homeless' if prediction==1 else 'Not Homeless'}**")
    st.write(f"Prediction confidence: **{probability*100:.2f}%**")
