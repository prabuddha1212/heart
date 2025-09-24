import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define the feature names in the correct order. This is the key fix.
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Path for the model file
MODEL_FILE = "heart_disease_model.pkl"

# --- Model Creation (If not found) ---
# This block will create a dummy model so the app can run immediately.t tasl
if not os.path.exists(MODEL_FILE):
    st.warning(f"Model file '{MODEL_FILE}' not found. Creating a simple dummy model.")

    # Create dummy data with the correct feature names and order
    dummy_data = {
        "age": np.random.randint(29, 77, size=100),
        "Gender": np.random.randint(0, 2, size=100),
        "cp": np.random.randint(0, 4, size=100),
        "trestbps": np.random.randint(100, 200, size=100),
        "chol": np.random.randint(150, 300, size=100),
        "fbs": np.random.randint(0, 2, size=100),
        "restecg": np.random.randint(0, 3, size=100),
        "thalach": np.random.randint(100, 200, size=100),
        "exang": np.random.randint(0, 2, size=100),
        "oldpeak": np.random.uniform(0.0, 4.0, size=100),
        "slope": np.random.randint(0, 3, size=100),
        "ca": np.random.randint(0, 4, size=100),
        "thal": np.random.randint(0, 4, size=100),
    }
    df_dummy = pd.DataFrame(dummy_data, columns=FEATURE_NAMES)
    target = np.random.randint(0, 2, size=100)

    # Train a simple Logistic Regression model
    dummy_model = LogisticRegression(max_iter=1000)
    dummy_model.fit(df_dummy, target)

    # Save the dummy model to the expected file name
    joblib.dump(dummy_model, MODEL_FILE)
    st.success("Dummy model created. You can replace it with your real model file.")

# --- Load the Model ---
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"Error: The model file '{MODEL_FILE}' was not found.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Heart Disease Prediction App")
st.markdown(
    "Use the sliders and selectors to input patient data and predict the risk of heart disease."
)

# Create input widgets in the sidebar for better organization
with st.sidebar:
    st.header("Patient Data")
    age = st.slider("Age", 29, 77, 50)
    sex = st.radio(
        "Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0], index=0
    )
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            ("Typical Angina", 0),
            ("Atypical Angina", 1),
            ("Non-Anginal Pain", 2),
            ("Asymptomatic", 3),
        ],
        format_func=lambda x: x[0],
        index=0,
    )
    trestbps = st.slider("Resting Blood Pressure", 94, 200, 120)
    chol = st.slider("Cholesterol", 126, 564, 200)
    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dl?",
        options=[("Yes", 1), ("No", 0)],
        format_func=lambda x: x[0],
        index=1,
    )
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[
            ("Normal", 0),
            ("ST-T Wave Abnormality", 1),
            ("Ventricular Hypertrophy", 2),
        ],
        format_func=lambda x: x[0],
        index=0,
    )
    thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = st.radio(
        "Exercise Induced Angina?",
        options=[("Yes", 1), ("No", 0)],
        format_func=lambda x: x[0],
        index=1,
    )
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
    slope = st.selectbox(
        "Slope of the Peak Exercise ST Segment",
        options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
        format_func=lambda x: x[0],
        index=1,
    )
    ca = st.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3], index=0
    )
    thal = st.selectbox(
        "Thalassemia",
        options=[("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)],
        format_func=lambda x: x[0],
        index=0,
    )

# Create a dictionary from the user inputs
input_data = {
    "age": age,
    "sex": sex[1],
    "cp": cp[1],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs[1],
    "restecg": restecg[1],
    "thalach": thalach,
    "exang": exang[1],
    "oldpeak": oldpeak,
    "slope": slope[1],
    "ca": ca,
    "thal": thal[1],
}

# --- Prediction Logic ---
if st.button("Predict"):
    # Convert the input dictionary to a DataFrame, ensuring the correct column order.
    # This is the crucial part that fixes the ValueError.
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Prediction: High Risk of Heart Disease")
        st.markdown(f"**Probability:** {prediction_prob:.2f}")
    else:
        st.success(f"Prediction: Low Risk of Heart Disease")
        st.markdown(f"**Probability:** {prediction_prob:.2f}")
