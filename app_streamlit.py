import streamlit as st
import pandas as pd
import joblib

# Define the feature names in the correct order (matching the training data)
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

# Load model pipeline
model = joblib.load("heart_disease_model.pkl")

st.title("Heart Disease Prediction")
st.markdown(
    "Enter the required medical parameters below to predict the likelihood of heart disease."
)

# Collect input features from user with help texts
cp = st.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3],
    help="Type of chest pain: 0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic",
)
restecg = st.selectbox(
    "Resting ECG (restecg)",
    options=[0, 1, 2],
    help="Resting electrocardiogram results: 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy",
)
slope = st.selectbox(
    "Slope of Peak Exercise ST Segment (slope)",
    options=[0, 1, 2],
    help="Slope of the peak exercise ST segment: 0=upsloping, 1=flat, 2=downsloping",
)
thal = st.selectbox(
    "Thalassemia (thal)",
    options=[1, 2, 3],
    help="Thalassemia: 1=normal, 2=fixed defect, 3=reversable defect",
)

age = st.slider(
    "Age", 20, 100, 50, help="Age in years. Higher age increases heart disease risk."
)
trestbps = st.slider(
    "Resting Blood Pressure",
    80,
    200,
    120,
    help="Resting blood pressure in mm Hg. Normal range: 90-120 mm Hg.",
)
chol = st.slider(
    "Serum Cholesterol",
    100,
    600,
    200,
    help="Serum cholesterol in mg/dl. Desirable: <200 mg/dl.",
)
thalach = st.slider(
    "Max Heart Rate Achieved",
    60,
    220,
    150,
    help="Maximum heart rate achieved during exercise. Normal: 60-100% of max predicted.",
)
oldpeak = st.slider(
    "ST Depression Induced by Exercise",
    0.0,
    10.0,
    1.0,
    help="ST depression induced by exercise relative to rest. Higher values indicate ischemia.",
)
sex = st.selectbox(
    "Sex", options=[0, 1], help="Sex: 0=female, 1=male. Males have higher risk."
)
fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    help="Fasting blood sugar >120 mg/dl: 0=false, 1=true. Indicates diabetes risk.",
)
exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    help="Exercise-induced angina: 0=no, 1=yes. Presence indicates coronary artery disease.",
)
ca = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy (ca)",
    options=[0, 1, 2, 3, 4],
    help="Number of major vessels (0-3) colored by fluoroscopy. Higher numbers indicate more blockage.",
)

if st.button("Predict"):
    # Input validation
    warnings = []
    if age < 18 or age > 90:
        warnings.append("Age seems unusual for heart disease prediction.")
    if trestbps < 90 or trestbps > 180:
        warnings.append("Resting blood pressure is outside normal range.")
    if chol > 300:
        warnings.append("Cholesterol level is very high.")
    if thalach < 60:
        warnings.append("Max heart rate is unusually low.")
    if oldpeak > 5:
        warnings.append("ST depression is very high.")
    if warnings:
        for warning in warnings:
            st.warning(warning)

    input_dict = {
        "cp": cp,
        "restecg": restecg,
        "slope": slope,
        "thal": thal,
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "sex": sex,
        "fbs": fbs,
        "exang": exang,
        "ca": ca,
    }

    input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    # Improved output formatting
    prob_percent = proba * 100
    no_prob_percent = (1 - proba) * 100
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error(
            f"⚠️ High Risk: The model predicts the presence of heart disease with a probability of {prob_percent:.1f}%."
        )
        st.progress(prob_percent / 100)
    else:
        st.success(
            f"✅ Low Risk: The model predicts no heart disease with a probability of {no_prob_percent:.1f}%."
        )
        st.progress(no_prob_percent / 100)
    st.markdown(f"**Probability of Heart Disease:** {prob_percent:.1f}%")
    st.markdown(f"**Probability of No Heart Disease:** {no_prob_percent:.1f}%")
