from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Load saved model pipeline
model = joblib.load("heart_disease_model.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    return "Heart Disease Prediction API"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # Create DataFrame from input JSON, keys must match feature names except target
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)[:, 1]

    return jsonify(
        {"prediction": int(prediction[0]), "probability": float(prediction_prob[0])}
    )


if __name__ == "__main__":
    app.run(debug=True)
