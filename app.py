import os
import warnings
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Suppress warnings that can be annoying during development
warnings.filter_token("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the name for our trained model file
MODEL_FILE = "heart_disease_model.pkl"
DATA_FILE = "heart.csv"

# --- Model Training and Saving Section ---
# This block of code will only run if the model file does not exist.
if not os.path.exists(MODEL_FILE):
    print(f"Model file '{MODEL_FILE}' not found.")
    print(f"Checking for data file '{DATA_FILE}' to train a new model.")

    # Check if the data file exists
    if not os.path.exists(DATA_FILE):
        print(
            "--------------------------------------------------------------------------------"
        )
        print(f"CRITICAL ERROR: The data file '{DATA_FILE}' was not found.")
        print(
            f"Please place your 'heart.csv' file in the same directory as this script."
        )
        print(f"Then, run the script again to generate the model file.")
        print(
            "--------------------------------------------------------------------------------"
        )
        exit()

    try:
        # Load the heart disease dataset
        df = pd.read_csv(DATA_FILE)

        # Assume the 'target' column is the prediction target.
        # Adjust this if your target column has a different name.
        X = df.drop("target", axis=1)
        y = df["target"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train a Logistic Regression model
        model_pipeline = LogisticRegression(max_iter=1000)
        model_pipeline.fit(X_train, y_train)

        # Save the trained model to a file
        joblib.dump(model_pipeline, MODEL_FILE)
        print(f"Model trained and saved successfully as '{MODEL_FILE}'.")

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        print(
            "Please check your 'heart.csv' file to ensure it's a valid CSV and has the correct columns."
        )
        exit()

# --- Flask App ---
app = Flask(__name__)

# Load the saved model pipeline
try:
    model = joblib.load(MODEL_FILE)
    print(f"Model loaded successfully from '{MODEL_FILE}'.")
except FileNotFoundError:
    print(
        f"Error: The model file '{MODEL_FILE}' was not found. Please place 'heart.csv' in the directory and run the script again to generate it."
    )
    exit()


@app.route("/")
def home():
    """Home route for the API."""
    return "Heart Disease Prediction API"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.
    Expects a JSON payload with features for prediction.
    """
    try:
        data = request.get_json(force=True)
        # Create a DataFrame from the input JSON.
        # Ensure the keys in the JSON match the feature names the model was trained on.
        # This is crucial for the model to work correctly.
        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)
        prediction_prob = model.predict_proba(input_df)[:, 1]

        return jsonify(
            {"prediction": int(prediction[0]), "probability": float(prediction_prob[0])}
        )
    except Exception as e:
        # Provide a helpful error message if the input data is malformed
        return jsonify({"error": f"Invalid input data: {e}"}), 400


if __name__ == "__main__":
    # The host='0.0.0.0' makes the app accessible on your local network
    app.run(debug=True, host="0.0.0.0")
