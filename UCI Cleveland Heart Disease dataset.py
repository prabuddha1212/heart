import pandas as pd

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Define column names as per dataset description
columns = [
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
    "target",
]

# Load dataset into a DataFrame, specifying '?' as missing values
heart_df = pd.read_csv(url, header=None, names=columns, na_values="?")

# Display first few rows
print(heart_df.head())
# Check for missing/null values in each column
print(heart_df.isnull().sum())
heart_df_clean = heart_df.dropna()
print(heart_df_clean.isnull().sum())
# Fill missing values in 'ca' with mode (most frequent value)
heart_df["ca"].fillna(heart_df["ca"].mode()[0], inplace=True)

# Fill missing values in 'thal' with mode
heart_df["thal"].fillna(heart_df["thal"].mode()[0], inplace=True)
print(heart_df.isnull().sum())
# Convert to categorical dtype
categorical_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
for col in categorical_columns:
    heart_df[col] = heart_df[col].astype("category")

# Example of one-hot encoding
heart_df_encoded = pd.get_dummies(heart_df, columns=categorical_columns)
print(heart_df_encoded.head())
# Display data types to confirm changes
