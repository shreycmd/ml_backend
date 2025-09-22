from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for development: allow all origins (no credentials with "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")


# Request schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API 0.1"}


@app.post("/predict")
def predict(data: HeartData):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame(
        {
            "age": [data.age],
            "sex": [data.sex],
            "cp": [data.cp],
            "trestbps": [data.trestbps],
            "chol": [data.chol],
            "fbs": [data.fbs],
            "restecg": [data.restecg],
            "thalach": [data.thalach],
            "exang": [data.exang],
            "oldpeak": [data.oldpeak],
            "slope": [data.slope],
            "ca": [data.ca],
            "thal": [data.thal],
        }
    )

    # Define categorical and continuous variables (same as in training)
    cate_val = ["cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    cont_val = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Apply one-hot encoding to categorical variables (same as training)
    input_processed = pd.get_dummies(input_data, columns=cate_val, drop_first=True)

    # Ensure all expected columns exist (add missing dummy columns with 0s)
    expected_columns = [
        "age",
        "sex",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "cp_1",
        "cp_2",
        "cp_3",
        "fbs_1",
        "restecg_1",
        "restecg_2",
        "exang_1",
        "slope_1",
        "slope_2",
        "ca_1",
        "ca_2",
        "ca_3",
        "ca_4",
        "thal_1",
        "thal_2",
        "thal_3",
    ]

    for col in expected_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Reorder columns to match training data
    input_processed = input_processed[expected_columns]

    # Apply scaling to continuous variables
    input_processed[cont_val] = scaler.transform(input_processed[cont_val])

    # Convert to numpy array for prediction
    features = input_processed.values

    prediction = model.predict(features)[0]
    # Return human-friendly string to avoid numpy serialization issues
    return int(prediction)
