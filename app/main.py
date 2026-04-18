from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PatientInput, PredictionOutput
from app.model_loader import load_model, load_encoder
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Healthcare Test Result Prediction API",
    description="Predicts patient test results as Normal, Abnormal, or Inconclusive",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

CATEGORICAL_COLUMNS = [
    "gender", "blood_type", "medical_condition",
    "insurance_provider", "admission_type", "medication"
]

FEATURE_COLUMNS = [
    "age", "gender", "blood_type", "medical_condition",
    "insurance_provider", "billing_amount",
    "admission_type", "medication", "length_of_stay"
]

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgboost"}

@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    try:
        model = load_model()

        input_dict = {
            "age": patient.Age,
            "gender": patient.Gender,
            "blood_type": patient.Blood_Type,
            "medical_condition": patient.Medical_Condition,
            "insurance_provider": patient.Insurance_Provider,
            "billing_amount": patient.Billing_Amount,
            "admission_type": patient.Admission_Type,
            "medication": patient.Medication,
            "length_of_stay": patient.Length_of_Stay
        }

        df = pd.DataFrame([input_dict])

        for col in CATEGORICAL_COLUMNS:
            le = load_encoder(col)
            df[col] = df[col].str.strip().str.title()
            df[col] = le.transform(df[col].astype(str))

        df = df[FEATURE_COLUMNS]

        # Get prediction and confidence
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        confidence = round(float(np.max(probabilities) * 100), 2)

        target_encoder = joblib.load(Path("models") / "encoder_target.joblib")
        predicted_label = target_encoder.inverse_transform([prediction])[0]

        return PredictionOutput(
            predicted_test_result=predicted_label,
            confidence=confidence,
            model_used="xgboost"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))