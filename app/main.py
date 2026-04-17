from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PatientInput, PredictionOutput
from app.model_loader import load_model, load_encoder
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Healthcare Test Result Prediction API",
    description="Predicts patient test results as Normal, Abnormal, or Inconclusive",
    version="1.0.0"
)

# Allow frontend to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend files
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

# ✅ load model ONCE (performance + stability fix)
model = load_model("best_model")


@app.get("/")
def root():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    try:
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

        # safer encoding (handles unseen values properly)
        for col in CATEGORICAL_COLUMNS:
            le = load_encoder(col)

            df[col] = df[col].astype(str).str.strip()

            df[col] = df[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_
                else le.transform([le.classes_[0]])[0]
            )

        df = df[FEATURE_COLUMNS]

        prediction = model.predict(df)[0]

        # FIXED: use loader instead of direct joblib access
        target_encoder = load_encoder("target")
        predicted_label = target_encoder.inverse_transform([prediction])[0]

        return PredictionOutput(
            predicted_test_result=predicted_label,
            model_used="best_model"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))