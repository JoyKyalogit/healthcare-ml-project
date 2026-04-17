from fastapi import APIRouter
from app.schemas import PatientInput, PredictionOutput
from ml.predict import predict_single

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    result = predict_single(patient.model_dump(by_alias=False))
    return {"predicted_test_result": result}

@router.get("/health")
def health():
    return {"status": "ok", "message": "Healthcare ML API is running"}