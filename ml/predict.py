import joblib
import pandas as pd
from app.model_loader import get_model, get_encoders, load_model

def predict_single(input_data: dict):
    model = get_model()
    encoders = get_encoders()
    if model is None or encoders is None:
        load_model()
        model = get_model()
        encoders = get_encoders()
    if model is None or encoders is None:
        model = joblib.load("models/best_model.pkl")
        encoders = joblib.load("models/encoders.pkl")

    df = pd.DataFrame([input_data])

    categorical_cols = [
        "gender", "blood_type", "medical_condition",
        "admission_type", "insurance_provider", "medication"
    ]

    for col in categorical_cols:
        le = encoders[col]
        val = str(df[col].iloc[0]).strip().title()
        if val in le.classes_:
            df[col] = le.transform([val])
        else:
            df[col] = 0

    feature_cols = [
        "age", "gender", "blood_type", "medical_condition",
        "billing_amount", "admission_type", "insurance_provider",
        "medication", "length_of_stay"
    ]

    df = df[feature_cols]
    prediction = model.predict(df)[0]
    label = encoders["test_results"].inverse_transform([prediction])[0]
    return label