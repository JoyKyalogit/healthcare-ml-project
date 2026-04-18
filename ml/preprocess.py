import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "age", "gender", "blood_type", "medical_condition",
    "insurance_provider", "billing_amount",
    "admission_type", "medication", "length_of_stay"
]

CATEGORICAL_COLUMNS = [
    "gender", "blood_type", "medical_condition",
    "insurance_provider", "admission_type", "medication"
]

TARGET_COLUMN = "test_results"

def load_data_from_db():
    from database.db_connection import engine
    print("Loading data from database...")
    df = pd.read_sql("SELECT * FROM cleaned_patients", engine)
    print(f"Loaded {len(df)} records")
    return df

def preprocess_features(df: pd.DataFrame, fit: bool = True):
    print("Preprocessing features...")
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df = df.dropna()

    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        if fit:
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, MODELS_DIR / f"encoder_{col}.joblib")
        else:
            le = joblib.load(MODELS_DIR / f"encoder_{col}.joblib")
            df[col] = le.transform(df[col].astype(str))

    target_encoder = LabelEncoder()
    if fit:
        df[TARGET_COLUMN] = target_encoder.fit_transform(df[TARGET_COLUMN])
        joblib.dump(target_encoder, MODELS_DIR / "encoder_target.joblib")
        print(f"Target classes: {list(target_encoder.classes_)}")
    else:
        target_encoder = joblib.load(MODELS_DIR / "encoder_target.joblib")
        df[TARGET_COLUMN] = target_encoder.transform(df[TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    print(f"Features shape: {X.shape}")
    return X, y

if __name__ == "__main__":
    df = load_data_from_db()
    X, y = preprocess_features(df, fit=True)
    print(f"Done! X: {X.shape}, y: {y.shape}")