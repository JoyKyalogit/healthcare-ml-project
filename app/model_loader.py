import joblib
from pathlib import Path

MODELS_DIR = Path("models")

def load_model(model_name: str = "best_model"):
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found at {model_path}. Run ml/train.py first.")
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    return model

def load_encoder(col: str):
    encoder_path = MODELS_DIR / f"encoder_{col}.joblib"
    if not encoder_path.exists():
        raise FileNotFoundError(f"❌ Encoder not found: {encoder_path}")
    return joblib.load(encoder_path)