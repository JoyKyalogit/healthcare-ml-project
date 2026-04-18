import joblib
from pathlib import Path

MODELS_DIR = Path("models")

def load_model():
    path = MODELS_DIR / "xgboost_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            " Model not found. Run: uv run python ml/train.py"
        )
    return joblib.load(path)

def load_encoder(col: str):
    path = MODELS_DIR / f"encoder_{col}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Encoder not found: {path}")
    return joblib.load(path)