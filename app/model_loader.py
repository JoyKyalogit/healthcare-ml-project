import joblib
from pathlib import Path
import gdown

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# =========================
# GOOGLE DRIVE FILE IDs
# =========================

MODEL_FILE_ID = "1toVS9TINu4gZE4A7oyU37pekdr0aJY5g"

ENCODER_FILE_IDS = {
    "gender": "1cd34Rslc8w0nGw1MKilrOc3Rs-_eFRiO",
    "blood_type": "1ATaoM0zxjMx4XKplFaCv6IaoiijBkBKy",
    "medical_condition": "188PP2P5aIAg5HuLYtCVQLqdMDbmM6jYi",
    "insurance_provider": "1m4nOMvOgxLNz8ezdtXhUxeT6TEE-21e3",
    "admission_type": "19wnxQZNUjfNzeqD_pytgWDbY2ZzEBpr0",
    "medication": "1cAMZkRfRU8R13yD94CK1YGh1JIZUmMzL",
    "target": "1soR10v4SVqh4VpkbeBKzQLjjPYj40ATA"
}

# =========================
#  DOWNLOAD HELPER
# =========================

def _download(file_id: str, output_path: Path):
    if not output_path.exists():
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            str(output_path),
            quiet=False
        )

# =========================
# MODEL LOADER
# =========================

def load_model(model_name: str = "best_model"):
    model_path = MODELS_DIR / f"{model_name}.joblib"

    _download(MODEL_FILE_ID, model_path)

    return joblib.load(model_path)

# =========================
#  ENCODER LOADER
# =========================

def load_encoder(col: str):
    encoder_path = MODELS_DIR / f"encoder_{col}.joblib"

    file_id = ENCODER_FILE_IDS.get(col)
    if not file_id:
        raise ValueError(f"No Google Drive ID for encoder: {col}")

    _download(file_id, encoder_path)

    return joblib.load(encoder_path)