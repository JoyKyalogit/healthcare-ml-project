import kagglehub
import shutil
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    print("⬇️ Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("prasad22/healthcare-dataset")
    print(f"✅ Downloaded to: {path}")
    for file in Path(path).glob("*.csv"):
        dest = RAW_DATA_DIR / file.name
        shutil.copy(file, dest)
        print(f"✅ Copied to {dest}")
    return str(RAW_DATA_DIR / "healthcare_dataset.csv")

if __name__ == "__main__":
    path = download_dataset()
    print(f"✅ Dataset ready at: {path}")