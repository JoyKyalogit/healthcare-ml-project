import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/healthcare_dataset.csv")
CLEAN_PATH = Path("data/processed/cleaned_data.csv")

def clean_data():
    print("🧹 Cleaning data...")
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    print(f"📊 Raw shape: {df.shape}")

    df = df.drop_duplicates()
    df = df.drop(columns=["Name", "Doctor", "Hospital", "Room Number"], errors="ignore")
    df = df.dropna()

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    df["date_of_admission"] = pd.to_datetime(df["date_of_admission"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])
    df["length_of_stay"] = (df["discharge_date"] - df["date_of_admission"]).dt.days
    df = df.drop(columns=["date_of_admission", "discharge_date"])

    cat_cols = ["gender", "blood_type", "medical_condition",
                "insurance_provider", "admission_type", "medication", "test_results"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    valid = ["Normal", "Abnormal", "Inconclusive"]
    df = df[df["test_results"].isin(valid)]

    df.to_csv(CLEAN_PATH, index=False)
    print(f"✅ Cleaned data saved to: {CLEAN_PATH}")
    print(f"📊 Final shape: {df.shape}")
    print(f"\n🎯 Target distribution:\n{df['test_results'].value_counts()}")
    return df

if __name__ == "__main__":
    clean_data()