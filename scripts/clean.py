import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/healthcare_dataset.csv")
CLEAN_PATH = Path("data/processed/cleaned_data.csv")

def clean_data():
    print(" Cleaning data...")
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(RAW_PATH)
    print(f" Raw data shape: {df.shape}")
    print(f" Columns: {list(df.columns)}")

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    # Drop irrelevant columns
    df = df.drop(columns=["Name", "Doctor", "Hospital", "Room Number"], errors="ignore")

    # Handle missing values
    df = df.dropna()
    print(f" After dropping nulls: {df.shape}")

    # Standardize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Convert date columns
    df["date_of_admission"] = pd.to_datetime(df["date_of_admission"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    # Calculate length of stay
    df["length_of_stay"] = (df["discharge_date"] - df["date_of_admission"]).dt.days

    # Drop date columns after extracting length of stay
    df = df.drop(columns=["date_of_admission", "discharge_date"])

    # Standardize categorical values
    cat_cols = ["gender", "blood_type", "medical_condition",
                "insurance_provider", "admission_type", "medication", "test_results"]
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Validate test_results values
    valid_results = ["Normal", "Abnormal", "Inconclusive"]
    df = df[df["test_results"].isin(valid_results)]
    print(f" After filtering valid test results: {df.shape}")

    # Save cleaned data
    df.to_csv(CLEAN_PATH, index=False)
    print(f" Cleaned data saved to: {CLEAN_PATH}")
    print(f"\n Test Results Distribution:")
    print(df["test_results"].value_counts())

    return df

if __name__ == "__main__":
    df = clean_data()
    print(f"\n Cleaning complete! Shape: {df.shape}")
    print(df.head())