import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session
from database.db_connection import engine, SessionLocal
from database.models import Base, RawPatient, CleanedPatient

RAW_PATH = Path("data/raw/healthcare_dataset.csv")
CLEAN_PATH = Path("data/processed/cleaned_data.csv")

def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables created!")

def load_raw_data():
    print("Loading raw data into database...")
    df = pd.read_csv(RAW_PATH)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df["date_of_admission"] = pd.to_datetime(df["date_of_admission"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    db: Session = SessionLocal()
    try:
        existing = db.query(RawPatient).count()
        if existing > 0:
            print(f"Raw table already has {existing} records. Skipping.")
            return

        records = []
        for _, row in df.iterrows():
            records.append(RawPatient(
                name=row.get("name"),
                age=row.get("age"),
                gender=row.get("gender"),
                blood_type=row.get("blood_type"),
                medical_condition=row.get("medical_condition"),
                date_of_admission=row.get("date_of_admission"),
                doctor=row.get("doctor"),
                hospital=row.get("hospital"),
                insurance_provider=row.get("insurance_provider"),
                billing_amount=row.get("billing_amount"),
                room_number=row.get("room_number"),
                admission_type=row.get("admission_type"),
                discharge_date=row.get("discharge_date"),
                medication=row.get("medication"),
                test_results=row.get("test_results")
            ))

        db.bulk_save_objects(records)
        db.commit()
        print(f"Loaded {len(records)} raw records into database!")
    except Exception as e:
        db.rollback()
        print(f"Error loading raw data: {e}")
    finally:
        db.close()

def load_cleaned_data():
    print(" Loading cleaned data into database...")
    df = pd.read_csv(CLEAN_PATH)

    db: Session = SessionLocal()
    try:
        existing = db.query(CleanedPatient).count()
        if existing > 0:
            print(f"Cleaned table already has {existing} records. Skipping.")
            return

        records = []
        for _, row in df.iterrows():
            records.append(CleanedPatient(
                age=row.get("age"),
                gender=row.get("gender"),
                blood_type=row.get("blood_type"),
                medical_condition=row.get("medical_condition"),
                insurance_provider=row.get("insurance_provider"),
                billing_amount=row.get("billing_amount"),
                admission_type=row.get("admission_type"),
                medication=row.get("medication"),
                length_of_stay=row.get("length_of_stay"),
                test_results=row.get("test_results")
            ))

        db.bulk_save_objects(records)
        db.commit()
        print(f"Loaded {len(records)} cleaned records into database!")
    except Exception as e:
        db.rollback()
        print(f" Error loading cleaned data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_tables()
    load_raw_data()
    load_cleaned_data()
    print("\nAll data loaded successfully!")