from sqlalchemy import Column, Integer, String, Float, Date, DateTime
from sqlalchemy.sql import func
from database.db_connection import Base

class RawPatient(Base):
    __tablename__ = "raw_patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    blood_type = Column(String)
    medical_condition = Column(String)
    date_of_admission = Column(Date)
    doctor = Column(String)
    hospital = Column(String)
    insurance_provider = Column(String)
    billing_amount = Column(Float)
    room_number = Column(Integer)
    admission_type = Column(String)
    discharge_date = Column(Date)
    medication = Column(String)
    test_results = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class CleanedPatient(Base):
    __tablename__ = "cleaned_patients"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    gender = Column(String)
    blood_type = Column(String)
    medical_condition = Column(String)
    insurance_provider = Column(String)
    billing_amount = Column(Float)
    admission_type = Column(String)
    medication = Column(String)
    length_of_stay = Column(Integer)
    test_results = Column(String)
    created_at = Column(DateTime, server_default=func.now())