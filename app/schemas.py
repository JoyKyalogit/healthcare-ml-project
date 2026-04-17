from pydantic import BaseModel
from typing import Optional

class PatientInput(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Billing_Amount: float
    Admission_Type: str
    Insurance_Provider: str
    Medication: str
    Length_of_Stay: Optional[int] = 7

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Gender": "Male",
                "Blood_Type": "O+",
                "Medical_Condition": "Diabetes",
                "Billing_Amount": 2000.5,
                "Admission_Type": "Emergency",
                "Insurance_Provider": "Cigna",
                "Medication": "Aspirin",
                "Length_of_Stay": 5
            }
        }

class PredictionOutput(BaseModel):
    predicted_test_result: str
    model_used: str = "best_model"