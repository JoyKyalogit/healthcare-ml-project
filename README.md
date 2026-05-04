# Healthcare ML Project

This project predicts a patient's likely test result (`Normal`, `Abnormal`, or `Inconclusive`) from healthcare profile data.  
It includes an end-to-end ML pipeline (ingestion, cleaning, training, evaluation) and a FastAPI app for live predictions.

## Live Demo

[https://healthcare-ml-project-o600.onrender.com](https://healthcare-ml-project-o600.onrender.com)

## What the project does

- Downloads healthcare data from Kaggle (`prasad22/healthcare-dataset`).
- Stores raw data in PostgreSQL.
- Cleans and transforms records for modeling.
- Trains and compares models, including XGBoost.
- Evaluates model quality with standard classification metrics.
- Saves trained artifacts (`joblib`) and serves predictions through FastAPI.
- Supports scheduled retraining via APScheduler.

## Tech Stack

- **Backend/API:** FastAPI, Uvicorn
- **ML/Data:** pandas, scikit-learn, XGBoost, joblib
- **Database:** PostgreSQL, SQLAlchemy
- **Scheduling:** APScheduler
- **Deployment:** Render

## Project Structure

- `app/` - FastAPI application, schemas, model loading
- `database/` - database connection and models
- `ml/` - preprocessing, training, evaluation, prediction logic
- `scripts/` - ingestion/cleaning/pipeline/scheduler scripts
- `frontend/` - simple web UI served by FastAPI
- `models/` - trained model and encoder artifacts
- `render.yaml` - Render deployment configuration

## Local Setup

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+psycopg2://postgres:your_password@localhost:5432/healthcare_db
```

### 3) Run the API

```powershell
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000` (frontend)
- `http://127.0.0.1:8000/docs` (Swagger docs)

## Run the ML Pipeline

Use the project scripts to:

1. Ingest data
2. Clean/transform data
3. Train/evaluate models
4. Save the best model

Example:

```powershell
python scripts/ingest.py
python scripts/clean.py
python ml/train.py
python ml/evaluate.py
```

## API Endpoint

### `POST /predict`

Sample request:

```json
{
  "Age": 45,
  "Gender": "Male",
  "Blood Type": "O+",
  "Medical Condition": "Diabetes",
  "Billing Amount": 2000.5,
  "Admission Type": "Emergency",
  "Insurance Provider": "Cigna",
  "Medication": "Aspirin",
  "Length of Stay": 7
}
```

Sample response:

```json
{
  "predicted_test_result": "Abnormal",
  "confidence": 92.41,
  "model_used": "xgboost"
}
```

## Deployment

The project is configured for Render using `render.yaml`.

- Live app URL: [https://healthcare-ml-project-o600.onrender.com](https://healthcare-ml-project-o600.onrender.com)
