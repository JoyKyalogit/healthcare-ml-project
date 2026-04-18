# Healthcare Analytics System

End-to-end healthcare ML pipeline using Kaggle data, PostgreSQL, XGBoost + baseline models, and FastAPI deployment for live predictions.

## What this project does

- Downloads the Kaggle healthcare dataset (`prasad22/healthcare-dataset`).
- Stores raw data in PostgreSQL.
- Cleans and standardizes records for ML readiness.
- Trains and compares:
  - XGBoost (required)
- Evaluates using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Saves the best model with `joblib`.
- Exposes a live API endpoint: `POST /predict`.
- Schedules weekly retraining every Saturday at 12:00 PM (Africa/Nairobi).

## Project structure

- `app/` - FastAPI app and request/response schemas
- `database/` - DB connection, ORM models, SQL helpers
- `ml/` - preprocessing, training, evaluation, prediction
- `scripts/` - ingestion, cleaning, full pipeline, scheduler
- `data/` - local dataset files
- `models/` - saved trained model artifacts
- `render.yaml` - free deployment config for Render

## Setup (PowerShell + UV)

### 1) Create folders in PowerShell

```powershell
mkdir healthcare-ml-project
cd healthcare-ml-project
mkdir app,data,database,ml,models,notebooks,scripts,frontend,tests
```

### 2) Initialize UV environment

```powershell
uv init .
uv sync
```

### 3) Kaggle authentication

1. Generate Kaggle API token from Kaggle account settings.
2. Place `kaggle.json` at:
   - Windows: `C:\Users\<YOUR_USER>\.kaggle\kaggle.json`

Reference: [Kaggle API Authentication](https://www.kaggle.com/docs/api#authentication)

### 4) Configure environment

Create `.env` in project root:

```env
DATABASE_URL=postgresql+psycopg2://postgres:your_password@localhost:5432/healthcare_db
```

## Run the full pipeline

```powershell
uv run pipeline-run
```

This runs:
1. Kaggle ingestion (`patients_raw`)
2. Cleaning/transformation (`patients_cleaned`)
3. Model training + evaluation + model save

## Start weekly retraining scheduler (Saturday 12:00 noon)

```powershell
uv run start-scheduler
```

## Run API locally

```powershell
uv run uvicorn app.main:app --reload
```

Open docs at: `http://127.0.0.1:8000/docs`

## Prediction endpoint

### POST `/predict`

Request:

```json
{
  "Age": 45,
  "Gender": "Male",
  "Blood Type": "O+",
  "Medical Condition": "Diabetes",
  "Billing Amount": 2000.5,
  "Admission Type": "Emergency",
  "Insurance Provider": "Cigna",
  "Medication": "Aspirin"
}
```

Response:

```json
{
  "predicted_test_result": "Abnormal"
}
```

## Deploy on Render (free)

1. Push this repo to GitHub.
2. Create a new Render Web Service from the GitHub repo.
3. Render auto-detects `render.yaml`.
4. Add `DATABASE_URL` in Render environment variables.
5. Deploy and test:
   - `https://<your-render-service>.onrender.com/predict`

## GitHub commands

```powershell
git init
git add .
git commit -m "Build end-to-end healthcare analytics pipeline and FastAPI deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/healthcare-ml-project.git
git push -u origin main
```

