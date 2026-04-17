import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ml.preprocess import load_data_from_db, preprocess_features
from ml.evaluate import evaluate_model

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def train_models():
    print("Starting model training...")

    # Load and preprocess data
    df = load_data_from_db()
    X, y = preprocess_features(df, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    results = {}

    # ── Model 1: Logistic Regression ──
    print("\n Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results["logistic_regression"] = lr_metrics
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib")
    print("Logistic Regression saved!")

    # ── Model 2: Random Forest ──
    print("\n Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results["random_forest"] = rf_metrics
    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")
    print(" Random Forest saved!")

    # ── Model 3: XGBoost ──
    print("\n Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results["xgboost"] = xgb_metrics
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")
    print("XGBoost saved!")

    # ── Pick Best Model ──
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = {
        "logistic_regression": lr_model,
        "random_forest": rf_model,
        "xgboost": xgb_model
    }[best_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print(f"\n Best model: {best_name} with accuracy: {results[best_name]['accuracy']:.4f}")
    print(" Best model saved as models/best_model.joblib")

    return results

if __name__ == "__main__":
    results = train_models()
    print("\n Training complete!")