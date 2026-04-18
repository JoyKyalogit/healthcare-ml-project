import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from ml.preprocess import load_data_from_db, preprocess_features
from ml.evaluate import evaluate_model

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def train_model():
    print("Starting XGBoost training...")

    df = load_data_from_db()
    X, y = preprocess_features(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Class distribution:\n{y_train.value_counts()}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "XGBoost")

    joblib.dump(model, MODELS_DIR / "xgboost_model.joblib")
    print(f"\nModel saved to models/xgboost_model.joblib")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    return metrics

if __name__ == "__main__":
    train_model()