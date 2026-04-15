# src/models/train_model.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.feature_engineering import feature_engineering_pipeline


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    return df


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train_model(X, y):

    print("Target Distribution:\n", y.value_counts())

    if len(set(y)) < 2:
        raise ValueError("Only one class present")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    print("ROC-AUC Score:", roc)
    print("\nClassification Report:\n", report)

    # 🔥 SAVE IMAGE
    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"ROC-AUC: {roc:.3f}")
    plt.savefig("outputs/model_performance.png")

    return model


def save_model(model, X):
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/credit_model.pkl")
    joblib.dump(X.columns.tolist(), "models/model_columns.pkl")


if __name__ == "__main__":
    df = load_data("data/loan_data.csv")
    df = preprocess_data(df)
    df = feature_engineering_pipeline(df)

    df = encode_categorical(df)

    X, y = prepare_data(df)

    model = train_model(X, y)

    save_model(model, X)

    print("\nModel Training Completed")