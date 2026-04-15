# src/models/predict.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import joblib

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.feature_engineering import feature_engineering_pipeline
from src.models.train_model import encode_categorical


def load_model():
    return joblib.load("models/credit_model.pkl")


def get_prediction(df: pd.DataFrame, model):
    """
    Get probability predictions
    """
    probs = model.predict_proba(df)[:, 1]
    return probs


def apply_decision(probabilities, threshold=0.1):  # ✅ changed here
    """
    Convert probability → decision
    """

    decisions = ["Reject" if p > threshold else "Approve" for p in probabilities]
    return decisions


if __name__ == "__main__":

    df = load_data("data/loan_data.csv")
    df = preprocess_data(df)
    df = feature_engineering_pipeline(df)
    df = encode_categorical(df)

    X = df.drop(columns=["target"])

    model = load_model()

    probs = get_prediction(X, model)

    decisions = apply_decision(probs, threshold=0.1)  # ✅ changed here

    output = pd.DataFrame({
        "Probability_of_Default": probs,
        "Decision": decisions
    })

    print(output.head())