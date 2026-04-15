# api/main.py

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()

model = joblib.load("models/credit_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")


class UserInput(BaseModel):
    loan_amnt: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    installment: float = Field(..., ge=0)
    int_rate: float = Field(..., ge=0)
    open_acc: int = Field(..., ge=0)
    total_acc: int = Field(..., ge=1)
    revol_bal: float = Field(..., ge=0)


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
def predict(user: UserInput):

    try:
        logger.info(f"Incoming request: {user.dict()}")

        df = pd.DataFrame([user.dict()])

        # Feature Engineering
        df["loan_to_income_ratio"] = df["loan_amnt"] / df["annual_inc"]
        df["installment_to_income_ratio"] = df["installment"] / df["annual_inc"]
        df["credit_per_account"] = df["revol_bal"] / (df["total_acc"] + 1)
        df["open_to_total_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

        df["high_loan_flag"] = (df["loan_amnt"] > 10000).astype(int)
        df["high_interest_flag"] = (df["int_rate"] > 12).astype(int)

        # Align columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # Prediction
        probs = model.predict_proba(df)

        if probs.shape[1] == 1:
            prob = 0.0
        else:
            prob = float(probs[0][1])

        # Expected Loss
        loan_amount = user.loan_amnt
        lgd = 0.6
        expected_loss = prob * loan_amount * lgd

        # Risk Level
        if prob < 0.05:
            risk_level = "Low Risk"
        elif prob < 0.15:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # 🔥 UPDATED DECISION LOGIC (DUAL CONDITION)
        if prob > 0.4:
            decision = "Reject"
        elif expected_loss > 2000:
            decision = "Reject"
        else:
            decision = "Approve"

        logger.info(f"PD: {prob}, EL: {expected_loss}, Decision: {decision}")

        return {
            "probability_of_default": prob,
            "expected_loss": expected_loss,
            "risk_level": risk_level,
            "decision": decision
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))