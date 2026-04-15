# src/features/feature_engineering.py

import sys
import os

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd


def create_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial ratio features
    """

    # Avoid division by zero
    df["annual_inc"] = df["annual_inc"].replace(0, 1)

    # Debt-to-Income Ratio
    if "loan_amnt" in df.columns:
        df["loan_to_income_ratio"] = df["loan_amnt"] / df["annual_inc"]

    # Installment burden
    if "installment" in df.columns:
        df["installment_to_income_ratio"] = df["installment"] / df["annual_inc"]

    return df


def create_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create credit behavior features
    """

    if "revol_bal" in df.columns and "total_acc" in df.columns:
        df["credit_per_account"] = df["revol_bal"] / (df["total_acc"] + 1)

    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["open_to_total_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

    return df


def create_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple risk indicator flags
    """

    # High loan amount flag
    if "loan_amnt" in df.columns:
        df["high_loan_flag"] = (df["loan_amnt"] > df["loan_amnt"].median()).astype(int)

    # High interest rate flag
    if "int_rate" in df.columns:
        df["high_interest_flag"] = (df["int_rate"] > df["int_rate"].median()).astype(int)

    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline
    """

    df = create_financial_ratios(df)
    df = create_credit_features(df)
    df = create_risk_flags(df)

    return df


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data

    df = load_data("data/loan_data.csv")
    df = preprocess_data(df)

    df = feature_engineering_pipeline(df)

    print("Feature Engineering Completed")
    print("Shape:", df.shape)
    print(df.head())