# src/data/preprocess.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:

    if "loan_status" not in df.columns:
        return df

    bad_status = ["Charged Off", "Default", "Late (31-120 days)"]

    df["target"] = df["loan_status"].apply(
        lambda x: 1 if x in bad_status else 0
    )

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove future / leakage features (VERY IMPORTANT)
    """

    leakage_cols = [
        "loan_status",
        "recoveries",
        "collection_recovery_fee",
        "total_rec_prncp",
        "total_rec_int",
        "last_pymnt_amnt",
        "last_pymnt_d",
        "next_pymnt_d"
    ]

    existing_cols = [col for col in leakage_cols if col in df.columns]
    df = df.drop(columns=existing_cols)

    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:

    drop_cols = ["id", "member_id", "url", "desc"]

    existing_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_cols)

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    threshold = 0.5
    missing_ratio = df.isnull().mean()

    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    df = df.drop(columns=cols_to_drop)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df = create_target(df)
    df = drop_leakage_columns(df)  # 🔥 MOST IMPORTANT
    df = drop_unnecessary_columns(df)
    df = handle_missing_values(df)

    return df


if __name__ == "__main__":
    from src.data.load_data import load_data

    df = load_data("data/loan_data.csv")

    df = preprocess_data(df)

    print("Preprocessing Completed")
    print(df.head())