# notebooks/eda.py

import sys
import os

# ✅ Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load_data import load_data


def basic_eda(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum().sort_values(ascending=False))


def target_analysis(df: pd.DataFrame):
    target_col = "loan_status"

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataset")

    print("\nTarget Distribution:\n", df[target_col].value_counts())

    sns.countplot(x=df[target_col])
    plt.title("Target Distribution")
    plt.xticks(rotation=45)
    plt.show()


def numerical_analysis(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    print("\nNumerical Columns:", num_cols)

    df[num_cols].hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data("data/loan_data.csv")

    print("----- BASIC EDA -----")
    basic_eda(df)

    print("\n----- TARGET ANALYSIS -----")
    target_analysis(df)

    print("\n----- NUMERICAL ANALYSIS -----")
    numerical_analysis(df)