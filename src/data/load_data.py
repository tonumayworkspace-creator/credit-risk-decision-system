# src/data/load_data.py

import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset with balanced sampling to avoid single-class problem
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(file_path, low_memory=False)

    # 🔥 Balanced sampling (CRITICAL FIX)
    if "loan_status" in df.columns:

        good_loans = df[df["loan_status"] == "Fully Paid"]
        bad_loans = df[df["loan_status"].isin(
            ["Charged Off", "Default", "Late (31-120 days)"]
        )]

        # Ensure enough data exists
        n_samples = min(len(good_loans), len(bad_loans), 10000)

        good_sample = good_loans.sample(n=n_samples, random_state=42)
        bad_sample = bad_loans.sample(n=n_samples, random_state=42)

        df = pd.concat([good_sample, bad_sample]).sample(frac=1, random_state=42)

    return df


if __name__ == "__main__":
    df = load_data("data/loan_data.csv")
    print("Loaded shape:", df.shape)
    print(df["loan_status"].value_counts())