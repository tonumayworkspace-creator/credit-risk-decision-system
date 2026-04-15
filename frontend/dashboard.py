# frontend/dashboard.py

import streamlit as st
import pandas as pd
import requests

st.title("📊 Credit Risk Monitoring Dashboard")

st.write("Batch prediction + analytics")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    results = []

    for _, row in df.iterrows():
        data = {
            "loan_amnt": row.get("loan_amnt", 0),
            "annual_inc": row.get("annual_inc", 0),
            "installment": row.get("installment", 0),
            "int_rate": row.get("int_rate", 0),
            "open_acc": row.get("open_acc", 0),
            "total_acc": row.get("total_acc", 1),
            "revol_bal": row.get("revol_bal", 0)
        }

        try:
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data
            )

            result = res.json()

            results.append({
                "PD": result["probability_of_default"],
                "Expected Loss": result["expected_loss"],
                "Risk": result["risk_level"],
                "Decision": result["decision"]
            })

        except:
            results.append({
                "PD": None,
                "Expected Loss": None,
                "Risk": "Error",
                "Decision": "Error"
            })

    result_df = pd.DataFrame(results)

    st.subheader("Results")
    st.dataframe(result_df)

    st.subheader("Risk Distribution")
    st.bar_chart(result_df["Risk"].value_counts())

    st.subheader("Decision Distribution")
    st.bar_chart(result_df["Decision"].value_counts())

    st.subheader("Average Metrics")
    st.write({
        "Avg PD": result_df["PD"].mean(),
        "Avg Expected Loss": result_df["Expected Loss"].mean()
    })