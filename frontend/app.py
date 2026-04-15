# frontend/app.py

import streamlit as st
import requests

st.title("💳 Credit Risk Decision System")

st.write("Enter customer details")

# Inputs
loan_amnt = st.number_input("Loan Amount", 1000)
annual_inc = st.number_input("Annual Income", 1000)
installment = st.number_input("Installment", 0)
int_rate = st.number_input("Interest Rate (%)", 0.0)
open_acc = st.number_input("Open Accounts", 0)
total_acc = st.number_input("Total Accounts", 1)
revol_bal = st.number_input("Revolving Balance", 0)

if st.button("Predict"):

    data = {
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "installment": installment,
        "int_rate": int_rate,
        "open_acc": open_acc,
        "total_acc": total_acc,
        "revol_bal": revol_bal
    }

    try:
        res = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        result = res.json()

        st.success(f"Probability of Default: {result['probability_of_default']:.4f}")
        st.warning(f"Expected Loss: ₹{result['expected_loss']:.2f}")
        st.info(f"Risk Level: {result['risk_level']}")
        st.error(f"Decision: {result['decision']}")

    except:
        st.error("API not running")