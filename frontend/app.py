# frontend/app.py

import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk System", layout="centered")

st.title("💳 Credit Risk Decision System")
st.caption("FinTech Risk Engine: PD + Expected Loss + Decision")

st.divider()

# Inputs
loan_amnt = st.number_input("Loan Amount", 1000)
annual_inc = st.number_input("Annual Income", 1000)
installment = st.number_input("Installment", 0)
int_rate = st.number_input("Interest Rate (%)", 0.0)
open_acc = st.number_input("Open Accounts", 0)
total_acc = st.number_input("Total Accounts", 1)
revol_bal = st.number_input("Revolving Balance", 0)

st.divider()

if st.button("🚀 Predict Risk"):

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

        st.subheader("📊 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Probability of Default", f"{result['probability_of_default']:.2f}")

        with col2:
            st.metric("Expected Loss", f"₹{result['expected_loss']:.0f}")

        # Risk color
        if result["risk_level"] == "Low Risk":
            st.success(f"Risk Level: {result['risk_level']}")
        elif result["risk_level"] == "Medium Risk":
            st.warning(f"Risk Level: {result['risk_level']}")
        else:
            st.error(f"Risk Level: {result['risk_level']}")

        # Decision highlight
        if result["decision"] == "Approve":
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except:
        st.error("⚠️ Backend API not connected")
