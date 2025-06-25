import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and feature names
model = joblib.load('loan_approval_rf_model.pkl')
feature_names = joblib.load('feature_columns.pkl')

# Streamlit App Title
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")

st.markdown("""
Use the form below to check if a loan application will likely be approved.
""")

# ---- User Input Form ----
with st.form("loan_form"):
    st.subheader("📋 Applicant Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    credit_history = st.selectbox("Credit History", ["Has Credit History", "No Credit History"])
    
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_term = st.selectbox("Loan Term (in months)", [360.0, 180.0, 120.0, 240.0, 300.0, 60.0])

    income_category = st.selectbox("Income Category", ["Low", "Medium", "High", "Very High"])

    submitted = st.form_submit_button("Predict Approval")

# ---- Preprocessing & Prediction ----
if submitted:
    # Feature engineering
    total_income = applicant_income + coapplicant_income
    loan_amount_log = np.log(loan_amount + 1)

    # Binary encoding
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1.0 if credit_history == "Has Credit History" else 0.0

    # One-hot encoding for Dependents
    dep_1 = 1 if dependents == "1" else 0
    dep_2 = 1 if dependents == "2" else 0
    dep_3_plus = 1 if dependents == "3+" else 0

    # One-hot encoding for Property Area
    area_semiurban = 1 if property_area == "Semiurban" else 0
    area_urban = 1 if property_area == "Urban" else 0

    # One-hot encoding for Income Category
    income_low = 1 if income_category == "Low" else 0
    income_medium = 1 if income_category == "Medium" else 0
    income_high = 1 if income_category == "High" else 0
    income_very_high = 1 if income_category == "Very High" else 0

    # Create dictionary for all possible features
    input_data = {
        'Gender': gender,
        'Married': married,
        'Education': education,
        'Self_Employed': self_employed,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'LoanAmount_log': loan_amount_log,
        'Total_Income': total_income,
        'Property_Area_Semiurban': area_semiurban,
        'Property_Area_Urban': area_urban,
        'Dependents_1': dep_1,
        'Dependents_2': dep_2,
        'Dependents_3+': dep_3_plus,
        'Income_Category_Low': income_low,
        'Income_Category_Medium': income_medium,
        'Income_Category_High': income_high,
        'Income_Category_Very High': income_very_high
    }

    # Build input DataFrame with all required columns in correct order
    input_df = pd.DataFrame([input_data], columns=feature_names).fillna(0)

    # Make prediction
    prediction = model.predict(input_df)

    # Display result
    st.subheader("🔍 Prediction Result:")
    if prediction[0] == 1:
        st.success("✅ Loan will likely be Approved!")
    else:
        st.error("❌ Loan will likely be Rejected.")

# ---- Optional Static Chart ----
st.markdown("---")
st.subheader("📈 Approval Rate by Property Area (Static Example)")
sample_chart = pd.DataFrame({
    'Property Area': ['Urban', 'Semiurban', 'Rural'],
    'Approval Rate (%)': [65, 80, 55]
}).set_index('Property Area')
st.bar_chart(sample_chart)
