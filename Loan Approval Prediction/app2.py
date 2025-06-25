import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature names
model = joblib.load('loan_approval_rf_model.pkl')
feature_names = joblib.load('feature_columns.pkl')

# Streamlit config
st.set_page_config(page_title="Loan Approval Dashboard", layout="centered")
st.title("🏦 Loan Approval Prediction App")

st.markdown("Upload your dataset and choose a Loan ID to predict approval status.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload CSV file (with Loan_ID column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Loan_ID' not in df.columns:
        st.error("❌ Please make sure your file includes a 'Loan_ID' column.")
    else:
        st.success("✅ File uploaded successfully!")
        
        selected_id = st.selectbox("Select Loan ID", df['Loan_ID'].unique())
        row = df[df['Loan_ID'] == selected_id].iloc[0]

        # --- Preprocessing same as training ---
        applicant_income = row['ApplicantIncome']
        coapplicant_income = row['CoapplicantIncome']
        total_income = applicant_income + coapplicant_income
        loan_amount_log = np.log(row['LoanAmount'] + 1) if row['LoanAmount'] > 0 else 0

        # Binary encodings
        gender = 1 if row['Gender'] == "Male" else 0
        married = 1 if row['Married'] == "Yes" else 0
        education = 1 if row['Education'] == "Graduate" else 0
        self_employed = 1 if row['Self_Employed'] == "Yes" else 0
        credit_history = 1.0 if row['Credit_History'] == 1.0 else 0.0

        # One-hot encodings
        dep_1 = 1 if str(row['Dependents']) == "1" else 0
        dep_2 = 1 if str(row['Dependents']) == "2" else 0
        dep_3_plus = 1 if str(row['Dependents']) in ["3+", "3"] else 0

        area = row['Property_Area']
        area_semiurban = 1 if area == "Semiurban" else 0
        area_urban = 1 if area == "Urban" else 0

        # Optional: Income Category
        if total_income < 2500:
            income_low, income_medium, income_high, income_very_high = 1, 0, 0, 0
        elif 2500 <= total_income < 4000:
            income_low, income_medium, income_high, income_very_high = 0, 1, 0, 0
        elif 4000 <= total_income < 6000:
            income_low, income_medium, income_high, income_very_high = 0, 0, 1, 0
        else:
            income_low, income_medium, income_high, income_very_high = 0, 0, 0, 1

        # Build input dict
        input_data = {
            'Gender': gender,
            'Married': married,
            'Education': education,
            'Self_Employed': self_employed,
            'Loan_Amount_Term': row['Loan_Amount_Term'],
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

        # Ensure column order
        input_df = pd.DataFrame([input_data], columns=feature_names).fillna(0)

        # Prediction
        prediction = model.predict(input_df)[0]

        # Show prediction
        st.subheader("🔍 Prediction for Loan ID: " + selected_id)
        if prediction == 1:
            st.success("✅ Loan will likely be Approved!")
        else:
            st.error("❌ Loan will likely be Rejected.")

        # Optional: Show raw row data
        with st.expander("🔎 View Applicant Data"):
            st.dataframe(row)
