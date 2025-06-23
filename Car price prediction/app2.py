import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🚗 Used Car Price Predictor")

# Load model and features
model = joblib.load('xgb_car_price_model2.pkl')
feature_cols = joblib.load('feature_columns2.pkl')

# User input
brand = st.selectbox("Brand", ['Ford', 'Hyundai', 'Lexus', 'Audi', 'Other'])
mileage = st.number_input("Mileage (in miles)", min_value=0)
horsepower = st.number_input("Horsepower", min_value=50, max_value=1500)
engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=8.0, step=0.1)
car_age = st.slider("Car Age (Years)", 0, 20)
fuel_type = st.selectbox("Fuel Type", ['Gasoline', 'Hybrid', 'E85 Flex Fuel'])
transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'Other'])
accident = st.selectbox("Any Accident Reported?", ['Yes', 'No'])
clean_title = st.selectbox("Clean Title?", ['Yes', 'No'])

# Preprocess
input_dict = {
    'milage_miles': mileage,
    'horsepower': horsepower,
    'engine_size_L': engine_size,
    'car_age': car_age,
    'accident_reported': 1 if accident == 'Yes' else 0,
    'clean_title': 1 if clean_title == 'Yes' else 0
}

# Add missing one-hot columns
for col in feature_cols:
    if col not in input_dict:
        input_dict[col] = 0

# One-hot encode selection
input_dict[f'brand_{brand}'] = 1 if f'brand_{brand}' in feature_cols else 0
input_dict[f'fuel_type_{fuel_type}'] = 1 if f'fuel_type_{fuel_type}' in feature_cols else 0
input_dict[f'transmission_{transmission}'] = 1 if f'transmission_{transmission}' in feature_cols else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])
input_df = input_df[feature_cols]  # Ensure correct order

# Predict
if st.button("Predict Price"):
    log_price = model.predict(input_df)[0]
    actual_price = np.expm1(log_price)
    st.success(f"💰 Estimated Car Price: **${actual_price:,.2f}**")


st.success(f"💰 Estimated Price: ${actual_price:,.2f}")
st.info("ℹ️ Typical error range: ±$6000 (based on MAE)")

