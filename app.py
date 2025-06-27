import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---- Load model and data ----
model = joblib.load("model/xgb_model.pkl")
df = pd.read_csv("data/processed_df.csv", parse_dates=["Date"])

# Feature list used in training
features = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "StoreType", "Assortment",
    "Month", "Year", "Day", "Weekofyear", "Quarter", "IsPromoMonth",
    "CompOpenMonthsAgo", "Promo2OpenWeeksAgo", "Sales_lag1", "Sales_lag7", "Sales_roll_mean_7"
]

# ---- Streamlit UI ----
st.set_page_config(page_title="📈 Rossmann Sales Predictor", layout="centered")
st.title("🏪 Rossmann Store Sales Forecast")

# Store and Date selection
store_ids = df["Store"].unique()
selected_store = st.selectbox("Select Store ID", sorted(store_ids))

# Filter data for this store
# Filter valid store data with no NaN in lag features
store_df = df[(df["Store"] == selected_store) & (~df["Sales_lag1"].isna())].sort_values("Date")

# Get valid date range
min_valid_date = store_df["Date"].min()
max_valid_date = store_df["Date"].max()

# Calendar-style date picker
selected_date = st.date_input("Select a Date to Predict", 
                              value=max_valid_date, 
                              min_value=min_valid_date, 
                              max_value=max_valid_date)

# ✅ FIX: Convert to datetime64[ns] to match DataFrame
selected_date = pd.to_datetime(selected_date)

# Continue safely
input_row = store_df[store_df["Date"] == selected_date]

if input_row.empty:
    st.error("No valid feature data for selected date. Try another.")
else:
    X = input_row[features]
    predicted_sales = model.predict(X)[0]
    st.success(f"📊 Predicted Sales on {selected_date.date()} for Store {selected_store}: ₹{predicted_sales:.2f}")

    with st.expander("📉 Show Last 30 Days Sales Trend"):
        trend_data = store_df[store_df["Date"] < selected_date].tail(30)
        st.line_chart(trend_data.set_index("Date")["Sales"])
