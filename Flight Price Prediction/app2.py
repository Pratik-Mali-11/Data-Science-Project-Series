# app.py

import streamlit as st
import pickle
from preprocessing import preprocess_input

# Load model
model = pickle.load(open("xgb_model.pkl", "rb"))

st.title("✈️ Flight Price Prediction App")

st.sidebar.header("Enter Flight Details")

# User Inputs
airline = st.sidebar.selectbox("Airline", ['Jet Airways', 'IndiGo', 'Air India', 'SpiceJet'])
source = st.sidebar.selectbox("Source City", ['Delhi', 'Kolkata', 'Mumbai'])
destination = st.sidebar.selectbox("Destination City", ['Cochin', 'Hyderabad'])
stops = st.sidebar.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops'])

journey_day = st.sidebar.number_input("Journey Day", 1, 31, 15)
journey_month = st.sidebar.number_input("Journey Month", 1, 12, 6)

dep_hour = st.sidebar.number_input("Departure Hour", 0, 23, 6)
dep_minute = st.sidebar.number_input("Departure Minute", 0, 59, 0)
arrival_hour = st.sidebar.number_input("Arrival Hour", 0, 23, 9)
arrival_minute = st.sidebar.number_input("Arrival Minute", 0, 59, 30)

duration_hr = st.sidebar.number_input("Duration Hours", 0, 30, 2)
duration_min = st.sidebar.number_input("Duration Minutes", 0, 59, 30)

# Prepare input dict
user_input = {
    "Airline": airline,
    "Source": source,
    "Destination": destination,
    "Total_Stops": stops,
    "Journey_Day": journey_day,
    "Journey_Month": journey_month,
    "Dep_Hour": dep_hour,
    "Dep_Minute": dep_minute,
    "Arrival_Hour": arrival_hour,
    "Arrival_Minute": arrival_minute,
    "Duration_hours": duration_hr,
    "Duration_mins": duration_min,
}


if st.sidebar.button("Predict Price"):
    try:
        processed_input = preprocess_input(user_input)
        prediction = model.predict(processed_input)[0]
        st.success(f"Estimated Flight Price: ₹{round(prediction, 2)}")
    except Exception as e:
        st.error(f"Error: {e}")
