# preprocessing.py

import pandas as pd
import pickle

# Load feature names used during training
with open("feature_names.pkl", "rb") as f:
    FEATURE_NAMES = pickle.load(f)

def preprocess_input(data):
    df = pd.DataFrame([data])

    stop_map = {
        'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4
    }
    df["Total_Stops"] = stop_map.get(data["Total_Stops"], 0)

    # Ensure matching column names from training
    df["Journey_Day"] = int(data["Journey_Day"])
    df["Journey_Month"] = int(data["Journey_Month"])
    df["Dep_hour"] = int(data["Dep_Hour"])
    df["Dep_Min"] = int(data["Dep_Minute"])
    df["Arrival_hour"] = int(data["Arrival_Hour"])
    df["Arrival_minute"] = int(data["Arrival_Minute"])
    df["Duration_hours"] = int(data["Duration_hours"])
    df["Duration_mins"] = int(data["Duration_mins"])

    # One-hot encoding manually (all that were used in training)
    airlines = ['Jet Airways', 'IndiGo', 'Air India', 'SpiceJet', 'GoAir',
                'Multiple carriers', 'Multiple carriers Premium economy',
                'Trujet', 'Vistara', 'Vistara Premium economy', 'Jet Airways Business']
    sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
    destinations = ['Cochin', 'Hyderabad', 'Delhi', 'Kolkata', 'New_Delhi']

    for col in airlines:
        df[f'Airline_{col}'] = 1 if data["Airline"] == col else 0

    for col in sources:
        df[f'Source_{col}'] = 1 if data["Source"] == col else 0

    for col in destinations:
        df[f'Destination_{col}'] = 1 if data["Destination"] == col else 0

    # Drop original raw fields if present
    df.drop(columns=["Airline", "Source", "Destination"], errors='ignore', inplace=True)

    # Ensure all expected features are present and in correct order
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0  # add missing dummy columns with 0

    df = df[FEATURE_NAMES]  # reorder columns to match training

    return df
