#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# 1) Define the input schema using Pydantic
type_hints = {
    'Age': int,
    'Delay_Minutes': float,
    'Booking_Days_In_Advance': int,
    'Weather_Impact': int,
    'Distance_km': int,
    'IsHoliday': int,
    'Departure_Hour': int,
    'Departure_Weekday': int,
    'Departure_Month': int,
    'Price': float,
    'BaggageAllowance': int,
    'Airline': str,
    'Origin': str,
    'Destination': str,
    'Flight_Status': str,
    'Gender': str,
    'Travel_Purpose': str,
    'Seat_Class': str,
    'MembershipTier': str,
    'Check_in_Method': str
}

class FlightData(BaseModel):
    __annotations__ = type_hints

# 2) Load the pre-trained pipeline
app = FastAPI()
with open('rf_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# 3) Define the prediction endpoint
@app.post("/predict")
def predict(data: FlightData):
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict
    pred_code = int(pipeline.predict(df)[0])
    pred_proba = float(pipeline.predict_proba(df)[0, 1])

    # Map to human-readable label
    label_map = {0: "show", 1: "no show"}
    return {
        "prediction": label_map[pred_code],
    }

# To run: uvicorn fastapi_app:app --reload

