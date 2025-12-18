# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:14:13 2025

@author: PRO BEN
"""
import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegressio

# 1. Load the model and data
@st.cache_resource
def load_model():
    with open('co2_model.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

model = load_model()
df = load_data()

# 2. Web App Interface
st.title("🚗 CO2 Emission Predictor")
st.write("Enter the car details below to predict its CO2 emission.")

# Create input widgets
col1, col2 = st.columns(2)

with col1:
    car_brand = st.selectbox("Select Car Brand", sorted(df['Car'].unique()))
    volume = st.number_input("Engine Volume (cc)", min_value=500, max_value=5000, value=1500)

with col2:
    # Filter models based on the selected car brand
    available_models = sorted(df[df['Car'] == car_brand]['Model'].unique())
    car_model = st.selectbox("Select Model", available_models)
    weight = st.number_input("Weight (kg)", min_value=500, max_value=3000, value=1200)

# 3. Prediction Logic
if st.button("Predict CO2"):
    # Create a dataframe for the input (matching the training format)
    input_data = pd.DataFrame({
        'Car': [car_brand],
        'Model': [car_model],
        'Volume': [volume],
        'Weight': [weight]
    })
    
    prediction = model.predict(input_data)
    

    st.success(f"The estimated CO2 emission is **{prediction[0]:.2f} g/km**")
