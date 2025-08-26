import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Load model
@st.cache_resource
def load_model():
    model_path = 'xgb_model.pkl'
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found")
        st.info("Please ensure:")
        st.info("1. xgb_model.pkl is in the app directory")
        st.info("2. File name is spelled correctly")
        return None
    
    try:
        # Check if file is empty
        if os.path.getsize(model_path) == 0:
            st.error("Model file is empty")
            return None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        st.success("Model loaded successfully!")
        return model
        
    except pickle.UnpicklingError as e:
        st.error(f"Model file format error: {str(e)}")
        st.info("Possible reasons:")
        st.info("- File is not a valid pickle file")
        st.info("- Incompatible Python version")
        st.info("- Model created with different library version")
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# Set page title
st.title("S/D Predictor")
st.markdown("Enter parameters to predict S/D value")

# Create input form
with st.form("input_form"):
    kc = st.number_input("KC", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    rep = st.number_input("Rep", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    red = st.number_input("Red", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    shield = st.number_input("Shield", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    submit = st.form_submit_button("Predict")

# Make prediction when user submits
if submit:
    if model is None:
        st.error("Cannot make prediction. Model failed to load. Please check model file.")
    else:
        # Create input data
        input_data = pd.DataFrame({
            'KC': [kc],
            'Rep': [rep],
            'Red': [red],
            'Shield': [shield]
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            st.success(f"Prediction result: {prediction[0]:.4f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please ensure input data format matches training data")
