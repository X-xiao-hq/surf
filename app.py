import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model function
def load_model(model_file):
    try:
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main function
def main():
    """S/D Prediction App"""
    st.title("S/D Predictor")
    st.subheader("Input Parameters")
    
    # Input fields
    kc = st.number_input("KC", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    rep = st.number_input("Rep", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    red = st.number_input("Red", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    shield = st.number_input("Shield", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    # Create feature array
    feature_list = [kc, rep, red, shield]
    single_sample = np.array(feature_list).reshape(1, -1)

    if st.button("Predict"):
        loaded_model = load_model("xgb_model.pkl")
        if loaded_model is not None:
            prediction = loaded_model.predict(single_sample)
            st.success(f"Predicted S/D value: {prediction[0]:.4f}")

if __name__ == '__main__':
    main()
