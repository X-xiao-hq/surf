import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 简单的依赖检查
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def main():
    st.title("S/D Predictor")
    
    if not XGBOOST_AVAILABLE:
        st.error("""
        ⚠️ XGBoost is not installed!
        
        Please run these commands in your terminal:
        
        **For pip:**
        ```
        pip install xgboost
        ```
        
        **For conda:**
        ```
        conda install -c conda-forge xgboost
        ```
        
        Then restart this app.
        """)
        return
    
    st.subheader("Input Parameters")
    
    kc = st.number_input("KC", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    rep = st.number_input("Rep", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    red = st.number_input("Red", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    shield = st.number_input("Shield", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    if st.button("Predict"):
        try:
            with open('xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            input_data = np.array([[kc, rep, red, shield]])
            prediction = model.predict(input_data)
            
            st.success(f"Prediction: {prediction[0]:.4f}")
            
        except FileNotFoundError:
            st.error("Model file 'xgb_model.pkl' not found")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
