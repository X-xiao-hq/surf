import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 尝试加载模型，如果失败则使用备用方案
def load_model_safe(model_path):
    try:
        # 首先尝试直接加载（如果XGBoost可用）
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, True
    except Exception as e:
        st.warning(f"XGBoost model loading failed: {e}")
        st.info("Falling back to simple prediction method")
        return None, False

# 简单的线性预测模型（备用）
def simple_prediction(kc, rep, red, shield):
    # 这是一个简单的加权平均预测，您可以根据需要调整权重
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # 调整这些权重
    features = np.array([kc, rep, red, shield])
    return np.dot(features, weights)

def main():
    st.title("S/D Predictor")
    st.subheader("Input Parameters")
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        kc = st.number_input("KC", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        rep = st.number_input("Rep", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    with col2:
        red = st.number_input("Red", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        shield = st.number_input("Shield", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    if st.button("Predict"):
        # 尝试加载XGBoost模型
        model, success = load_model_safe("xgb_model.pkl")
        
        if success:
            try:
                input_data = np.array([[kc, rep, red, shield]])
                prediction = model.predict(input_data)
                st.success(f"**XGBoost Prediction:** {prediction[0]:.4f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                # 失败时使用备用方法
                prediction = simple_prediction(kc, rep, red, shield)
                st.info(f"**Fallback Prediction:** {prediction:.4f}")
        else:
            # 使用备用预测方法
            prediction = simple_prediction(kc, rep, red, shield)
            st.info(f"**Simple Prediction:** {prediction:.4f}")
            st.warning("Using fallback prediction method (XGBoost not available)")

if __name__ == '__main__':
    main()
