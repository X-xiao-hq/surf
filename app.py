import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import subprocess

# 检查并安装依赖（更安全的版本）
def check_and_install_dependencies():
    try:
        # 尝试导入xgboost，如果失败则提示手动安装
        try:
            import xgboost
            return True
        except ImportError:
            st.error("""
            **XGBoost is not installed. Please install it manually:**
            
            **Option 1 (Recommended):**
            ```bash
            pip install xgboost --user
            ```
            
            **Option 2 (Pre-built version):**
            ```bash
            pip install xgboost==1.7.0
            ```
            
            **Option 3 (Conda):**
            ```bash
            conda install -c conda-forge xgboost
            ```
            
            After installation, please restart the app.
            """)
            return False
    except Exception as e:
        st.error(f"Dependency check failed: {e}")
        return False

# Load model function with better error handling
def load_model(model_file):
    try:
        # 检查文件是否存在
        if not os.path.exists(model_file):
            st.error(f"❌ Model file '{model_file}' not found")
            st.info("Please ensure the model file is in the same directory as this script")
            return None
        
        # 检查文件大小
        if os.path.getsize(model_file) == 0:
            st.error("❌ Model file is empty")
            return None
        
        # 尝试加载模型
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)
        
        st.success("✅ Model loaded successfully")
        return loaded_model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("This might be due to:")
        st.info("- XGBoost not installed")
        st.info("- Model file corruption")
        st.info("- Version incompatibility")
        return None

# Main function
def main():
    """S/D Prediction App"""
    st.title("S/D Predictor")
    
    # 检查依赖
    if not check_and_install_dependencies():
        return
    
    st.subheader("Input Parameters")
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        kc = st.number_input("KC", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        rep = st.number_input("Rep", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    with col2:
        red = st.number_input("Red", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        shield = st.number_input("Shield", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    if st.button("Predict", type="primary"):
        loaded_model = load_model("xgb_model.pkl")
        if loaded_model is not None:
            try:
                # Create feature array
                feature_list = [kc, rep, red, shield]
                single_sample = np.array(feature_list).reshape(1, -1)
                
                prediction = loaded_model.predict(single_sample)
                st.success(f"**Predicted S/D value:** {prediction[0]:.4f}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == '__main__':
    main()
