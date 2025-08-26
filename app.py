# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 加载模型
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)  # 这里需要缩进
    return model

model = load_model()

# 设置页面标题
st.title("S/D 预测器")
st.markdown("输入以下参数以预测 S/D 值")

# 创建输入表单
with st.form("input_form"):  # 这里应该是冒号: 而不是分号;
    kc = st.number_input("KC", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    rep = st.number_input("Rep", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    red = st.number_input("Red", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    shield = st.number_input("Shield", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    submitted = st.form_submit_button("预测")

# 当用户提交时进行预测
if submitted:
    # 构建输入数据
    input_data = pd.DataFrame({
        'KC': [kc],  # 注意：这里应该是小写的 kc，并且注释需要完整
        'Rep': [rep],
        'Red': [red],
        'Shield': [shield]
    })
    
    # 进行预测（这里需要添加实际的预测代码）
    try:
        prediction = model.predict(input_data)
        st.success(f"预测结果: {prediction[0]}")
    except Exception as e:
        st.error(f"预测时出错: {e}")
