import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 加载模型
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# 设置页面标题
st.title("S/D 预测器")
st.markdown("输入以下参数以预测 S/D 值")

# 创建输入表单
with st.form("input_form"):
    kc = st.number_input("KC", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    rep = st.number_input("Rep", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    red = st.number_input("Red", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    shield = st.number_input("Shield", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    submitted = st.form_submit_button("预测")

# 当用户提交时进行预测
if submitted:
    # 构建输入数据（注意：模型可能需要更多特征，这里只使用4个）
    # 如果模型训练时使用了更多特征，请在这里补充默认值或提供输入方式
    input_data = pd.DataFrame({
        'KC': [kc],
        'Rep': [rep],
        'Red': [red],
        'Shield': [shield]
        # 如果还有其他特征，请在这里添加，例如：
        # 'feature5': [default_value],
        # ...
    })
    
    # 确保列的顺序与训练时一致（如果不同，请调整）
    # 可以使用 model.feature_names_in_ 查看特征顺序（如果可用）
    if hasattr(model, 'feature_names_in_'):
        input_data = input_data[model.feature_names_in_]
    
    # 进行预测
    prediction = model.predict(input_data)[0]
    
    # 显示结果
    st.success(f"预测的 S/D 值为: {prediction:.4f}")
