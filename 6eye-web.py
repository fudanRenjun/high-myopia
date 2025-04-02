import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载随机森林模型
model = joblib.load('RF9.pkl')

# 定义新的特征名称
feature_names = [
    "EOS", "PCT", "P_BASO", "M", "UA", "LPB",
    "BASO", "P_EOS", "TG"
]

# Streamlit 用户界面
st.title("Screening complications of high myopia")
st.write('Please enter the following clinical indicators to screen complications of high myopia:')

# 用户输入特征数据 - 修改为9个新特征
input_eos = st.number_input("Eosinophil count(10^9/L):", min_value=0.0, max_value=100.0, value=0.13, format="%.2f")
input_pct = st.number_input("PCT (%):", min_value=0.0, max_value=100.0, value=0.19, format="%.2f")
input_p_baso = st.number_input("Basophil percentage(%):", min_value=0.0, max_value=100.0, value=0.7,
                               format="%.1f")
input_m = st.number_input("Monocyte count(10^9/L):", min_value=0.0, max_value=100.0, value=0.34, format="%.2f")
input_ua = st.number_input("UA (μmol/L):", min_value=0.0, max_value=1000.0, value=370.0, format="%.1f")
input_lpb = st.number_input("LPB (g/L):", min_value=0.0, max_value=100.0, value=1.03, format="%.1f")
input_baso = st.number_input("Basophil count(10^9/L):", min_value=0.0, max_value=100.0, value=0.04, format="%.2f")
input_p_eos = st.number_input("Eosinophil percentage(%):", min_value=0.0, max_value=100.0, value=2.3,
                              format="%.1f")
input_tg = st.number_input("TG (mmol/L):", min_value=0.0, max_value=1000.0, value=1.0, format="%.1f")

# 将输入的数据转化为模型的输入格式
feature_values = [
    input_eos, input_pct, input_p_baso, input_m, input_ua,
    input_lpb, input_baso, input_p_eos, input_tg
]
features = np.array([feature_values])

# 数字标签和文本标签的映射关系
label_mapping = {
    0: "High myopia",
    1: "cataract",
    2: "age-related macular degeneration",
    3: "Chorioretinopathy",
    4: "Glaucoma",
    5: "Retinal detachment",
}

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    predicted_label = label_mapping[predicted_class]
    st.write(f"**Predicted Class:** {predicted_label}")

    # 显示每个类别的预测概率
    st.write("**Prediction Probabilities:**")
    for i, proba in enumerate(predicted_proba):
        label = label_mapping[i]
        st.write(f"{label}: {proba * 100:.2f}%")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100
    advice = f"The model predicts that your probability of being in class {predicted_label} is {probability:.1f}%."
    st.write(advice)
