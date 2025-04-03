import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 加载模型和定义特征名称（保持不变）
model = joblib.load('RF9.pkl')
feature_names = ["EOS", "PCT", "P_BASO", "M", "UA", "LPB", "BASO", "P_EOS", "TG"]

# Streamlit 用户界面（保持不变）
st.title("Screening complications of high myopia")
st.write('Please enter the following clinical indicators to screen complications of high myopia:')

# 用户输入部分（保持不变）
input_eos = st.number_input("Eosinophil count(10^9/L):", min_value=0.0, max_value=100.0, value=0.07, format="%.2f")
input_pct = st.number_input("Plateletcrit (%):", min_value=0.0, max_value=100.0, value=0.17, format="%.2f")
input_p_baso = st.number_input("Basophil percentage(%):", min_value=0.0, max_value=100.0, value=0.5, format="%.1f")
input_m = st.number_input("Monocyte count(10^9/L):", min_value=0.0, max_value=100.0, value=0.37, format="%.2f")
input_ua = st.number_input("Uric Acid (μmol/L):", min_value=0.0, max_value=1000.0, value=180.0, format="%.1f")
input_lpb = st.number_input("Lipoprotein B (g/L):", min_value=0.0, max_value=100.0, value=0.8, format="%.1f")
input_baso = st.number_input("Basophil count(10^9/L):", min_value=0.0, max_value=100.0, value=0.02, format="%.2f")
input_p_eos = st.number_input("Eosinophil percentage(%):", min_value=0.0, max_value=100.0, value=1.7, format="%.1f")
input_tg = st.number_input("Triglycerides (mmol/L):", min_value=0.0, max_value=1000.0, value=1.95, format="%.2f")

# 数据整理和预测（保持不变）
feature_values = [input_eos, input_pct, input_p_baso, input_m, input_ua, input_lpb, input_baso, input_p_eos, input_tg]
features = np.array([feature_values])

label_mapping = {
    0: "High myopia",
    1: "Cataract",
    2: "Macular degeneration",
    3: "Chorioretinopathy",
    4: "Glaucoma",
    5: "Retinal detachment"
}

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    predicted_label = label_mapping[predicted_class]
    
    # 显示预测结果（保持不变）
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Probability:** {predicted_proba[predicted_class] * 100:.1f}%")

    # ---------- 新增：绘制柱状图 ----------
    st.write("**Prediction Probability Distribution**")
    
    # 准备数据
    labels = list(label_mapping.values())
    probabilities = predicted_proba * 100  # 转换为百分比
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, probabilities, color='skyblue')
    
    # 自定义样式
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Predicted Probabilities for Each Class', fontsize=14)
    ax.set_ylim(0, 100)  # 固定y轴范围
    
    # 在柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # 旋转x轴标签避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 显示图表
    st.pyplot(fig)
