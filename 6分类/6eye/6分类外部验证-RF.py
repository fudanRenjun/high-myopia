import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林
from joblib import dump

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载训练集和测试集数据
train_data = pd.read_csv('E:/RS/6eye/feature/RF/train-80-1.csv')  # 训练集路径
test_data = pd.read_csv('E:/RS/6eye/feature/RF/validation-20-1.csv')    # 测试集路径

# 假设第一列是标签列，其他列是特征
X_train = train_data.iloc[:, 1:]  # 训练集特征列
y_train = train_data.iloc[:, 0]  # 训练集标签列

X_test = test_data.iloc[:, 1:]   # 测试集特征列
y_test = test_data.iloc[:, 0]    # 测试集标签列

# 构建随机森林模型
model_rf = RandomForestClassifier(class_weight='balanced')
model_rf.fit(X_train, y_train)

# 预测测试集
y_pred = model_rf.predict(X_test)
y_pred_proba = model_rf.predict_proba(X_test)

# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵热力图
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 8}, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('Confusion Matrix Heat Map', fontsize=8)
plt.show()

# 计算并打印每个类别的指标
num_classes = conf_matrix.shape[0]
for i in range(num_classes):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    tn = conf_matrix.sum() - (tp + fp + fn)

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) != 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) != 0 else 0
    sensitivity = recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    print(f"Class {i}:")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value: {positive_predictive_value:.4f}")
    print(f"Negative Predictive Value: {negative_predictive_value:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")

# 计算ROC曲线和AUC
y_test_one_rf = label_binarize(y_test, classes=np.unique(y_train))
rf_AUC = {}
rf_FPR = {}
rf_TPR = {}

for i in range(y_test_one_rf.shape[1]):
    rf_FPR[i], rf_TPR[i], _ = roc_curve(y_test_one_rf[:, i], y_pred_proba[:, i])
    rf_AUC[i] = auc(rf_FPR[i], rf_TPR[i])
print("ROC AUC for each class:", rf_AUC)

# 计算宏平均ROC曲线和AUC
rf_FPR_final = np.unique(np.concatenate([rf_FPR[i] for i in range(y_test_one_rf.shape[1])]))
rf_TPR_all = np.zeros_like(rf_FPR_final)
for i in range(y_test_one_rf.shape[1]):
    rf_TPR_all += np.interp(rf_FPR_final, rf_FPR[i], rf_TPR[i])
rf_TPR_final = rf_TPR_all / y_test_one_rf.shape[1]
rf_AUC_final = auc(rf_FPR_final, rf_TPR_final)
print(f"Macro Average AUC with Random Forest: {rf_AUC_final}")

# 绘制ROC曲线
plt.figure(figsize=(10, 5), dpi=300)
for i, color in zip(range(y_test_one_rf.shape[1]), ['b', 'g', 'r', 'c', 'm','y']):
    plt.plot(rf_FPR[i], rf_TPR[i], color=color, linestyle='-', label=f'Class {i} ROC AUC={rf_AUC[i]:.4f}', lw=0.8)
plt.plot(rf_FPR_final, rf_TPR_final, color='black', linestyle='-', label=f'Macro Avg ROC AUC={rf_AUC_final:.4f}', lw=1)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('Random Forest Classification ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()

# 计算PR曲线和平均精确度
pr_AUC = {}
pr_Precision = {}
pr_Recall = {}

for i in range(y_test_one_rf.shape[1]):
    pr_Recall[i], pr_Precision[i], _ = precision_recall_curve(y_test_one_rf[:, i], y_pred_proba[:, i])
    pr_AUC[i] = average_precision_score(y_test_one_rf[:, i], y_pred_proba[:, i])
print("PR AUC for each class:", pr_AUC)

# 绘制PR曲线
plt.figure(figsize=(10, 5), dpi=300)
for i, color in zip(range(y_test_one_rf.shape[1]), ['b', 'g', 'r', 'c', 'm','y']):
    plt.plot(pr_Recall[i], pr_Precision[i], color=color, linestyle='-', label=f'Class {i} PR AUC={pr_AUC[i]:.4f}', lw=0.8)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Recall', fontsize=7)
plt.ylabel('Precision', fontsize=7)
plt.title('Random Forest Classification PR Curves and AUC', fontsize=8)
plt.legend(loc='lower left', framealpha=0.9, fontsize=5)
plt.show()

# 保存模型
dump(model_rf, 'E:/RS/6eye/feature/RF/RF9_shuffled.pkl')

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# 加载外部验证集
external_validation_data = pd.read_csv('E:/RS/6eye/feature/RF/外部验证/yan2.1.csv')  # 外部验证集路径

# 假设第一列是标签列，其他列是特征
X_external = external_validation_data.iloc[:, 1:]  # 外部验证集特征列
y_external = external_validation_data.iloc[:, 0]  # 外部验证集标签列

# 加载模型
loaded_model = load('E:/RS/6eye/feature/RF/RF9_shuffled.pkl')

# 使用加载的模型进行预测
y_external_pred = loaded_model.predict(X_external)
y_external_pred_proba = loaded_model.predict_proba(X_external)

# 打印外部验证集分类报告
print("External Validation Classification Report:")
print(classification_report(y_external, y_external_pred))

# 绘制混淆矩阵热力图
external_conf_matrix = confusion_matrix(y_external, y_external_pred)
plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(external_conf_matrix, annot=True, annot_kws={'size': 8}, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('External Validation Confusion Matrix Heat Map', fontsize=8)
plt.show()

# 合并外部验证集数据
external_validation_data['True_Label'] = y_external
external_validation_data['Predicted_Label'] = y_external_pred

# 标记错误预测
external_validation_data['Error'] = external_validation_data['True_Label'] != external_validation_data['Predicted_Label']
external_validation_data['Misclassified_As'] = external_validation_data.apply(
    lambda row: row['Predicted_Label'] if row['Error'] else None, axis=1)

# 保存整合后的预测结果到CSV
output_path = 'E:/RS/6eye/feature/RF/external_prediction_results.csv'
external_validation_data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Prediction results saved to {output_path}")

# 计算ROC曲线和AUC
y_external_one_rf = label_binarize(y_external, classes=np.unique(y_external))
external_rf_AUC = {}
external_rf_FPR = {}
external_rf_TPR = {}

for i in range(y_external_one_rf.shape[1]):
    external_rf_FPR[i], external_rf_TPR[i], _ = roc_curve(y_external_one_rf[:, i], y_external_pred_proba[:, i])
    external_rf_AUC[i] = auc(external_rf_FPR[i], external_rf_TPR[i])
print("External Validation ROC AUC for each class:", external_rf_AUC)

# 计算宏平均ROC曲线和AUC
external_rf_FPR_final = np.unique(np.concatenate([external_rf_FPR[i] for i in range(y_external_one_rf.shape[1])]))
external_rf_TPR_all = np.zeros_like(external_rf_FPR_final)
for i in range(y_external_one_rf.shape[1]):
    external_rf_TPR_all += np.interp(external_rf_FPR_final, external_rf_FPR[i], external_rf_TPR[i])
external_rf_TPR_final = external_rf_TPR_all / y_external_one_rf.shape[1]
external_rf_AUC_final = auc(external_rf_FPR_final, external_rf_TPR_final)
print(f"Macro Average AUC for External Validation: {external_rf_AUC_final}")

# 绘制外部验证集ROC曲线
plt.figure(figsize=(10, 5), dpi=300)
for i, color in zip(range(y_external_one_rf.shape[1]), ['b', 'g', 'r', 'c', 'm', 'y']):
    plt.plot(external_rf_FPR[i], external_rf_TPR[i], color=color, linestyle='-',
             label=f'Class {i} ROC AUC={external_rf_AUC[i]:.4f}', lw=0.8)
plt.plot(external_rf_FPR_final, external_rf_TPR_final, color='black', linestyle='-',
         label=f'Macro Avg ROC AUC={external_rf_AUC_final:.4f}', lw=1)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('External Validation ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()

from sklearn.calibration import calibration_curve

# 逐类别校准曲线
plt.figure(figsize=(10, 6), dpi=300)
for i in range(y_pred_proba.shape[1]):
    # 将当前类别视为正例，其他为负例
    binary_y_true = (y_test == i).astype(int)
    prob_true, prob_pred = calibration_curve(binary_y_true, y_pred_proba[:, i], n_bins=10, strategy='quantile')

    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}', linewidth=0.7, markersize=2)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated', linewidth=0.7)
plt.xlabel('Mean Predicted Probability', fontsize=7)
plt.ylabel('Fraction of Positives', fontsize=7)
plt.title('Per-Class Calibration Curves (One-vs-Rest)', fontsize=8)
plt.legend(loc='upper left', fontsize=6)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.grid(True, alpha=0.3)
plt.show()