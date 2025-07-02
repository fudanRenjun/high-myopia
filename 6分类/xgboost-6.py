import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载CSV文件
data = pd.read_csv('E:/RS/6eye/merged_data.csv')

# 检查缺失值并填充
print(data.isnull().sum())
data_filled = data.fillna(data.mean())

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 划分训练集和测试集
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 使用SMOTE进行过采样
smote = SMOTE(sampling_strategy={ 0:2127, 1: 2127, 2: 2127, 3: 2127, 4: 2127, 5: 2127}, random_state=42)
xtrain_resampled, ytrain_resampled = smote.fit_resample(xtrain, ytrain)

# 查看平衡后的样本分布
print("Resampled label distribution:", np.bincount(ytrain_resampled))

# 数据标准化
scaler = MinMaxScaler()
xtrain_s = scaler.fit_transform(xtrain_resampled)
xtest_s = scaler.transform(xtest)

# 构建XGBoost模型
model_rf = xgb.XGBClassifier(objective='multi:softmax',colsample_bytree=1.0, learning_rate= 0.2, max_depth= 7, n_estimators=200, subsample=0.8,)
model_rf.fit(xtrain_s, ytrain_resampled)

# 预测测试集
y_pred = model_rf.predict(xtest_s)
y_pred_proba = model_rf.predict_proba(xtest_s)

# 打印分类报告
print("Classification Report:")
print(classification_report(ytest, y_pred))

# 绘制混淆矩阵热力图
conf_matrix = confusion_matrix(ytest, y_pred)
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
ytest_one_rf = label_binarize(ytest, classes=np.unique(y))
rf_AUC = {}
rf_FPR = {}
rf_TPR = {}

for i in range(ytest_one_rf.shape[1]):
    rf_FPR[i], rf_TPR[i], _ = roc_curve(ytest_one_rf[:, i], y_pred_proba[:, i])
    rf_AUC[i] = auc(rf_FPR[i], rf_TPR[i])
print("ROC AUC for each class:", rf_AUC)

# 计算宏平均ROC曲线和AUC
rf_FPR_final = np.unique(np.concatenate([rf_FPR[i] for i in range(ytest_one_rf.shape[1])]))
rf_TPR_all = np.zeros_like(rf_FPR_final)
for i in range(ytest_one_rf.shape[1]):
    rf_TPR_all += np.interp(rf_FPR_final, rf_FPR[i], rf_TPR[i])
rf_TPR_final = rf_TPR_all / ytest_one_rf.shape[1]
rf_AUC_final = auc(rf_FPR_final, rf_TPR_final)
print(f"Macro Average AUC with XGBoost: {rf_AUC_final}")

# 绘制ROC曲线
plt.figure(figsize=(10, 5), dpi=300)
for i, color in zip(range(ytest_one_rf.shape[1]), ['b', 'g', 'r', 'c', 'm','y']):
    plt.plot(rf_FPR[i], rf_TPR[i], color=color, linestyle='-', label=f'Class {i} ROC AUC={rf_AUC[i]:.4f}', lw=0.8)
plt.plot(rf_FPR_final, rf_TPR_final, color='black', linestyle='-', label=f'Macro Avg ROC AUC={rf_AUC_final:.4f}', lw=1)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('XGBoost Classification ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()

# 计算PR曲线和平均精确度
pr_AUC = {}
pr_Precision = {}
pr_Recall = {}

for i in range(ytest_one_rf.shape[1]):
    pr_Recall[i], pr_Precision[i], _ = precision_recall_curve(ytest_one_rf[:, i], y_pred_proba[:, i])
    pr_AUC[i] = average_precision_score(ytest_one_rf[:, i], y_pred_proba[:, i])
print("PR AUC for each class:", pr_AUC)

# 绘制PR曲线
plt.figure(figsize=(10, 5), dpi=300)
for i, color in zip(range(ytest_one_rf.shape[1]), ['b', 'g', 'r', 'c', 'm','y']):
    plt.plot(pr_Recall[i], pr_Precision[i], color=color, linestyle='-', label=f'Class {i} PR AUC={pr_AUC[i]:.4f}', lw=0.8)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Recall', fontsize=7)
plt.ylabel('Precision', fontsize=7)
plt.title('XGBoost Classification PR Curves and AUC', fontsize=8)
plt.legend(loc='lower left', framealpha=0.9, fontsize=5)
plt.show()