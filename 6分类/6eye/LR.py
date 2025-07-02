import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier  # 用于多分类的逻辑回归
import seaborn as sns
from imblearn.over_sampling import SMOTE  # 导入SMOTE（平衡采样）

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载训练集和测试集数据
train_data = pd.read_csv('E:/RS/6eye/train-80.csv')  # 训练集路径
test_data = pd.read_csv('E:/RS/6eye/validation-20.csv')    # 测试集路径

# 假设第一列是标签列，其他列是特征
X_train = train_data.iloc[:, 1:]  # 训练集特征列
y_train = train_data.iloc[:, 0]  # 训练集标签列

X_test = test_data.iloc[:, 1:]   # 测试集特征列
y_test = test_data.iloc[:, 0]    # 测试集标签列

# 数据标准化
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 使用SMOTE进行过采样，平衡训练集
smote = SMOTE(random_state=42)
X_train_s_res, y_train_res = smote.fit_resample(X_train_s, y_train)

# 构建逻辑回归模型
lr = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))

# 训练LR模型
lr.fit(X_train_s_res, y_train_res)

# 预测测试集
y_pred = lr.predict(X_test_s)
y_pred_proba = lr.predict_proba(X_test_s)

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
y_test_one_lr = label_binarize(y_test, classes=np.unique(y_train))
lr_AUC = {}
lr_FPR = {}
lr_TPR = {}

for i in range(y_test_one_lr.shape[1]):
    lr_FPR[i], lr_TPR[i], _ = roc_curve(y_test_one_lr[:, i], y_pred_proba[:, i])
    lr_AUC[i] = auc(lr_FPR[i], lr_TPR[i])
print("ROC AUC for each class:", lr_AUC)

# 计算宏平均ROC曲线和AUC
lr_FPR_final = np.unique(np.concatenate([lr_FPR[i] for i in range(y_test_one_lr.shape[1])]))
lr_TPR_all = np.zeros_like(lr_FPR_final)
for i in range(y_test_one_lr.shape[1]):
    lr_TPR_all += np.interp(lr_FPR_final, lr_FPR[i], lr_TPR[i])
lr_TPR_final = lr_TPR_all / y_test_one_lr.shape[1]
lr_AUC_final = auc(lr_FPR_final, lr_TPR_final)
print(f"Macro Average AUC with LR: {lr_AUC_final}")

# 绘制ROC曲线
plt.figure(figsize=(10, 5), dpi=300)
for i, color in zip(range(y_test_one_lr.shape[1]), ['b', 'g', 'r', 'c', 'm', 'y']):
    plt.plot(lr_FPR[i], lr_TPR[i], color=color, linestyle='-', label=f'Class {i} ROC AUC={lr_AUC[i]:.4f}', lw=0.8)
plt.plot(lr_FPR_final, lr_TPR_final, color='black', linestyle='-', label=f'Macro Avg ROC AUC={lr_AUC_final:.4f}', lw=1)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('LR Classification ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()

# 计算PR曲线和平均精确度
pr_AUC = {}
pr_Precision = {}
pr_Recall = {}

for i in range(y_test_one_lr.shape[1]):
    pr_Recall[i], pr_Precision[i], _ = precision_recall_curve(y_test_one_lr[:, i], y_pred_proba[:, i])
    pr_AUC[i] = average_precision_score(y_test_one_lr[:, i], y_pred_proba[:, i])
print("PR AUC for each class:", pr_AUC)

# 绘制PR曲线
plt.figure(figsize=(10, 5), dpi=300)
for i, color in zip(range(y_test_one_lr.shape[1]), ['b', 'g', 'r', 'c', 'm', 'y']):
    plt.plot(pr_Recall[i], pr_Precision[i], color=color, linestyle='-', label=f'Class {i} PR AUC={pr_AUC[i]:.4f}', lw=0.8)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Recall', fontsize=7)
plt.ylabel('Precision', fontsize=7)
plt.title('LR Classification PR Curves and AUC', fontsize=8)
plt.legend(loc='lower left', framealpha=0.9, fontsize=5)
plt.show()
