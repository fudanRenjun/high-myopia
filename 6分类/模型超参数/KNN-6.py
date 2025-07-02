import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, label_binarize
import seaborn as sns

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载CSV文件
data = pd.read_csv('E:/RS/6eye/train3.csv')

print(data.isnull().sum())
data_filled = data.fillna(data.mean())

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 划分训练集和测试集
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = MinMaxScaler()
xtrain_s = scaler.fit_transform(xtrain)
xtest_s = scaler.transform(xtest)

# 设置KNN超参数网格
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # 邻居数
    'weights': ['uniform', 'distance'],  # 权重
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # 最近邻计算算法
    'leaf_size': [20, 30, 40, 50]  # 树叶大小
}

# 使用GridSearchCV进行超参数搜索
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(xtrain_s, ytrain)

# 打印最优超参数组合
print("Best parameters found: ", grid_search.best_params_)

# 使用最优的KNN模型进行预测
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(xtest_s)
y_pred_proba = best_knn.predict_proba(xtest_s)

# 打印分类报告
print("Classification Report:")
print(classification_report(ytest, y_pred))

# 计算混淆矩阵
conf_matrix = confusion_matrix(ytest, y_pred)
# 绘制热力图
plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':4},
            fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('Confusion matrix heat map', fontsize=8)
plt.show()

# 计算并打印每个类别的指标
num_classes = conf_matrix.shape[0]
for i in range(num_classes):
    tp = conf_matrix[i, i]  # True Positive for class i
    fp = conf_matrix[:, i].sum() - tp  # False Positive for class i
    fn = conf_matrix[i, :].sum() - tp  # False Negative for class i
    tn = conf_matrix.sum() - (tp + fp + fn)  # True Negative for class i

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) != 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) != 0 else 0
    sensitivity = recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    print(f"Class {i + 1}:")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value: {positive_predictive_value:.4f}")
    print(f"Negative Predictive Value: {negative_predictive_value:.4f}")
    print(f"Sensitivity (Recall for class {i + 1}): {sensitivity:.4f}")

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
print(f"Macro Average AUC with KNN: {rf_AUC_final}")

# 绘制ROC曲线
plt.figure(figsize=(10, 5), dpi=300)
for i in range(len(rf_AUC)):
    plt.plot(rf_FPR[i], rf_TPR[i], label=f'Class {i + 1} ROC  AUC={rf_AUC[i]:.4f}', lw=0.8)
plt.plot(rf_FPR_final, rf_TPR_final, color='black', label=f'Macro Average ROC AUC={rf_AUC_final:.4f}', lw=1)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('KNN Classification ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', fontsize=6)
plt.show()

# 计算PR曲线和AUC
pr_AUC = {}
pr_Precision = {}
pr_Recall = {}

for i in range(ytest_one_rf.shape[1]):
    pr_Recall[i], pr_Precision[i], _ = precision_recall_curve(ytest_one_rf[:, i], y_pred_proba[:, i])
    pr_AUC[i] = average_precision_score(ytest_one_rf[:, i], y_pred_proba[:, i])
print("PR AUC for each class:", pr_AUC)

# 绘制PR曲线
plt.figure(figsize=(10, 5), dpi=300)
for i in range(len(pr_AUC)):
    plt.plot(pr_Recall[i], pr_Precision[i], label=f'Class {i + 1} PR AUC={pr_AUC[i]:.4f}', lw=0.8)
plt.xlabel('Recall', fontsize=7)
plt.ylabel('Precision', fontsize=7)
plt.title('KNN Classification PR Curves and AUC', fontsize=8)
plt.legend(loc='lower left', fontsize=6)
plt.show()
