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
data = pd.read_csv('E:/RS/6eye/Feature/train-20.csv')

# 检查缺失值并填充
print(data.isnull().sum())
data_filled = data.fillna(data.mean())

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 划分训练集和测试集
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 使用SMOTE进行过采样
smote = SMOTE(sampling_strategy={0: 2000, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5:2000}, random_state=42)
xtrain_resampled, ytrain_resampled = smote.fit_resample(xtrain, ytrain)

# 查看平衡后的样本分布
print("Resampled label distribution:", np.bincount(ytrain_resampled))

# 数据标准化
scaler = MinMaxScaler()
xtrain_s = scaler.fit_transform(xtrain_resampled)
xtest_s = scaler.transform(xtest)

# 构建XGBoost模型
model_rf = xgb.XGBClassifier(objective='multi:softmax', num_class=6, random_state=42, use_label_encoder=False)
model_rf.fit(xtrain_s, ytrain_resampled)

# 获取特征重要性
feature_importances = model_rf.feature_importances_

# 对特征进行排序，选取最重要的特征
sorted_idx = np.argsort(feature_importances)[::-1]  # 从大到小排序

# 存储每个模型（按特征数目）的AUC结果
auc_results = []

# 逐步选择1到20个最重要的特征
for num_features in range(1, 41):
    selected_features = sorted_idx[:num_features]  # 选择前num_features个特征
    X_train_selected = xtrain_s[:, selected_features]
    X_test_selected = xtest_s[:, selected_features]

    # 重新训练模型
    model_rf.fit(X_train_selected, ytrain_resampled)

    # 预测
    y_pred = model_rf.predict(X_test_selected)
    y_pred_proba = model_rf.predict_proba(X_test_selected)

    # 计算ROC曲线和AUC
    ytest_one_rf = label_binarize(ytest, classes=np.unique(y))
    rf_AUC = {}
    for i in range(ytest_one_rf.shape[1]):
        rf_FPR, rf_TPR, _ = roc_curve(ytest_one_rf[:, i], y_pred_proba[:, i])
        rf_AUC[i] = auc(rf_FPR, rf_TPR)

    # 计算宏平均AUC
    rf_FPR_final = np.unique(np.concatenate([rf_FPR for _ in range(ytest_one_rf.shape[1])]))
    rf_TPR_all = np.zeros_like(rf_FPR_final)
    for i in range(ytest_one_rf.shape[1]):
        rf_TPR_all += np.interp(rf_FPR_final, rf_FPR, rf_TPR)
    rf_TPR_final = rf_TPR_all / ytest_one_rf.shape[1]
    rf_AUC_final = auc(rf_FPR_final, rf_TPR_final)

    # 存储AUC结果
    auc_results.append([num_features, rf_AUC_final])

# 创建DataFrame
auc_df = pd.DataFrame(auc_results, columns=['Num_Features', 'Macro_AUC'])

# 保存AUC结果为CSV文件
auc_df.to_csv('E:/RS/6eye/Feature/feature_importance_auc-1.csv', index=False)

print("Feature importance AUC results have been saved to 'feature_importance_auc.csv'.")
