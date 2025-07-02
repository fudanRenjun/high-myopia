import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
import seaborn as sns
import xgboost as xgb
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# --------------------
# 1. 训练 & 测试集评估
# --------------------
train_data = pd.read_csv('E:/RS/6eye/train-80_shuffled.csv')
test_data  = pd.read_csv('E:/RS/6eye/validation-20_shuffled.csv')

X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test  = test_data.iloc[:, 1:]
y_test  = test_data.iloc[:, 0]

model_rf = GaussianNB()

model_rf.fit(X_train, y_train)

y_pred       = model_rf.predict(X_test)
y_pred_proba = model_rf.predict_proba(X_test)

print("=== Test Set Classification Report ===")
print(classification_report(y_test, y_pred))

cm_test = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='YlGnBu', annot_kws={'size':8}, cbar_kws={'shrink':0.75})
plt.tick_params(labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('Test Set Confusion Matrix', fontsize=8)
plt.close()

# --------------------
# 2. 保存模型
# --------------------
dump(model_rf, 'E:/RS/6eye/XGB.pkl')

# --------------------
# 3. 外部验证集评估
# --------------------
external_data = pd.read_csv('E:/RS/6eye/yan1-填充_shuffled.csv')
X_ext = external_data.iloc[:, 1:]
y_ext = external_data.iloc[:, 0]

loaded_model     = load('E:/RS/6eye/XGB.pkl')
y_ext_pred       = loaded_model.predict(X_ext)
y_ext_pred_proba = loaded_model.predict_proba(X_ext)

print("=== External Validation Classification Report ===")
print(classification_report(y_ext, y_ext_pred))

cm_ext = confusion_matrix(y_ext, y_ext_pred)
plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(cm_ext, annot=True, fmt='d', cmap='YlGnBu', annot_kws={'size':8}, cbar_kws={'shrink':0.75})
plt.tick_params(labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('External Validation Confusion Matrix', fontsize=8)
plt.close()

# 多分类 ROC & AUC（外部验证集）
classes = np.unique(y_train)  # 或者 np.unique(y_ext)
y_ext_bin = label_binarize(y_ext, classes=classes)
n_classes = y_ext_bin.shape[1]

external_rf_FPR = {}
external_rf_TPR = {}
external_rf_AUC = {}

for i in range(n_classes):
    external_rf_FPR[i], external_rf_TPR[i], _ = roc_curve(y_ext_bin[:, i], y_ext_pred_proba[:, i])
    external_rf_AUC[i] = auc(external_rf_FPR[i], external_rf_TPR[i])

external_rf_FPR_final = np.unique(
    np.concatenate([external_rf_FPR[i] for i in range(n_classes)])
)
external_rf_TPR_all = np.zeros_like(external_rf_FPR_final)
for i in range(n_classes):
    external_rf_TPR_all += np.interp(
        external_rf_FPR_final, external_rf_FPR[i], external_rf_TPR[i]
    )
external_rf_TPR_final = external_rf_TPR_all / n_classes
external_rf_AUC_final = auc(external_rf_FPR_final, external_rf_TPR_final)

# 绘制外部验证集 ROC 曲线
plt.figure(figsize=(10, 5), dpi=300)
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for i in range(n_classes):
    plt.plot(
        external_rf_FPR[i],
        external_rf_TPR[i],
        color=colors[i % len(colors)],
        linestyle='-',
        lw=0.8,
        label=f'Class {i} ROC AUC={external_rf_AUC[i]:.4f}'
    )
plt.plot(
    external_rf_FPR_final,
    external_rf_TPR_final,
    color='black',
    linestyle='-',
    lw=1,
    label=f'Macro Avg ROC AUC={external_rf_AUC_final:.4f}'
)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('External Validation ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()

# 统计外部验证各类指标并保存至 Excel
metrics_list = []
for i in range(n_classes):
    tp = cm_ext[i, i]
    fp = cm_ext[:, i].sum() - tp
    fn = cm_ext[i, :].sum() - tp
    tn = cm_ext.sum() - (tp + fp + fn)

    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    ppv         = tp / (tp + fp) if tp + fp > 0 else 0
    npv         = tn / (tn + fn) if tn + fn > 0 else 0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    f1          = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0

    metrics_list.append({
        'Class':    i,
        'AUC值':    external_rf_AUC[i],
        '敏感性':    sensitivity,
        '特异性':    specificity,
        'PPV':       ppv,
        'NPV':       npv,
        '准确率':    accuracy,
        'F1':        f1
    })

df = pd.DataFrame(metrics_list).set_index('Class').T
df.columns = [str(c) for c in df.columns]
df['mean'] = df.mean(axis=1)

df.to_excel('E:/RS/6eye/external_validation_metrics.xlsx', index=True)
print("✅ 外部验证指标已保存至 E:/RS/6eye/external_validation_metrics.xlsx")
print(df)