import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载训练集和测试集数据
train_data = pd.read_csv('E:/RS/6eye/feature/RF/train-80-1.csv')  # 训练集路径
test_data = pd.read_csv('E:/RS/6eye/feature/RF/validation-20-1.csv')  # 测试集路径

# 假设第一列是标签列，其他列是特征
X_train = train_data.iloc[:, 1:]  # 训练集特征列
y_train = train_data.iloc[:, 0]  # 训练集标签列

X_test = test_data.iloc[:, 1:]  # 测试集特征列
y_test = test_data.iloc[:, 0]  # 测试集标签列

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
for i, color in zip(range(y_test_one_rf.shape[1]), ['b', 'g', 'r', 'c', 'm', 'y']):
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
for i, color in zip(range(y_test_one_rf.shape[1]), ['b', 'g', 'r', 'c', 'm', 'y']):
    plt.plot(pr_Recall[i], pr_Precision[i], color=color, linestyle='-', label=f'Class {i} PR AUC={pr_AUC[i]:.4f}',
             lw=0.8)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Recall', fontsize=7)
plt.ylabel('Precision', fontsize=7)
plt.title('Random Forest Classification PR Curves and AUC', fontsize=8)
plt.legend(loc='lower left', framealpha=0.9, fontsize=5)
plt.show()

# ==================== 决策曲线分析 (DCA) ====================

# 定义不同的阈值组
thresh_group = np.arange(0, 1, 0.05)


# 定义 calculate_net_benefit_model 函数
def calculate_net_benefit_model(thresh_group, y_probs, y_test, class_index):
    net_benefit_model = []
    n = len(y_test)

    for threshold in thresh_group:
        tp = 0  # True Positives
        fp = 0  # False Positives
        for prob, actual in zip(y_probs[:, class_index], y_test):
            if prob >= threshold:  # 预测为当前类
                if actual == class_index:
                    tp += 1
                else:
                    fp += 1
        # 计算当前阈值下的净收益
        threshold_prob = threshold / (1 - threshold)
        net_benefit = (tp / n) - (fp / n) * threshold_prob
        net_benefit_model.append(net_benefit)

    return net_benefit_model


# 定义 calculate_net_benefit_all 函数
def calculate_net_benefit_all(thresh_group, y_test):
    net_benefit_all = []
    n = len(y_test)

    for threshold in thresh_group:
        threshold_prob = threshold / (1 - threshold)
        net_benefit = []
        for i in range(num_classes):  # 使用实际的类别数量
            total_positive = np.sum(y_test == i)  # 当前类的正样本数
            net_benefit_value = (total_positive / n) - ((n - total_positive) / n) * threshold_prob
            net_benefit.append(net_benefit_value)
        net_benefit_all.append(net_benefit)

    return np.array(net_benefit_all)  # 返回二维数组，包含每个阈值下每个类的净收益


# 定义 plot_DCA 函数（修改版）
def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, class_index):
    ax.plot(thresh_group, net_benefit_model, label=f"Model (Class {class_index})", linestyle='-', linewidth=2)
    ax.plot(thresh_group, net_benefit_all[:, class_index], label="Treat All", linestyle='--', linewidth=2)
    ax.plot(thresh_group, np.zeros(len(thresh_group)), label="Treat None", linestyle=':', linewidth=2)

    treat_all_none_max = np.maximum(net_benefit_all[:, class_index], np.zeros(len(thresh_group)))
    ax.fill_between(thresh_group, net_benefit_model, treat_all_none_max,
                    where=(np.array(net_benefit_model) > treat_all_none_max),
                    color='lightcoral', alpha=0.3, interpolate=True, label='Model > Treat All and None')

    ax.set_title(f"Decision Curve Analysis - Class {class_index}", fontsize=5)
    ax.set_xlabel("Threshold Probability", fontsize=5)
    ax.set_ylabel("Net Benefit", fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5, pad=2)  # pad 控制距离，默认是4
    ax.set_xlabel("Threshold Probability", fontsize=5, labelpad=2)  # labelpad 控制距离
    ax.set_ylabel("Net Benefit", fontsize=5, labelpad=2)
    # 调整标题与图之间的距离（titlepad）
    ax.set_title(f"Decision Curve Analysis - Class {class_index}", fontsize=5, pad=2)  # 默认 pad=6

    # 获取横坐标0刻度的净收益值
    zero_thresh_net_benefit = net_benefit_model[0]  # 第一个阈值是0

    # 根据0刻度的净收益值动态设置Y轴范围
    y_min = min(-0.03, zero_thresh_net_benefit - 0.05)  # 保持最小-0.03，但可以更低
    y_max = max(0.25, zero_thresh_net_benefit + 0.05)  # 保持最大0.25，但可以更高

    # 设置Y轴范围和刻度
    ax.set_ylim(y_min, y_max)

    # 创建合理的刻度间隔
    y_ticks = np.linspace(y_min, y_max, num=5)  # 生成5个均匀分布的刻度
    ax.set_yticks(np.round(y_ticks, 2))  # 保留两位小数

    ax.tick_params(axis='both', which='major', labelsize=5)  # 设置刻度值的大小
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize=5)

    return ax


# 创建子图
rows = (num_classes + 1) // 2  # 计算需要的行数
fig, axs = plt.subplots(rows, 2, figsize=(12, 3 * rows), dpi=300)  # 动态调整子图大小
if num_classes == 1:
    axs = np.array([axs])  # 确保单类别时也能正确处理

# 计算和绘制每个类的净收益
for i in range(num_classes):
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_proba, y_test, i)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)

    if rows > 1:
        ax = axs[i // 2, i % 2]  # 选择对应的子图
    else:
        ax = axs[i % 2]

    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, i)

# 如果类别数为奇数，隐藏最后一个空子图
if num_classes % 2 != 0 and rows > 1:
    axs[-1, -1].axis('off')

plt.tight_layout()  # 调整子图间距
plt.show()