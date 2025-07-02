from imblearn.metrics import specificity_score
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, roc_auc_score,
                             accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report, f1_score)
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
model = load('E:/RS/6eye/feature/RF/RF9.pkl')

# 导入CSV文件
df = pd.read_csv('E:/RS/6eye/feature/RF/外部验证/yan2.csv')

# 假设第三列是标签列（0-5），从第四列开始是特征
X = df.iloc[:, 3:]  # 特征列
y_true = df.iloc[:, 2]  # 标签列

# 进行预测
y_pred = model.predict(X)
y_score = model.predict_proba(X)  # 获取所有类别的预测概率

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 绘制混淆矩阵热力图
plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 4},
            fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75},
            xticklabels=[0, 1, 2, 3, 4, 5], yticklabels=[0, 1, 2, 3, 4, 5])
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('Confusion Matrix Heat Map', fontsize=8)
plt.show()

# 计算分类报告（包含precision, recall, f1-score等）
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))

# 计算总体性能指标
accuracy = accuracy_score(y_true, y_pred)  # 准确性
precision_macro = precision_score(y_true, y_pred, average='macro')  # 宏平均精确率
recall_macro = recall_score(y_true, y_pred, average='macro')  # 宏平均召回率
f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)  # 宏平均F1得分

# 打印总体性能指标
print(f"\nOverall Performance Metrics:")
print(f"Overall Accuracy: {accuracy:.3f}")
print(f"Macro-average Precision: {precision_macro:.3f}")
print(f"Macro-average Recall: {recall_macro:.3f}")
print(f"Macro-average F1 Score: {f1_macro:.3f}")

# 绘制多类ROC曲线（OvR策略）
classes = [0, 1, 2, 3, 4, 5]
n_classes = len(classes)
y_true_bin = label_binarize(y_true, classes=classes)

# 计算每个类的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均ROC曲线和AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 计算宏平均ROC曲线和AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # 使用np.interp替代scipy.interp

mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Class {0} (AUC = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (AUC = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='black', linestyle='-', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc="lower right")
plt.show()

# 计算每个类别的详细性能指标
metrics_dict = {
    'Class': [],
    'Sensitivity (Recall)': [],
    'Specificity': [],
    'Accuracy': [],
    'PPV (Precision)': [],
    'NPV': [],
    'F1-Score': [],
    'ROC AUC': [],
    'PR AUC': []
}

for i in range(n_classes):
    # 计算二分类的混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true_bin[:, i], y_pred == i).ravel()

    # 计算各项指标
    sensitivity = recall_score(y_true_bin[:, i], y_pred == i)
    specificity = specificity_score(y_true_bin[:, i], y_pred == i)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    ppv = precision_score(y_true_bin[:, i], y_pred == i, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    f1 = f1_score(y_true_bin[:, i], y_pred == i)
    roc_auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])

    # 计算PR AUC
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    pr_auc = average_precision_score(y_true_bin[:, i], y_score[:, i])

    # 添加到字典
    metrics_dict['Class'].append(i)
    metrics_dict['Sensitivity (Recall)'].append(sensitivity)
    metrics_dict['Specificity'].append(specificity)
    metrics_dict['Accuracy'].append(accuracy)
    metrics_dict['PPV (Precision)'].append(ppv)
    metrics_dict['NPV'].append(npv)
    metrics_dict['F1-Score'].append(f1)
    metrics_dict['ROC AUC'].append(roc_auc)
    metrics_dict['PR AUC'].append(pr_auc)

# 创建DataFrame并打印
metrics_df = pd.DataFrame(metrics_dict)
print("\nDetailed Performance Metrics for Each Class:")
print(metrics_df.round(3))

metrics_df.to_csv('E:/RS/6eye/feature/RF/外部验证/class_performance_metrics.csv')
print("详细性能指标已保存为 class_performance_metrics.csv")

# 绘制多类PR曲线（OvR策略）
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])

# 计算每个类的PR曲线和AUC
precision = dict()
recall = dict()
pr_auc = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    pr_auc[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

# 计算微平均PR曲线和AUC
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
pr_auc["micro"] = average_precision_score(y_true_bin, y_score, average="micro")

# 计算宏平均PR曲线和AUC
all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
mean_precision = np.zeros_like(all_recall)
for i in range(n_classes):
    mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])  # 注意要反转数组

mean_precision /= n_classes
precision["macro"] = mean_precision
recall["macro"] = all_recall
pr_auc["macro"] = auc(recall["macro"], precision["macro"])

for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Class {0} (AP = {1:0.2f})'
                   ''.format(i, pr_auc[i]))

plt.plot(recall["macro"], precision["macro"],
         label='Macro-average PR curve (AP = {0:0.2f})'
               ''.format(pr_auc["macro"]),
         color='black', linestyle='-', linewidth=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class Precision-Recall Curves')
plt.legend(loc="lower left")
plt.show()

# ====================== DCA 决策曲线 ======================
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
        for i in range(n_classes):  # 使用实际的类别数量
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

    # 调整刻度值与刻度线的距离（pad）
    ax.tick_params(axis='both', which='major', labelsize=5, pad=2)  # 默认 pad=4

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize=5)

    return ax


# 创建子图
rows = (n_classes + 1) // 2  # 计算需要的行数
fig, axs = plt.subplots(rows, 2, figsize=(12, 3 * rows), dpi=300)  # 动态调整子图大小
if n_classes == 1:
    axs = np.array([axs])  # 确保单类别时也能正确处理

# 计算和绘制每个类的净收益
for i in range(n_classes):
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_score, y_true, i)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_true)

    if rows > 1:
        ax = axs[i // 2, i % 2]  # 选择对应的子图
    else:
        ax = axs[i % 2]

    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, i)

# 如果类别数为奇数，隐藏最后一个空子图
if n_classes % 2 != 0 and rows > 1:
    axs[-1, -1].axis('off')

plt.tight_layout(pad=1.0)  # 调整子图间距
plt.show()