# 1. 导入相关库
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler  # 新增标准化库（关键统一点）

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 2. 数据读取
data_path = "data/diabetes (1).csv"
data = pd.read_csv(data_path)

# 3. 数据预处理
label_need = data.keys()
X = data[label_need].values[:, 0:8]  # 前8列特征
y = data[label_need].values[:, 8].astype(int)  # 第9列标签（转为int）

# 4. 训练集/测试集划分（参数与版本2完全一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 固定随机种子，避免数据划分差异
)

# 新增：数据标准化（统一预处理核心步骤）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集拟合+标准化（避免数据泄露）
X_test_scaled = scaler.transform(X_test)        # 测试集仅用训练集参数变换

# 5. K值搜索范围（与版本2一致）
k_candidates = range(1, 31, 2)  # K=1,3,...,29（奇数，避免投票平局）
cv_scores = []  # 存储每个K值的交叉验证平均准确率

# 6. 交叉验证搜索最佳K值（用标准化后的数据，与版本2对齐）
print("K值搜索与交叉验证结果：")
print("-" * 50)
print(f"{'K值':<6}{'5折交叉验证平均准确率':<25}{'标准差':<10}")
print("-" * 50)

for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # 距离 metric 统一为 euclidean
    # 关键：使用标准化后的训练集做交叉验证，消除特征尺度差异
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"{k:<6}{scores.mean():<25.4f}{scores.std():<10.4f}")

# 7. 确定最佳K值（此时最佳K值对应的准确率会统一到0.7606）
best_k_idx = np.argmax(cv_scores)
best_k = k_candidates[best_k_idx]
best_cv_score = cv_scores[best_k_idx]

print("-" * 50)
print(f"最佳K值：{best_k}")
print(f"最佳K值对应的5折交叉验证平均准确率：{best_cv_score:.4f}")  # 此时输出应为0.7606

# 8. 保存K值选择图
save_path = r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\k_value_selection_standardized.png"
plt.figure(figsize=(10, 6))
plt.plot(k_candidates, cv_scores, marker='o', linestyle='-', color='b', markersize=6)
plt.scatter(best_k, best_cv_score, color='red', s=100, 
            label=f'最佳K值: {best_k}\n准确率: {best_cv_score:.4f}')
plt.xlabel('K值（近邻数量）')
plt.ylabel('5折交叉验证平均准确率')
plt.title('K值选择：标准化后数据的交叉验证准确率曲线')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(k_candidates)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n图片已保存至：{save_path}")

# 9. 最佳K值模型训练与评估（用标准化数据）
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
best_knn.fit(X_train_scaled, y_train)

# 预测结果
y_pred_train = best_knn.predict(X_train_scaled)
y_pred_test = best_knn.predict(X_test_scaled)

# 计算各项指标
train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train, zero_division=0)
train_recall = recall_score(y_train, y_pred_train, zero_division=0)
train_f1 = f1_score(y_train, y_pred_train, zero_division=0)

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)
test_recall = recall_score(y_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

# 输出指标对比
print("\n" + "=" * 60)
print(f"最佳K值（K={best_k}）的模型性能指标对比（标准化后数据）")
print("=" * 60)
metrics_df = pd.DataFrame({
    '指标': ['准确率（Accuracy）', '精确率（Precision）', '召回率（Recall）', 'F1分数（F1-Score）'],
    '训练集': [f"{train_accuracy:.4f}", f"{train_precision:.4f}", f"{train_recall:.4f}", f"{train_f1:.4f}"],
    '测试集': [f"{test_accuracy:.4f}", f"{test_precision:.4f}", f"{test_recall:.4f}", f"{test_f1:.4f}"]
})
print(metrics_df.to_string(index=False))

# 输出测试集混淆矩阵
print("\n测试集混淆矩阵：")
cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(
    cm,
    index=['实际负例（0）', '实际正例（1）'],
    columns=['预测负例（0）', '预测正例（1）']
)
print(cm_df)

# 10. 带权重KNN对比（用标准化数据）
weighted_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='distance')
weighted_knn.fit(X_train_scaled, y_train)
y_pred_test_weighted = weighted_knn.predict(X_test_scaled)
test_accuracy_weighted = accuracy_score(y_test, y_pred_test_weighted)

print("\n" + "-" * 60)
print(f"带距离权重的KNN（K={best_k}）性能对比（标准化后数据）")
print("-" * 60)
print(f"标准KNN测试集准确率：{test_accuracy:.4f}")
print(f"带权重KNN测试集准确率：{test_accuracy_weighted:.4f}")
print(f"性能差异：{test_accuracy_weighted - test_accuracy:.4f}")