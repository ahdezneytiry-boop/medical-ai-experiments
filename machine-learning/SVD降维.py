# 1. 导入必要库（仅保留SVD相关核心库）
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import TruncatedSVD  # 仅保留SVD相关库
from sklearn.preprocessing import StandardScaler

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 2. 数据读取与预处理（核心支撑步骤，不可少）
data_path = r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\diabetes (1).csv"
data = pd.read_csv(data_path)

label_need = data.keys()
X = data[label_need].values[:, 0:8]  # 8个特征
y = data[label_need].values[:, 8].astype(int)  # 标签

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化（与SVD降维保持一致）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. K值搜索与最佳K值确定（SVD建模需依赖最佳K值，保留核心流程）
k_candidates = range(1, 31, 2)
cv_scores = []

print("K值搜索与交叉验证结果（支撑SVD建模）：")
print("-" * 50)
print(f"{'K值':<6}{'5折交叉验证平均准确率':<25}{'标准差':<10}")
print("-" * 50)

for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"{k:<6}{scores.mean():<25.4f}{scores.std():<10.4f}")

best_k_idx = np.argmax(cv_scores)
best_k = k_candidates[best_k_idx]
best_cv_score = cv_scores[best_k_idx]

print("-" * 50)
print(f"最佳K值：{best_k}（用于SVD降维后建模）")
print(f"最佳K值对应的5折交叉验证平均准确率：{best_cv_score:.4f}")

# 4. SVD降维核心流程（保留完整核心输出）
print("\n" + "=" * 70)
print("【SVD核心输出】不同维度下的测试集性能对比")
print("=" * 70)

svd_dimensions = range(1, 9)  # 测试维度1-8
svd_performance = []

for d in svd_dimensions:
    # SVD降维（仅用训练集拟合，避免数据泄露）
    svd = TruncatedSVD(n_components=d, random_state=42)
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)
    
    # 用最佳K值训练KNN模型
    knn_svd = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn_svd.fit(X_train_svd, y_train)
    
    # 性能计算
    y_pred_svd = knn_svd.predict(X_test_svd)
    accuracy = accuracy_score(y_test, y_pred_svd)
    precision = precision_score(y_test, y_pred_svd, zero_division=0)
    recall = recall_score(y_test, y_pred_svd, zero_division=0)
    f1 = f1_score(y_test, y_pred_svd, zero_division=0)
    explained_variance = svd.explained_variance_ratio_.sum()
    
    svd_performance.append({
        '维度d': d,
        '累计解释方差': round(explained_variance, 4),
        '准确率': round(accuracy, 4),
        '精确率': round(precision, 4),
        '召回率': round(recall, 4),
        'F1分数': round(f1, 4)
    })

# 输出SVD核心性能表格
svd_df = pd.DataFrame(svd_performance)
print(svd_df.to_string(index=False))

# 5. SVD核心可视化（维度-性能关系图）
save_path_svd = r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\svd_dimension_performance.png"
plt.figure(figsize=(12, 6))

# 子图1：SVD维度与各项性能指标（核心可视化）
ax1 = plt.subplot(1, 2, 1)
metrics = ['准确率', '精确率', '召回率', 'F1分数']
for metric in metrics:
    ax1.plot(svd_dimensions, svd_df[metric], marker='o', linewidth=2, label=metric)
ax1.set_xlabel('SVD降维后的维度d')
ax1.set_ylabel('性能指标值')
ax1.set_title('SVD维度与测试集性能指标关系')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(svd_dimensions)

# 子图2：SVD维度与累计解释方差（核心可视化）
ax2 = plt.subplot(1, 2, 2)
ax2.plot(svd_dimensions, svd_df['累计解释方差'], marker='s', color='green', linewidth=2)
ax2.set_xlabel('SVD降维后的维度d')
ax2.set_ylabel('累计解释方差比例')
ax2.set_title('SVD维度与累计解释方差关系')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(svd_dimensions)
ax2.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(save_path_svd, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSVD核心可视化图已保存至：{save_path_svd}")

# 6. SVD最佳维度与原始维度对比（核心对比输出）
print("\n" + "=" * 70)
print("【SVD核心对比】最佳维度 vs 原始维度")
print("=" * 70)

# 原始维度（8维）性能
best_knn_original = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
best_knn_original.fit(X_train_scaled, y_train)
y_pred_original = best_knn_original.predict(X_test_scaled)
original_metrics = {
    '方法': '原始维度（8维）',
    '维度d': 8,
    '准确率': round(accuracy_score(y_test, y_pred_original), 4),
    '精确率': round(precision_score(y_test, y_pred_original, zero_division=0), 4),
    '召回率': round(recall_score(y_test, y_pred_original, zero_division=0), 4),
    'F1分数': round(f1_score(y_test, y_pred_original, zero_division=0), 4)
}

# SVD最佳维度性能（按准确率排序）
best_svd_idx = svd_df['准确率'].idxmax()
best_svd_metrics = {
    '方法': f'SVD最佳维度（{svd_df.loc[best_svd_idx, "维度d"]}维）',
    '维度d': svd_df.loc[best_svd_idx, '维度d'],
    '准确率': svd_df.loc[best_svd_idx, '准确率'],
    '精确率': svd_df.loc[best_svd_idx, '精确率'],
    '召回率': svd_df.loc[best_svd_idx, '召回率'],
    'F1分数': svd_df.loc[best_svd_idx, 'F1分数']
}

# 输出核心对比表格
comparison_df = pd.DataFrame([original_metrics, best_svd_metrics])
print(comparison_df.to_string(index=False))