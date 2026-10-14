# 1. 导入相关库
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA  # 新增PCA库
from sklearn.preprocessing import StandardScaler  # 新增标准化库

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 2. 数据读取
data_path = "data/diabetes (1).csv"
data = pd.read_csv(data_path)

# 3. 数据预处理
label_need = data.keys()
X = data[label_need].values[:, 0:8]  # 前8列特征
y = data[label_need].values[:, 8].astype(int)  # 第9列标签

# 4. 训练集/测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 新增：数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集拟合+标准化
X_test_scaled = scaler.transform(X_test)  # 测试集仅用训练集参数标准化（避免数据泄露）

# 5. K值搜索范围（保持不变）
k_candidates = range(1, 31, 2)
cv_scores = []

# 6. 交叉验证搜索最佳K值
print("K值搜索与交叉验证结果：")
print("-" * 50)
print(f"{'K值':<6}{'5折交叉验证平均准确率':<25}{'标准差':<10}")
print("-" * 50)

for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')  # 用标准化数据
    cv_scores.append(scores.mean())
    print(f"{k:<6}{scores.mean():<25.4f}{scores.std():<10.4f}")

# 7. 确定最佳K值
best_k_idx = np.argmax(cv_scores)
best_k = k_candidates[best_k_idx]
best_cv_score = cv_scores[best_k_idx]

print("-" * 50)
print(f"最佳K值：{best_k}")
print(f"最佳K值对应的5折交叉验证平均准确率：{best_cv_score:.4f}")

# 8. K值选择可视化
save_path_k = r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\k_value_selection.png"
plt.figure(figsize=(10, 6))
plt.plot(k_candidates, cv_scores, marker='o', linestyle='-', color='b', markersize=6)
plt.scatter(best_k, best_cv_score, color='red', s=100, 
            label=f'最佳K值: {best_k}\n准确率: {best_cv_score:.4f}')
plt.xlabel('K值（近邻数量）')
plt.ylabel('5折交叉验证平均准确率')
plt.title('K值选择：交叉验证准确率曲线')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(k_candidates)
plt.savefig(save_path_k, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nK值选择图已保存至：{save_path_k}")

print("\n" + "=" * 70)
print("PCA降维：不同维度下的测试集性能对比")
print("=" * 70)

# 定义要测试的降维维度
pca_dimensions = range(1, 9)  # 维度d=1,2,...,8
pca_performance = []  # 存储各维度性能指标

# 对每个维度d进行PCA降维+KNN训练+性能评估
for d in pca_dimensions:
    # 1. PCA降维（仅用训练集拟合PCA，避免数据泄露）
    pca = PCA(n_components=d, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)  # 标准化后的数据降维
    X_test_pca = pca.transform(X_test_scaled)
    
    # 2. 用最佳K值训练KNN模型
    knn_pca = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn_pca.fit(X_train_pca, y_train)
    
    # 3. 测试集预测与性能计算
    y_pred_pca = knn_pca.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred_pca)
    precision = precision_score(y_test, y_pred_pca, zero_division=0)
    recall = recall_score(y_test, y_pred_pca, zero_division=0)
    f1 = f1_score(y_test, y_pred_pca, zero_division=0)
    
    # 4. 存储结果（含PCA解释方差比例，反映维度保留的信息）
    explained_variance = pca.explained_variance_ratio_.sum()  # 前d维累计解释方差
    pca_performance.append({
        '维度d': d,
        '累计解释方差': round(explained_variance, 4),
        '准确率': round(accuracy, 4),
        '精确率': round(precision, 4),
        '召回率': round(recall, 4),
        'F1分数': round(f1, 4)
    })

# 转换为DataFrame并输出表格
pca_df = pd.DataFrame(pca_performance)
print(pca_df.to_string(index=False))

save_path_pca = r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\pca_dimension_performance.png"
plt.figure(figsize=(12, 6))

# 子图1：维度与各项性能指标
ax1 = plt.subplot(1, 2, 1)
metrics = ['准确率', '精确率', '召回率', 'F1分数']
for metric in metrics:
    ax1.plot(pca_dimensions, pca_df[metric], marker='o', linewidth=2, label=metric)
ax1.set_xlabel('PCA降维后的维度d')
ax1.set_ylabel('性能指标值')
ax1.set_title('PCA维度与测试集性能指标关系')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(pca_dimensions)

# 子图2：维度与累计解释方差
ax2 = plt.subplot(1, 2, 2)
ax2.plot(pca_dimensions, pca_df['累计解释方差'], marker='s', color='orange', linewidth=2)
ax2.set_xlabel('PCA降维后的维度d')
ax2.set_ylabel('累计解释方差比例')
ax2.set_title('PCA维度与累计解释方差关系')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(pca_dimensions)
ax2.set_ylim(0.5, 1.0)  # 限定y轴范围，更清晰

plt.tight_layout()
plt.savefig(save_path_pca, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nPCA维度-性能图已保存至：{save_path_pca}")

print("\n" + "=" * 60)
print(f"原始维度（8维）最佳K值（K={best_k}）的模型性能")
print("=" * 60)

# 用标准化后的原始数据（8维）训练模型
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
best_knn.fit(X_train_scaled, y_train)
y_pred_test = best_knn.predict(X_test_scaled)

# 计算原始维度测试集指标
test_accuracy_original = accuracy_score(y_test, y_pred_test)
test_precision_original = precision_score(y_test, y_pred_test, zero_division=0)
test_recall_original = recall_score(y_test, y_pred_test, zero_division=0)
test_f1_original = f1_score(y_test, y_pred_test, zero_division=0)

# 输出原始维度与最佳PCA维度的对比
print("原始维度（8维）测试集性能：")
print(f"准确率：{test_accuracy_original:.4f} | 精确率：{test_precision_original:.4f}")
print(f"召回率：{test_recall_original:.4f} | F1分数：{test_f1_original:.4f}")

# 找出PCA最佳维度（按准确率排序）
best_pca_idx = pca_df['准确率'].idxmax()
best_pca_d = pca_df.loc[best_pca_idx, '维度d']
print(f"\nPCA最佳维度：{best_pca_d}维（准确率：{pca_df.loc[best_pca_idx, '准确率']:.4f}）")
print(f"该维度累计解释方差：{pca_df.loc[best_pca_idx, '累计解释方差']:.4f}")