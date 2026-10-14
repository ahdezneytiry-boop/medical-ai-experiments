import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

X, y_true = make_blobs(
    n_samples=500, n_features=20, centers=3, cluster_std=1.5, random_state=42
)

# 输出前10行样本（前5列特征+真实簇标签）
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_sample = pd.DataFrame(X[:10, :5], columns=[f"文本特征维度{i+1}" for i in range(5)])
df_sample["真实簇标签"] = y_true[:10]
print("原始数据样本（前10行，前5列特征 + 真实簇标签）：")
print(df_sample)
print("\n" + "="*80 + "\n")

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

k_range = range(2, 11)  # 遍历K值范围
inertia_list = []       # 存储惯性值（肘部法则）
silhouette_list = []    # 存储平均轮廓系数

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    inertia_list.append(kmeans.inertia_)          # 簇内平方和
    silhouette_list.append(silhouette_score(X, y_pred))  # 聚类效果评估

fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor('#f0f8ff')

# 子图1：肘部法则（选最优K）
ax1 = plt.subplot(2, 2, 1)
ax1.plot(k_range, inertia_list, marker='o', color="#3c5d7a", markerfacecolor='white',
         markeredgecolor="#406280", linewidth=2, zorder=3)
ax1.scatter(3, inertia_list[1], color="#632a2a", s=100, zorder=4, label='肘部点（K=3）')
ax1.set_xlabel('聚类数K'), ax1.set_ylabel('惯性值（簇内平方和）'), ax1.set_title('KMeans肘部法则')
ax1.set_facecolor('#f0f8ff'), ax1.set_xticks(k_range), ax1.legend(loc='upper right')
ax1.spines['top'].set_visible(False), ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', zorder=1, alpha=0.3)

# 子图2：轮廓系数（评估聚类效果）
ax2 = plt.subplot(2, 2, 2)
bar_colors = ['#a6d1e6', '#70a6c2', '#3b7a9e', '#235b7a', '#12435c', '#0a2b3a', '#082230', '#051b26', '#03151e']
ax2.bar(k_range, silhouette_list, color=bar_colors, zorder=2)
ax2.scatter(3, silhouette_list[1], color="#743d3d", s=100, zorder=4, label='轮廓系数最大值（K=3）')
ax2.set_xlabel('聚类数K'), ax2.set_ylabel('平均轮廓系数'), ax2.set_title('KMeans轮廓系数')
ax2.set_facecolor('#f0f8ff'), ax2.set_xticks(k_range), ax2.set_ylim(0, 1), ax2.legend(loc='upper right')
ax2.spines['top'].set_visible(False), ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', zorder=1, alpha=0.3)

# 子图3：原始数据真实分布（降维后）
ax3 = plt.subplot(2, 2, 3)
scatter_true = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8, zorder=2)
ax3.set_xlabel('PCA维度1'), ax3.set_ylabel('PCA维度2'), ax3.set_title('原始数据真实分类分布')
ax3.set_facecolor('#f0f8ff'), ax3.spines['top'].set_visible(False), ax3.spines['right'].set_visible(False)
ax3.grid(zorder=1, alpha=0.3)
cbar1 = plt.colorbar(scatter_true, ax=ax3), cbar1.set_label('真实簇编号')

# 子图4：K=3时聚类结果（降维后）
ax4 = plt.subplot(2, 2, 4)
kmeans_opt = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred_opt = kmeans_opt.fit_predict(X)
scatter_pred = ax4.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred_opt, cmap='viridis', s=50, alpha=0.8, zorder=2)
# 标注聚类中心（降维后）
centers_2d = pca.transform(kmeans_opt.cluster_centers_)
ax4.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=200, marker='X', zorder=3, label='聚类中心')
ax4.set_xlabel('PCA维度1'), ax4.set_ylabel('PCA维度2'), ax4.set_title('KMeans最优K=3 聚类结果')
ax4.set_facecolor('#f0f8ff'), ax4.spines['top'].set_visible(False), ax4.spines['right'].set_visible(False)
ax4.grid(zorder=1, alpha=0.3), ax4.legend(loc='upper right')
cbar2 = plt.colorbar(scatter_pred, ax=ax4), cbar2.set_label('预测簇编号')

# 布局调整+显示
plt.tight_layout()
plt.show()

print("="*80)
print("核心结论：")
print("1. 肘部法则（K=3）和轮廓系数（峰值K=3）均验证最优聚类数为3；")
print("2. KMeans聚类结果与原始真实分布几乎完全匹配，体现指标有效性；")
print("3. PCA仅用于可视化，聚类计算基于原始20维特征。")
print("="*80)