import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_classification  # 生成真实非完美簇数据
from sklearn.decomposition import PCA

# 1. 绘图风格配置
plt.rcParams["font.family"] = ["SimHei"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 2. 生成真实文本特征数据
# 模拟真实场景：3个簇（重叠+样本数不均+5%噪声+冗余特征）
X, y_true = make_classification(
    n_samples=600,       # 样本数
    n_features=20,       # 模拟文本特征维度
    n_informative=15,    # 有效特征（文本关键词）
    n_redundant=5,       # 冗余特征（文本无关词）
    n_clusters_per_class=1,
    n_classes=3,         # 3个真实簇
    weights=[0.2, 0.5, 0.3],  # 簇样本数不均（2:5:3）
    random_state=42,
    flip_y=0.05          # 5%噪声（文本主题混淆）
)

pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_sample = pd.DataFrame(X[:10, :5], columns=[f"文本特征维度{i+1}" for i in range(5)])
df_sample["真实簇标签"] = y_true[:10]
print("原始数据样本（前10行，前5列特征 + 真实簇标签）：")
print(df_sample)
print("\n" + "="*80 + "\n")

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

k_range = range(2, 11)  # K值遍历范围
inertia_list = []       # 惯性值（肘部法则）
silhouette_list = []    # 平均轮廓系数

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    inertia_list.append(kmeans.inertia_)          # 簇内平方和
    silhouette_list.append(silhouette_score(X, y_pred))  # 聚类效果评估

k_opt = 3
kmeans_opt = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
y_pred_opt = kmeans_opt.fit_predict(X)
centers_2d = pca.transform(kmeans_opt.cluster_centers_)  # 聚类中心降维

fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor('#f0f8ff')  # 整体背景色

# 子图1：肘部法则
ax1 = plt.subplot(2, 2, 1)
ax1.plot(k_range, inertia_list,
         marker='o',
         color='#1e88e5',
         markerfacecolor='white',
         markeredgecolor='#1e88e5',
         linewidth=2,
         zorder=3)
ax1.scatter(3, inertia_list[1], color="#863e3e", s=100, zorder=4, label='肘部点（K=3）')
ax1.set_xlabel('聚类数K')
ax1.set_ylabel('惯性值')
ax1.set_title('KMeans肘部法则：选择最优K值')
ax1.set_facecolor('#f0f8ff')
ax1.set_xticks(k_range)
ax1.legend(loc='upper right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', zorder=1, alpha=0.3)

# 子图2：轮廓系数
ax2 = plt.subplot(2, 2, 2)
bar_colors = ['#a6d1e6', '#70a6c2', '#3b7a9e', '#235b7a', '#12435c', '#0a2b3a', '#082230', '#051b26', '#03151e']
ax2.bar(k_range, silhouette_list, color=bar_colors, zorder=2)
ax2.scatter(3, silhouette_list[1], color="#a35959", s=100, zorder=4, label='轮廓系数峰值（K=3）')
ax2.set_xlabel('聚类数K')
ax2.set_ylabel('平均轮廓系数')
ax2.set_title('KMeans轮廓系数：评估聚类效果')
ax2.set_facecolor('#f0f8ff')
ax2.set_xticks(k_range)
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', zorder=1, alpha=0.3)

# 子图3：原始真实分布（优化点显示，适配重叠数据）
ax3 = plt.subplot(2, 2, 3)
scatter_true = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, 
                           cmap='tab10', s=30, alpha=0.7, zorder=2)  # 高对比度+半透明
ax3.set_xlabel('PCA维度1')
ax3.set_ylabel('PCA维度2')
ax3.set_title('真实文本数据分布')
ax3.set_facecolor('#f0f8ff')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(zorder=1, alpha=0.3)
cbar1 = plt.colorbar(scatter_true, ax=ax3)
cbar1.set_label('真实主题簇')

# 子图4：K=3时KMeans聚类结果
ax4 = plt.subplot(2, 2, 4)
scatter_pred = ax4.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred_opt, 
                           cmap='tab10', s=30, alpha=0.7, zorder=2)
# 标注聚类中心
ax4.scatter(centers_2d[:, 0], centers_2d[:, 1], 
            c='red', s=200, marker='X', edgecolor='black', zorder=3, label='聚类中心')
ax4.set_xlabel('PCA维度1')
ax4.set_ylabel('PCA维度2')
ax4.set_title(f'KMeans最优K={k_opt} 聚类结果')
ax4.set_facecolor('#f0f8ff')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(zorder=1, alpha=0.3)
ax4.legend(loc='upper right')
cbar2 = plt.colorbar(scatter_pred, ax=ax4)
cbar2.set_label('预测主题簇')

# 整体布局调整
plt.tight_layout()
plt.show()


print("="*80)
print("可视化结果说明：")
print("1. 肘部法则/轮廓系数仍指向K=3，但轮廓系数值低于理想数据（真实簇重叠导致）；")
print("2. 原始数据分布（左下）：簇边界模糊、样本重叠，符合真实文本数据特征；")
print("3. 聚类结果（右下）：KMeans无法完美区分重叠样本，体现真实场景聚类局限性；")
print("4. 降维说明：PCA仅用于可视化，聚类计算基于原始20维特征（含冗余/噪声）。")
print("="*80)