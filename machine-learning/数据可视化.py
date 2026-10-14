'''import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 整理K值搜索核心数据
K_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
cv_accuracy = [0.6807, 0.7084, 0.7183, 0.7281, 0.7362, 0.7427, 0.7281, 0.7199, 0.7248, 0.7183, 0.7101, 0.7101, 0.7166, 0.7150, 0.7150]
std_dev = [0.0325, 0.0137, 0.0136, 0.0279, 0.0233, 0.0347, 0.0289, 0.0278, 0.0287, 0.0205, 0.0230, 0.0147, 0.0197, 0.0251, 0.0261]

# 创建图表，沿用模板风格
fig, ax1 = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#f0f8ff')

# 定义柱状图颜色（最佳K值用特殊颜色突出）
bar_colors = ['#a6d1e6'] * len(K_values)
best_k_index = K_values.index(11)
bar_colors[best_k_index] = '#ff7f7f'  # 最佳K值用红色突出

# 主轴：柱状图展示交叉验证平均准确率
ax1.bar(K_values, cv_accuracy, color=bar_colors, zorder=2, edgecolor='#333', linewidth=0.5)
ax1.set_xlabel('K值', fontsize=11)
ax1.set_ylabel('5折交叉验证平均准确率', fontsize=11)
ax1.set_ylim(0.65, 0.78)  # 调整y轴范围，突出差异
ax1.set_facecolor('#f0f8ff')

# 次轴：折线图展示标准差
ax2 = ax1.twinx()
ax2.plot(K_values, std_dev, 
         marker='o', 
         color='#1e88e5', 
         markerfacecolor='white',
         markeredgecolor='#1e88e5',
         markeredgewidth=2,
         linewidth=2, 
         zorder=3)
ax2.set_ylabel('标准差', fontsize=11)
ax2.set_ylim(0.01, 0.04)  # 标准差范围
ax2.set_facecolor('#f0f8ff')

# 图表标题和标注
plt.title('KNN模型K值搜索与交叉验证结果（最佳K=11）', fontsize=14, pad=20)
ax1.set_xticks(K_values)
ax1.tick_params(axis='x', rotation=45)  # x轴标签旋转，避免重叠

# 图例
ax1.legend(['交叉验证平均准确率'], loc='upper left', frameon=False)
ax2.legend(['标准差'], loc='upper right', frameon=False)

# 隐藏顶部边框，添加网格
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.grid(axis='y', zorder=1, alpha=0.3)

# 在最佳K值位置添加标注
ax1.annotate('最佳K=11\n准确率=0.7427', 
             xy=(11, 0.7427), 
             xytext=(11, 0.755),
             ha='center',
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='#ff5555', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0', edgecolor='#ff5555', alpha=0.8))

plt.tight_layout()
plt.savefig(r'C:\Users\ahdezneytiry-boop1\Desktop\实验三\k_value_analysis.png', dpi=300, bbox_inches='tight', facecolor='#f0f8ff')
plt.show()
'''



import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

dim_types = [' 原始维度（8 维）', 'PCA 最佳维度（3 维）']
accuracy = [0.7078, 0.7273]
precision = [0.5962, 0.6140]
recall = [0.5636, 0.6364]
f1 = [0.5794, 0.6250]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#f0f8ff')

ax.plot (dim_types, accuracy, marker='o', color='#1e88e5', markerfacecolor='white', markeredgecolor='#1e88e5', linewidth=2, label=' 准确率 ')
ax.plot (dim_types, precision, marker='s', color='#ff7f0e', markerfacecolor='white', markeredgecolor='#ff7f0e', linewidth=2, label=' 精确率 ')
ax.plot (dim_types, recall, marker='^', color='#2ca02c', markerfacecolor='white', markeredgecolor='#2ca02c', linewidth=2, label=' 召回率 ')
ax.plot (dim_types, f1, marker='d', color='#9467bd', markerfacecolor='white', markeredgecolor='#9467bd', linewidth=2, label='F1 分数 ')

ax.set_xlabel (' 维度类型 ')
ax.set_ylabel (' 性能指标值 ')
ax.set_title (' 原始维度与 PCA 最佳维度的模型性能对比 ')
ax.set_ylim (0.5, 0.75)

ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, txt in enumerate(accuracy):ax.annotate(f'{txt:.4f}', (dim_types[i], accuracy[i]), textcoords="offset points", xytext=(0, 10), ha='center')
for i, txt in enumerate(precision):ax.annotate(f'{txt:.4f}', (dim_types[i], precision[i]), textcoords="offset points", xytext=(0, 10), ha='center')
for i, txt in enumerate(recall):ax.annotate(f'{txt:.4f}', (dim_types[i], recall[i]), textcoords="offset points", xytext=(0, 10), ha='center')
for i, txt in enumerate(f1):ax.annotate(f'{txt:.4f}', (dim_types[i], f1[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.tight_layout ()
plt.savefig (r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\dim_performance_comparison.png", dpi=300, bbox_inches='tight', facecolor='#f0f8ff')
plt.show ()