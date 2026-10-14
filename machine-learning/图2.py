import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 中文显示与学术风格全局配置
plt.rcParams["font.family"] = ["SimHei", "Times New Roman"]  # 中英混排支持
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

# 学术配色（柔和蓝色系，专业美观）
color_main = "#81BCEC"  # 钢蓝色（主色）
color_secondary = "#959DDB"  # 天蓝色（辅助色）
color_excellent = "#1A5276"  # 深蓝（优秀标准标注）

# ===================== 1. 模型性能指标图 =====================
fig, ax1 = plt.subplots(figsize=(7, 5))

metrics = ['准确率', 'F1分数', 'ROC-AUC']
values = [0.7904, 0.8039, 0.8683]
excellent_thresholds = [0.80, 0.80, 0.85]

x = np.arange(len(metrics))
width = 0.5

# 柱状图：性能指标
bars = ax1.bar(x, values, width, color=color_secondary, edgecolor='black', linewidth=0.5)

# 标注优秀标准线
for i, thresh in enumerate(excellent_thresholds):
    ax1.axhline(thresh, xmin=i/len(metrics), xmax=(i+1)/len(metrics), 
                color=color_excellent, linestyle='--', linewidth=1.5, 
                label=f'优秀标准: ≥{thresh}' if i==0 else "")
    ax1.text(x[i], thresh + 0.005, f'≥{thresh}', ha='center', va='bottom', 
             color=color_excellent, fontweight='bold')

# 数值标注
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, 
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')

# 轴与图例设置
ax1.set_xlabel('性能指标', fontsize=12)
ax1.set_ylabel('数值（0-1）', fontsize=12)
ax1.set_ylim(0.75, 0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)

ax1.legend(frameon=False)
ax1.set_title('随机森林模型性能指标（AMES致突变性预测）', fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('rf_performance_metrics.png', bbox_inches='tight')
plt.close()

# ===================== 2. 混淆矩阵图 =====================
fig, ax = plt.subplots(figsize=(6, 5))

rf_cm = np.array([[659, 159], [224, 785]])
labels = ['非致突变（0）', '致突变（1）']

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=labels, yticklabels=labels,
            cbar=False, linewidths=0.5, linecolor='black', 
            annot_kws={'fontweight': 'bold'})

ax.set_title('随机森林混淆矩阵（AMES致突变性预测）', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('预测标签', fontsize=11)
ax.set_ylabel('真实标签', fontsize=11)

plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', bbox_inches='tight')
plt.close()

# ===================== 3. 特征重要性Top5图 =====================
fig, ax = plt.subplots(figsize=(8, 4))

feature_names = ['指纹位725', '指纹位171', '指纹位960', '指纹位979', '指纹位753']
importances = [0.0145, 0.0111, 0.0103, 0.0093, 0.0091]

x = np.arange(len(feature_names))
width = 0.6

bars = ax.bar(x, importances, width, color=color_secondary, edgecolor='black', linewidth=0.5)

# 数值标注
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005, 
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Morgan指纹位', fontsize=12)
ax.set_ylabel('特征重要性', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(feature_names, fontsize=10, rotation=45, ha='right')
ax.set_title('随机森林Top5特征重要性（AMES致突变性预测）', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('rf_feature_importance_top5.png', bbox_inches='tight')
plt.close()

print("三张学术风格图表已生成：")
print("1. 性能指标图：rf_performance_metrics.png")
print("2. 混淆矩阵图：rf_confusion_matrix.png")
print("3. 特征重要性Top5图：rf_feature_importance_top5.png")