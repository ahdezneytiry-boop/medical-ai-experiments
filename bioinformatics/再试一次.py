import pandas as pd
import matplotlib.pyplot as plt

# 仅保留负号显示设置（必须）
plt.rcParams['axes.unicode_minus'] = False

# 数据读取与统计（保持不变）
df = pd.read_excel("data/附件1.xlsx")
team_counts = df['车队编码'].value_counts().sort_index()

# 创建画布
plt.figure(figsize=(12, 6.5))

# 绘制柱状图（移除所有字体相关参数）
ax = team_counts.plot(
    kind='bar', 
    color='#2563EB',  # 蓝色系更专业
    edgecolor='white',
    width=0.7  # 调整柱子宽度
)

# 纯英文坐标轴（核心修改）
plt.xlabel('Team Code', fontsize=14, labelpad=12)
plt.ylabel('Number of Routes', fontsize=14, labelpad=15)

# 坐标刻度（英文优化）
plt.xticks(
    rotation=45, 
    ha='right', 
    fontsize=12,
    fontfamily='DejaVu Sans'  # 通用英文字体
)
plt.yticks(fontsize=12, fontfamily='DejaVu Sans')
plt.ylim(0, team_counts.max()*1.1)  # 顶部留白

# 数据标签（英文环境优化）
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f'{height}',
        (p.get_x() + p.get_width()/2, height),
        ha='center', 
        va='bottom',
        fontsize=12,
        color='white',
        fontfamily='DejaVu Sans',
        bbox=dict(facecolor='#1E293B', alpha=0.9, pad=4)
    )

# 英文标题
plt.title('Distribution of Routes by Team', fontsize=16, pad=20)

# 网格与布局（英文适配）
plt.grid(axis='y', linestyle='--', alpha=0.4, which='major')
plt.gcf().subplots_adjust(bottom=0.2)
plt.tight_layout()

plt.show()
