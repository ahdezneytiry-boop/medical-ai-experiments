import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 1. 中文字体与负号设置（保留原配置）
if plt.rcParams['font.family'] == 'sans-serif':
    # Windows优先用黑体，macOS用苹方，Linux用思源黑体
    if plt.get_backend() == 'MacOSX':
        rcParams['font.sans-serif'] = ['PingFang HK']
    elif 'Windows' in plt.get_backend():
        rcParams['font.sans-serif'] = ['SimHei']
    else:
        rcParams['font.sans-serif'] = ['Source Han Sans CN']
rcParams['axes.unicode_minus'] = False        # 负号显示

# 2. 数据读取（假设路径正确）
df = pd.read_excel("data/附件1.xlsx")

# 3. 数据统计（新增排序优化）
team_counts = df['车队编码'].value_counts().sort_index()  # 按车队编码升序排列

# 4. 创建画布（调整尺寸）
plt.figure(figsize=(12, 6.5))  # 宽度增加，避免横坐标拥挤

# 5. 绘制柱状图（获取绘图对象）
ax = team_counts.plot(
    kind='bar', 
    color='#EF4444',  # 更专业的红色（替代默认纯红）
    edgecolor='white'  # 增加柱子边框对比度
)
for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
    item.set_fontfamily(rcParams['font.sans-serif'][0])
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontfamily(rcParams['font.sans-serif'][0])
# 6. 横坐标优化（核心修改）
plt.xlabel('车队编码', fontsize=14, labelpad=12)  # 标签文字加大，间距调整
plt.xticks(
    rotation=45,       # 45度倾斜避免重叠
    ha='right',        # 右对齐
    fontsize=12,       # 字体加大
)
plt.gcf().subplots_adjust(bottom=0.2)  # 增加底部空间防止标签被截断

# 7. 纵坐标优化
plt.ylabel('线路数量（条）', fontsize=14, labelpad=15)  # 添加单位，间距调整
plt.yticks(fontsize=12)  # 刻度字体
plt.ylim(0, team_counts.max()*1.1)  # 顶部留白10%，避免柱子顶天

# 8. 添加数据标签（关键改进）
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f'{height}',          # 显示数值
        (p.get_x() + p.get_width()/2, height),  # 居中位置
        ha='center', va='bottom',  # 垂直底部对齐
        fontsize=12,
        color='white',       # 白色标签更显眼
        bbox=dict(          # 增加背景框
            facecolor='#2D3748', 
            alpha=0.8, 
            pad=4
        )
    )

# 9. 标题与整体优化
plt.title('各车队负责线路数量分布', fontsize=16, pad=20)  # 标题加大，间距调整
plt.grid(axis='y', linestyle='--', alpha=0.6, which='major')  # 优化网格透明度

# 10. 显示图形（新增tight_layout）
plt.tight_layout()  # 自动调整边距
plt.show()
