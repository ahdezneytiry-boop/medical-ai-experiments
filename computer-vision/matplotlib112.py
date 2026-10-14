import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = ['2019-05-01', '2019-05-02', '2019-05-03', '2019-05-04', '2019-05-05',
     '2019-05-06', '2019-05-07', '2019-05-08', '2019-05-09', '2019-05-10']
y = [1, 1, 2, 3, 4, 6, 7, 9, 11, 16]
z = [1, 2, 3, 4, 7, 9, 12, 13, 15, 20]

# 设置画布大小
plt.figure(figsize=(20, 4))

tick_spacing = 4
fig, ax = plt.subplots(1, 1)
ax.xaxis.set_major_locator(ticker.MaxNLocator(tick_spacing))

# 标题和轴标签
plt.title("title")
plt.xlabel('date')
plt.ylabel('num')

# 绘制线条
ax.plot(x, y, label='yy', linewidth=2, color='g', marker='o',
        markerfacecolor='red', markersize=4)
ax.plot(x, z, label='zz', linewidth=2, color='y', marker='o',
        markerfacecolor='black', markersize=4)

# 给 z 数据的每个点加标签（原逻辑保留）
for a, b in zip(x, z):
    plt.text(a, b, str(b) + '%', ha='center', va='bottom', fontsize=12)

# 新增：给 y 数据的每个点加标签（模仿 z 的逻辑）
for a, b in zip(x, y):
    plt.text(a, b, str(b), ha='center', va='top', fontsize=12)  # va='top' 避免和点重叠

# 旋转x轴标签
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_horizontalalignment('right')

# 图例和显示
plt.legend(loc='upper left')
plt.savefig(r'F:\计算机视觉\photo', bbox_inches='tight')
plt.show()