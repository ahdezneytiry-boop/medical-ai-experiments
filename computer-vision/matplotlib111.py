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

# 标题
plt.title("title")
# 横坐标描述
plt.xlabel('date')
# 纵坐标描述
plt.ylabel('num')

# 这里设置线宽、线型、线条颜色、点大小等参数
ax.plot(x, y, label='yy', linewidth=2, color='g', marker='o',
        markerfacecolor='red', markersize=4)
ax.plot(x, z, label='zz', linewidth=2, color='y', marker='o',
        markerfacecolor='black', markersize=4)
# 每个数据点加标签
for a, b in zip(x, z):
    plt.text(a, b, str(b) + '%', ha='center', va='bottom', fontsize=12)
# 只给最后一个点加标签
plt.text(x[-1], y[-1], y[-1], ha='center', va='bottom', fontsize=15)

# 旋转x轴标签
for label in ax.get_xticklabels():
    label.set_rotation(30)  # 旋转30度
    label.set_horizontalalignment('right')  # 向右旋转

# 图例显示及位置确定
plt.legend(loc='upper left')
plt.savefig(r'F:\计算机视觉\photo', bbox_inches='tight')
plt.show()