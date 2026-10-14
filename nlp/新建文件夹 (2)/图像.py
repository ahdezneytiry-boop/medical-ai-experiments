import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------
# 1. 4大新一线城市数据分析师薪资（低一档城市数据）
# --------------------------
da_salary_data = pd.DataFrame({
    "工作经验段": ["应届生", "1-3年", "3-5年", "5-10年", "10年以上"],
    "杭州": [7.8, 13.5, 19.8, 27.2, 35.5],  # 新一线中互联网产业强，薪资最高
    "成都": [7.2, 12.0, 17.5, 24.0, 30.8],
    "武汉": [7.0, 11.8, 17.2, 23.5, 29.8],
    "南京": [7.3, 12.2, 17.8, 24.5, 31.2]
})

# 基础配置（Windows适配+中文正常显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 城市颜色（4个新一线城市，颜色柔和且区分度高）
city_colors = {
    "杭州": "#FFB6C1",  # 浅粉色（对应互联网活力）
    "成都": "#98FB98",  # 浅绿色（对应生活氛围）
    "武汉": "#87CEFA",  # 浅蓝色（对应长江经济带）
    "南京": "#DDA0DD"   # 浅紫色（对应历史文化属性）
}
cities = ["杭州", "成都", "武汉", "南京"]  # 4个代表性新一线城市
experience = da_salary_data["工作经验段"].tolist()
x_pos = np.arange(len(experience))  # X轴：经验段位置（0-4）

# --------------------------
# 2. 绘制新一线城市3D瀑布图（清晰无密集）
# --------------------------
fig = plt.figure(figsize=(10, 7))  # 适配4个城市，Y轴间隔充足
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=26, azim=-45)  # 优化视角，4条线不重叠、不遮挡

# 遍历4个新一线城市绘制
for idx, city in enumerate(cities):
    salary = da_salary_data[city].values
    y_pos = np.full_like(x_pos, idx)  # Y轴均匀分布，无拥挤感
    
    # 1. 薪资趋势线（清晰体现增长梯度）
    ax.plot(x_pos, y_pos, salary, color=city_colors[city], linewidth=2.8, alpha=0.95)
    
    # 2. 薪资节点（突出每个经验段的薪资值）
    ax.scatter(x_pos, y_pos, salary, color=city_colors[city], s=90, alpha=1, edgecolors='white', linewidth=1.1)
    
    # 3. 精简标注（小字体+无背景，不影响视觉）
    for x, y, z in zip(x_pos, y_pos, salary):
        ax.text(
            x, y, z + 0.4, 
            f"{z:.1f}k", 
            ha='center', va='bottom', 
            fontsize=7.8, 
            fontweight='normal'
        )
    
    # 4. 瀑布立面（低透明度，通透不厚重）
    for i in range(len(x_pos)-1):
        verts = [
            [x_pos[i], y_pos[i], 0],
            [x_pos[i+1], y_pos[i+1], 0],
            [x_pos[i+1], y_pos[i+1], salary[i+1]],
            [x_pos[i], y_pos[i], salary[i]],
            [x_pos[i], y_pos[i], 0]
        ]
        poly = Poly3DCollection([verts], alpha=0.22, facecolor=city_colors[city])
        ax.add_collection3d(poly)

# --------------------------
# 3. 图表美化+Windows桌面保存
# --------------------------
# 坐标轴标签（明确“新一线城市”主题）
ax.set_xlabel("工作经验段", fontsize=10.5, fontweight='bold', labelpad=10)
ax.set_ylabel("4大新一线城市", fontsize=10.5, fontweight='bold', labelpad=10)
ax.set_zlabel("平均薪资（k/月）", fontsize=10.5, fontweight='bold', labelpad=10)

# 刻度标签（适配新一线城市）
ax.set_xticks(x_pos)
ax.set_xticklabels(experience, fontsize=8.2)
ax.set_yticks(np.arange(len(cities)))
ax.set_yticklabels(cities, fontsize=8.2)

# Z轴范围（贴合新一线城市薪资水平，无多余空白）
max_salary = da_salary_data[cities].values.max()
ax.set_zlim(0, max_salary * 1.07)

# 图表标题（突出“低一档城市”定位）
ax.set_title("数据分析师岗位-4大新一线城市薪资3D瀑布图（低一档城市版）", fontsize=12, fontweight='bold', pad=12)

# 图例（位置优化，不遮挡图表）
legend_elements = [plt.Line2D([0], [0], color=city_colors[city], lw=3, label=city) for city in cities]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.97, 0.97))

# 保存到桌面（修改“ahdezneytiry-boop1”为你的Windows用户名）
plt.tight_layout()
save_path = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\数据分析师薪资3D图_新一线城市版.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()

# 运行提示
print(f"✅ 新一线城市版图表已保存到：{save_path}")
print("💡 关键差异：比一线城市薪资低15%-25%，杭州因互联网产业薪资最接近一线城市")