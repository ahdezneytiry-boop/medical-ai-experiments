import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns

# -------------------------- 1. 基础设置与数据准备（保留所有字符串列） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei')

# 读取你的原始数据（此处用示例数据，替换为你的数据路径即可）
# 若你的数据是Excel，用 data = pd.read_excel("你的文件路径.xls")
data = pd.DataFrame({
    "姓名": ["小红", "小明", "小刚", "小丽", "小强", "小美", "小亮", "小雅"],
    "学历": ["本科", "本科", "硕士", "本科", "硕士", "本科", "硕士", "本科"],
    "学校": ["A", "B", "A", "B", "A", "B", "A", "B"],
    "成绩": [85, 78, 92, 88, 95, 76, 90, 82],
    "能力": [7, 5, 9, 8, 10, 4, 8, 6]
})
print("=== 原始数据（含所有字符串列） ===")
print(data)


# -------------------------- 2. 生成4个关键数值列（与你的数据匹配） --------------------------
# 1. 离散字符串列（学历、学校）生成哑变量（确保列名与后续一致）
dummies = pd.get_dummies(data[["学历", "学校"]], prefix=["学历", "学校"])
# 2. 连续数值列（成绩、能力）标准化
scaler = preprocessing.StandardScaler()
scaled_cont = scaler.fit_transform(data[["成绩", "能力"]])
scaled_cont = pd.DataFrame(scaled_cont, columns=["标准化成绩", "标准化能力"], index=data.index)

# 3. 合并数据（保留所有原始列 + 4个关键数值列）
data_final = pd.concat([data, dummies, scaled_cont], axis=1)
# 确认4个关键数值列存在（若你的列名不同，需修改此处！）
four_key_cols = ["学历_本科", "学校_A", "标准化成绩", "标准化能力"]
print("\n=== 确认4个关键数值列已存在 ===")
print("当前数据中的关键列：", [col for col in four_key_cols if col in data_final.columns])


# -------------------------- 3. 修复四格热力图（核心：构建2x2矩阵） --------------------------
def four_grid_heatmap(df):
    # 步骤1：明确划分“分类哑变量列”和“连续变量列”（各2列，共4列）
    cat_cols = ["学历_本科", "学校_A"]  # 2个分类相关的数值列
    cont_cols = ["标准化成绩", "标准化能力"]  # 2个连续数值列
    
    # 步骤2：计算2x2相关性矩阵（分类列 vs 连续列）
    # 用corr()计算所有4列的相关性，再筛选出“分类列×连续列”的交叉部分
    full_corr = df[cat_cols + cont_cols].corr()  # 先算4列完整相关性
    cross_corr = full_corr.loc[cat_cols, cont_cols]  # 筛选出2行（分类列）×2列（连续列）的矩阵
    
    # 步骤3：绘制四格热力图（此时矩阵是标准2x2，无报错）
    plt.subplots(figsize=(6, 4))  # 适配四格的尺寸
    sns.heatmap(
        cross_corr,
        annot=True,        # 显示具体相关系数
        vmax=1, vmin=-1,   # 数值范围（-1到1，符合相关性定义）
        square=True,       # 强制正方形格子，美观
        cmap="RdBu_r",     # 红蓝色系（正相关红，负相关蓝，易区分）
        fmt=".3f",         # 保留3位小数，清晰
        cbar_kws={"shrink": 0.8}  # 调整颜色条大小，避免拥挤
    )
    plt.title("热力图", fontsize=12)
    plt.tight_layout()  # 自动调整布局，防止标签被截断
    plt.show()

# 提取4个关键数值列的数据（确保无缺失值）
data_four_numeric = data_final[four_key_cols].copy()
data_four_numeric = data_four_numeric.fillna(data_four_numeric.mean())  # 填充缺失值（若有）

# 调用修复后的函数，生成四格热力图
four_grid_heatmap(data_four_numeric)
print("\n四格热力图已成功生成！")