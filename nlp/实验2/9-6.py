import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
import missingno  # 可视化缺失值的库

# -------------------------- 1. 基础设置与数据读取 --------------------------
# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']       # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示为方块的问题
sns.set(font='SimHei')                             # Seaborn中文设置

# 读取数据（请确保dummy.xls路径正确，若路径错误需修改）
data = pd.read_excel("data/dummy.xlsx")

# 数据基础信息查看
print("=== 原始数据 ===")
print(data)
print("\n=== 数据前5行 ===")
print(data.head())
print("\n=== 数据行列大小 ===")
print(data.shape)
print("\n=== 数据统计描述 ===")
print(data.describe())


# -------------------------- 2. 缺失值处理（保留所有列） --------------------------
# 查看各列缺失值数量与百分比
total = data.isnull().sum().sort_values(ascending=False)  # 缺失值总数
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)  # 缺失值百分比
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("\n=== 缺失值统计（保留所有列） ===")
print(missing_data.head(20))

# 可视化缺失值（显示所有列的缺失情况）
missingno.matrix(data)
plt.title("所有列的缺失值分布")
plt.show()

# 筛选：保留至少50%非空值的列（不删除任何符合条件的列，包括字符串列）
data = data.dropna(thresh=data.shape[0]*0.5, axis=1) 
# 删除完全为空的行（不影响列的保留）
data = data.dropna(axis=0, how='all')
print("\n=== 缺失值处理后的数据（保留所有符合条件的列） ===")
print(data)


# -------------------------- 3. 重复值处理（保留所有列） --------------------------
print("\n=== 重复记录数 ===")
print(data.duplicated().sum())
# 删除重复行（保留首次出现的行，所有列均参与重复判断）
data = data.drop_duplicates()
print("\n=== 去重后的数据（保留所有列） ===")
print(data)


# -------------------------- 4. 特征分类与处理（保留所有原始列） --------------------------
# 列分类（明确字符串列与数值列，不删除任何列）
id_col = ['姓名']                  # 字符串列：标识信息
cat_col = ['学历', '学校']         # 字符串列：离散型特征
cont_col = ['成绩', '能力']        # 数值列：连续型特征

# 查看各类型原始列（验证所有列均保留）
print("\n=== 字符串列：姓名 ===")
print(data[id_col])
print("\n=== 字符串列：学历、学校 ===")
print(data[cat_col])
print("\n=== 数值列：成绩、能力 ===")
print(data[cont_col])

# 离散型字符串列：统计频次（不修改原始列）
print("\n=== 离散字符串列频次统计 ===")
for i in cat_col:
    print(f"\n{i}列频次：")
    print(pd.Series(data[i]).value_counts())
    # 绘制离散列频次图（基于原始字符串列）
    plt.figure(figsize=(8, 4))
    pd.Series(data[i]).value_counts().plot(kind='bar')
    plt.title(f'{i}列频次分布')
    plt.show()

# 离散型字符串列：生成哑变量（数值型，用于后续分析）
dummies = pd.get_dummies(data[cat_col], prefix=cat_col)  # 哑变量列名带前缀，避免冲突
print("\n=== 离散字符串列生成的哑变量= ===")
print(dummies)

# 连续型数值列：处理偏度（使其更接近正态分布）
print("\n=== 连续数值列偏度处理前 ===")
print(data[cont_col].describe())
# 计算偏度，仅处理偏度>0.75的列
skewed_feats = data[cont_col].apply(lambda x: (x.dropna()).skew())
skewed_feats = skewed_feats[skewed_feats > 0.75]
print("\n=== 偏度>0.75的连续列 ===")
print(skewed_feats.index.tolist())
# 对数转换（仅处理偏度超标的列，不修改原始列）
data_skewed_processed = data[cont_col].copy()
if not skewed_feats.empty:
    data_skewed_processed[skewed_feats.index] = np.log1p(data_skewed_processed[skewed_feats.index])
print("\n=== 连续数值列偏度处理后 ===")
print(data_skewed_processed.describe())

# 连续型数值列：标准化（数值型，用于后续分析）
scaler = preprocessing.StandardScaler()  # 标准化器
scaled = scaler.fit_transform(data_skewed_processed)  # 标准化处理
scaled = pd.DataFrame(scaled, columns=cont_col, index=data.index)  # 保持索引与原始数据一致
print("\n=== 连续数值列标准化后= ===")
print(scaled)


# -------------------------- 5. 合并数据（保留所有原始列+衍生数值列） --------------------------
# 合并逻辑：原始所有列 + 哑变量 + 标准化连续列
# 步骤1：先合并原始数据与哑变量（保留所有原始字符串列和数值列）
data_with_dummies = pd.concat([data, dummies], axis=1)
# 步骤2：再合并标准化后的连续列（替换原始连续列，避免重复；也可保留原始连续列，用新列名区分）
# 此处选择保留原始连续列，标准化列加前缀“scaled_”
scaled.columns = [f'scaled_{col}' for col in scaled.columns]
data_final = pd.concat([data_with_dummies, scaled], axis=1)

print("\n=== 最终数据（= ===")
print("最终数据列名：", data_final.columns.tolist())
print("最终数据形状：", data_final.shape)
print("\n最终数据前5行：")
print(data_final.head())


# -------------------------- 6. 相关性分析（仅用数值列，保留所有字符串列） --------------------------
# 关键步骤：筛选所有数值型列（排除字符串列），仅用数值列做相关性分析
# 数值型列判断： dtype为int64或float64
numeric_cols = data_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\n=== 用于相关性分析的数值型列 ===")
print(numeric_cols)

# 提取纯数值数据（临时使用，不修改原始data_final，保留所有字符串列）
data_numeric_for_corr = data_final[numeric_cols].copy()
# 处理数值列中可能残留的缺失值（用均值填充，避免分析报错）
data_numeric_for_corr = data_numeric_for_corr.fillna(data_numeric_for_corr.mean())

# 计算相关性
print("\n=== 数值型变量之间的相关性 ===")
corr_result = data_numeric_for_corr.corr()
print(corr_result)

# 绘制相关性热力图（仅用数值列）
def corr_heat(df):
    dfData = abs(df.corr())  # 取相关系数绝对值，便于观察
    plt.subplots(figsize=(12, 10))  # 调整图大小，适配多列
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues", fmt=".2f")
    plt.title("数值型变量相关性热力图（保留所有字符串列）")
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    plt.show()

# 调用热力图函数（仅传入数值列数据）
corr_heat(data_numeric_for_corr)