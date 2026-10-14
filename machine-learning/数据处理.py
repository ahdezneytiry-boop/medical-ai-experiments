
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv("data/UCIHeartDisease_Original.csv")

# 将'?'替换为NaN
df['ca'] = df['ca'].replace('?', np.nan)
df['thal'] = df['thal'].replace('?', np.nan)

# 转换为数值类型
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])

# 检查缺失值数量
print("缺失值数量：")
print(df.isnull().sum())

# 查看缺失值的行
print("\n包含缺失值的行：")
print(df[df.isnull().any(axis=1)])

# 使用中位数填充缺失值
df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].median())

# 再次检查缺失值
print("\n处理后的缺失值数量：")
print(df.isnull().sum())

# 查看数据类型
print("\n处理后的数据类型：")
print(df.dtypes)

# 查看目标变量的分布
print("\n目标变量分布：")
print(df['target'].value_counts().sort_index())

# 保存处理后的数据用于后续分析
df.to_csv("data/UCIHeartDisease_Original处理结果.csv", index=False)
print("好了，结束了")
