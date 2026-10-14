import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns

# 修改字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei')

# 读取文件，根据实际情况修改文件路径
data = pd.read_excel(r"C:\Users\ahdezneytiry-boop1\Desktop\实验2\dummy.xlsx")

print(data)
print("data head:\n", data.head())
print("data shape:\n", data.shape)
print("data descibe:\n", data.describe())

# 列级别的判断，但凡某一列有 null 值或空的，则为真
data.isnull().any()

# 将列中为空或者 null 的个数统计出来，并将缺失值最多的排前
total = data.isnull().sum().sort_values(ascending=False)
print("total:\n", total)

# 输出百分比
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

data = data.dropna(thresh=data.shape[0] * 0.5, axis=1)
# 至少有一半以上是非空的列筛选出来
# 如果某一行全部都是 na 才删除，默认情况下是只保留没有空值的行
data.dropna(axis=0, how='all')
print(data)

# 统计重复记录数
print("重复记录数：", data.duplicated().sum())
data = data.drop_duplicates()

data.columns

# 第一步，将整个 data 的连续型字段和离散型字段进行归类
id_col = ['姓名']
cat_col = ['学历', '学校']
cont_col = ['成绩', '能力']

print(data[cat_col])
print(data[cont_col])

# 计算出现的频次
for i in cat_col:
    print(pd.Series(data[i]).value_counts())
    plt.plot(data[i])

# 对于离散型数据，对其获取哑变量
dummies = pd.get_dummies(data[cat_col])
print("哑变量:\n", dummies)

# 对于连续型数据的统计
print("连续型数据统计:\n", data[cont_col].describe())

# 对于连续型数据，看偏度，将大于 0.75 的数值用 log 转化，使之符合正态分布
skewed_feats = data[cont_col].apply(lambda x: (x.dropna()).skew())
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
data[skewed_feats] = np.log1p(data[skewed_feats])

# 对于连续型数据，对其进行标准化
scaled = preprocessing.scale(data[cont_col])
scaled = pd.DataFrame(scaled, columns=cont_col)
print(scaled)

m = dummies.join(scaled)
data_cleaned = data[id_col].join(m)
print("标准化:\n", data_cleaned)

# 计算相关性之前，剔除字符串类型列
data_for_corr = data_cleaned.drop(columns=id_col)

# 变量之间的相关性：
print("变量之间的相关性:\n", data_for_corr.corr())


# 以下是相关性的热力图
def corr_heat(df):
    dfData = abs(df.corr())
    plt.subplots(figsize=(9, 9))
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    plt.show()


corr_heat(data_for_corr)