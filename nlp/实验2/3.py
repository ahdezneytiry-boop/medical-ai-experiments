import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 读取数据
df = pd.read_csv(r"C:\Users\ahdezneytiry-boop1\Desktop\实验2\peptide.csv", header=0)
sequences = df.iloc[:, 0].values

# 初始化CountVectorizer（字符级）
vectorizer = CountVectorizer(analyzer='char', lowercase=False)

# 拟合并转换
X_count = vectorizer.fit_transform(sequences)
print("CountVectorizer特征矩阵形状：", X_count.shape)
print("氨基酸特征列表：", vectorizer.get_feature_names_out())

# 转换为DataFrame查看
X_count_df = pd.DataFrame(X_count.toarray(), columns=vectorizer.get_feature_names_out())
print("前5条序列的计数特征：\n", X_count_df.head())

# 保存特征矩阵
X_count_df.to_csv("peptide_countvector.csv", index=False)