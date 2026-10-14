import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 读取数据
df = pd.read_csv(r"C:\Users\ahdezneytiry-boop1\Desktop\实验2\peptide.csv", header=0)
sequences = df.iloc[:, 0].values  # 提取肽段序列
print("肽段序列数量：", len(sequences))
print("序列长度：", len(sequences[0]))

# 提取所有唯一氨基酸
all_chars = sorted(list(set(''.join(sequences))))
n_chars = len(all_chars)
print("氨基酸种类数：", n_chars)
print("氨基酸列表：", all_chars)

# 字符映射为索引
char_to_idx = {char: i for i, char in enumerate(all_chars)}

# one-hot编码
def one_hot_encode(seq):
    # 初始化序列矩阵：长度×氨基酸种类
    seq_matrix = np.zeros((len(seq), n_chars))
    for i, char in enumerate(seq):
        seq_matrix[i, char_to_idx[char]] = 1
    return seq_matrix.flatten()  # 展平为一维向量

# 对所有序列编码
X_onehot = np.array([one_hot_encode(seq) for seq in sequences])
print("one-hot特征矩阵形状：", X_onehot.shape)
print("第一条序列的前10维特征：", X_onehot[0][:10])

# 保存特征矩阵
pd.DataFrame(X_onehot).to_csv("peptide_onehot.csv", index=False, header=False)