import pandas as pd
import numpy as np

# 1. 加载特征矩阵（仅TF-IDF特征）
feat_matrix_path = "data/招聘数据_特征矩阵_基础版.csv"
feat_df = pd.read_csv(feat_matrix_path, encoding='gbk')

# 2. 筛选TF-IDF特征列，计算每个关键词的平均权重（全样本重要性）
tfidf_cols = [col for col in feat_df.columns if col.startswith('tfidf_')]
tfidf_avg = feat_df[tfidf_cols].mean(axis=0)  # 按列求均值

# 3. 排序取Top20关键词（去除"tfidf_"前缀）
top_keywords = tfidf_avg.sort_values(ascending=False).head(20)
top_keywords.index = [idx.replace('tfidf_', '') for idx in top_keywords.index]

# 输出结果
print("=== 数据分析师岗位核心关键词TOP20（TF-IDF权重）===")
for i, (word, weight) in enumerate(top_keywords.items(), 1):
    print(f"{i:2d}. {word} → 平均权重：{weight:.4f}")