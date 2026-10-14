import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os

# -------------------------- 1. 本地加载数据（无联网需求） --------------------------
def load_merged_data():
    data_path = "data/招聘数据_合并预处理后.csv"
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 请确认文件在桌面：{data_path}")
        exit()
    # 尝试中文编码加载
    encodings = ['gbk', 'utf-8']
    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            # 还原分词列表
            df["岗位名称_seg"] = df["岗位名称_seg"].apply(eval)
            df["岗位名称_seg_str"] = df["岗位名称_seg"].apply(lambda x: " ".join(x) if x else "")
            print(f"✅ 加载成功：{len(df)}行数据，编码={encoding}")
            return df
        except Exception as e:
            continue
    print("❌ 编码错误：用Excel打开文件→另存为“CSV（逗号分隔）”再重试")
    exit()

df = load_merged_data()


# -------------------------- 2. 提取TF-IDF+BoW基础特征（核心功能无联网） --------------------------
def extract_basic_features(df):
    print("\n🔄 提取基础特征（TF-IDF + 词袋模型）...")
    # 1. 词袋模型（BoW：统计关键词频次，可解释性强）
    bow_vec = CountVectorizer(
        max_features=800,  # 保留800个高频关键词（适配数据分析师岗位数据）
        stop_words=None
    )
    bow_mat = bow_vec.fit_transform(df["岗位名称_seg_str"])
    bow_df = pd.DataFrame(
        bow_mat.toarray(),
        columns=[f"bow_{w}" for w in bow_vec.get_feature_names_out()],
        index=df.index
    )

    # 2. TF-IDF（突出“重要关键词”，如“高级”“Analyst”）
    tfidf_vec = TfidfVectorizer(
        max_features=800,
        smooth_idf=True  # 避免词频为0的计算错误
    )
    tfidf_mat = tfidf_vec.fit_transform(df["岗位名称_seg_str"])
    tfidf_df = pd.DataFrame(
        tfidf_mat.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf_vec.get_feature_names_out()],
        index=df.index
    )

    # 合并特征到原数据
    df_with_features = pd.concat([df, bow_df, tfidf_df], axis=1)
    print(f"✅ 特征提取完成：新增{len(bow_df.columns)+len(tfidf_df.columns)}个特征（BoW{len(bow_df.columns)}个 + TF-IDF{len(tfidf_df.columns)}个）")
    return df_with_features, bow_vec, tfidf_vec

df_with_features, bow_vec, tfidf_vec = extract_basic_features(df)


# -------------------------- 3. 保存结果到桌面（GBK编码，避免中文乱码） --------------------------
def save_results(df):
    full_path = "data/招聘数据_特征工程后_基础版.csv"
    # 单独保存纯特征矩阵（建模直接用）
    feature_cols = [col for col in df.columns if col.startswith(("bow_", "tfidf_"))]
    matrix_path = "data/招聘数据_特征矩阵_基础版.csv"
    
    df.to_csv(full_path, index=False, encoding='gbk')
    df[feature_cols].to_csv(matrix_path, index=False, encoding='gbk')
    print(f"\n💾 结果已保存到桌面：")
    print(f"   1. 完整数据（原始字段+特征）：{full_path}")
    print(f"   2. 纯特征矩阵（建模用）：{matrix_path}")

save_results(df_with_features)


# -------------------------- 4. 特征结果概览 --------------------------
def overview(df):
    print(f"\n📊 特征工程结果概览：")
    print(f"1. 数据规模：{len(df)}行 × {len(df.columns)}个字段")
    # 统计字段类型
    orig_cols = [c for c in df.columns if not c.startswith(("bow_", "tfidf_"))]
    bow_cols = [c for c in df.columns if c.startswith("bow_")]
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    print(f"2. 字段分布：")
    print(f"   - 原始预处理字段：{len(orig_cols)}个（如岗位名称、工作地址）")
    print(f"   - BoW特征：{len(bow_cols)}个（关键词频次，如bow_数据分析师）")
    print(f"   - TF-IDF特征：{len(tfidf_cols)}个（关键词重要性，如tfidf_Analyst）")
    # 显示前3个TF-IDF特征示例
    print(f"3. 核心特征示例（前5行样本）：")
    print(df[tfidf_cols[:3]].head().round(4))

overview(df_with_features)