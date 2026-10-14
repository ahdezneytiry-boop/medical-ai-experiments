import pandas as pd
import numpy as np

# -------------------------- 新增：内置岗位层级标注函数（解决KeyError） --------------------------
def label_level(job_name):
    """直接在整合代码中生成岗位层级，无需依赖外部数据"""
    job_name = str(job_name).lower()
    if '高级' in job_name or '资深' in job_name or 'senior' in job_name:
        return 3  # 高级
    elif '中级' in job_name or 'mid' in job_name:
        return 2  # 中级
    elif '助理' in job_name or 'junior' in job_name:
        return 1  # 初级
    else:
        return 2  # 默认中级

# -------------------------- 1. 读取所有需整合的数据 --------------------------
# 原始预处理数据（基础信息）
raw_df = pd.read_csv("data/招聘数据_合并预处理后.csv", encoding="gbk")
# 薪资结构化数据
salary_df = pd.read_csv("data/招聘数据_薪资结构化后.csv", encoding="gbk")
# 特征矩阵（用于提取关键词）
feat_df = pd.read_csv("data/招聘数据_特征矩阵_基础版.csv", encoding="gbk")
tfidf_cols = [col for col in feat_df.columns if col.startswith("tfidf_")]

# -------------------------- 2. 关键修复：生成“岗位层级”列（直接在代码中生成） --------------------------
raw_df["岗位层级"] = raw_df["岗位名称_cleaned"].apply(label_level)
print("✅ 已生成'岗位层级'列，无需依赖外部数据")

# -------------------------- 3. 处理各模块结果 --------------------------
# 3.1 提取每条岗位的TOP3关键词（基于TF-IDF权重）
def get_top3_keywords(row, tfidf_cols):
    row_tfidf = row[tfidf_cols]
    non_zero_kw = row_tfidf[row_tfidf > 0].sort_values(ascending=False)
    top3 = [kw.replace("tfidf_", "") for kw in non_zero_kw.index[:3]]
    return ",".join(top3) if top3 else "无"

raw_df["核心关键词_TOP3"] = feat_df.apply(lambda row: get_top3_keywords(row, tfidf_cols), axis=1)

# 3.2 处理薪资数据（添加“平均薪资”和“薪资有效性”）
salary_df["平均薪资(k/月)"] = salary_df[["最低薪资(k/月)", "最高薪资(k/月)"]].mean(axis=1).round(1)
salary_df["薪资是否有效"] = salary_df["最低薪资(k/月)"].apply(lambda x: 1 if pd.notna(x) and 0 < x <= 100 else 0)

# 3.3 关联薪资数据（通过“岗位名称_cleaned”确保1对1关联）
salary_df.rename(columns={"岗位名称_cleaned": "关联_岗位名称"}, inplace=True)
raw_df = raw_df.merge(
    salary_df[["关联_岗位名称", "薪资", "最低薪资(k/月)", "最高薪资(k/月)", "平均薪资(k/月)", "薪资是否有效"]],
    left_on="岗位名称_cleaned",
    right_on="关联_岗位名称",
    how="left"
)
raw_df.drop("关联_岗位名称", axis=1, inplace=True)

# -------------------------- 4. 整合为最终结构化表（筛选核心字段） --------------------------
final_cols = [
    # 基础信息
    "岗位名称_cleaned", "公司", "工作地址_standard", "平台来源",
    # NLP结果
    "核心关键词_TOP3", "岗位层级",
    # 结构化薪资
    "薪资", "最低薪资(k/月)", "最高薪资(k/月)", "平均薪资(k/月)", "薪资是否有效"
]
final_df = raw_df[final_cols].copy()

# 数据质量优化
final_df.drop_duplicates(subset=["岗位名称_cleaned", "公司"], inplace=True)  # 去重重复岗位
final_df["岗位层级"].fillna(2, inplace=True)  # 兜底处理（实际不会缺失）

# -------------------------- 5. 保存最终结构化表 --------------------------
save_path = "data/数据分析师岗位_结构化成果表.csv"
final_df.to_csv(save_path, index=False, encoding="gbk")
print(f"\n✅ 结构化成果表已保存：{save_path}")
print(f"📊 数据规模：{len(final_df)}条有效岗位数据，{len(final_cols)}个核心字段")
print(f"\n🔍 前3条示例数据：")
print(final_df[["岗位名称_cleaned", "工作地址_standard", "核心关键词_TOP3", "岗位层级", "平均薪资(k/月)"]].head(3).to_string(index=False))