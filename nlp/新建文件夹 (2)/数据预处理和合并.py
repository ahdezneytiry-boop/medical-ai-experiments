import pandas as pd
import re
import jieba
from collections import Counter

def load_and_align_fields():
    # 加载数据
    liepin_path = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\猎聘-列表页-限云采集.csv"
    job51_path = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\前程无忧-搜索关键词采集列表页.csv"
    liepin_df = pd.read_csv(liepin_path, encoding='utf-8').fillna("")
    job51_df = pd.read_csv(job51_path, encoding='utf-8').fillna("")


    liepin_align = liepin_df.rename(columns={
        "职位名称": "岗位名称",
        "区域": "工作地址",
        "公司行业": "行业",
        "公司名称": "公司",
        "公司规模": "规模",
        "融资情况": "融资情况"
    })[["关键词", "岗位名称", "工作地址", "薪资", "公司", "行业", "规模", "融资情况"]]
    liepin_align["平台来源"] = "猎聘"  # 新增平台标识

    job51_align = job51_df.rename(columns={
        "标签": "岗位标签",
        "公司类型": "公司类型"
    })[["关键词", "岗位名称", "工作地址", "薪资", "公司", "行业", "规模", "岗位标签", "公司类型"]]
    job51_align["平台来源"] = "前程无忧"  # 新增平台标识
    job51_align["融资情况"] = ""  # 猎聘有、前程无忧无的字段，用空字符串填充（保证字段数一致）

    # 确保两个DataFrame字段完全一致（顺序无关，列名需相同）
    common_cols = list(liepin_align.columns)
    job51_align = job51_align[common_cols]

    print(f" 字段对齐完成：猎聘{len(liepin_align)}行，前程无忧{len(job51_align)}行，统一为{len(common_cols)}个字段")
    return liepin_align, job51_align

liepin_align, job51_align = load_and_align_fields()


# -------------------------- 2. 文本清洗+标准化（复用预处理逻辑，统一处理合并前数据） --------------------------
# 清洗函数
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s\-\·]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if (len(text) >= 2 and not text.isdigit()) else ""

# 地址标准化（统一提取城市）
def standardize_address(address):
    if not address:
        return ""
    match = re.search(r"([^\-\·]+)[\-\·]", address)
    return match.group(1) if match else address

# 分词函数（自定义招聘词典）
jieba.add_word("数据分析师")
jieba.add_word("数据分析工程师")
jieba.add_word("Tableau")
jieba.add_word("SQL")
stopwords = set(["的", "了", "是", "在", "数据", "分析", "公司", "有限", "北京", "上海", "深圳"])

def segment_text(text):
    if not text:
        return []
    words = jieba.lcut(text)
    return [w for w in words if w not in stopwords and len(w)>=2 and not w.isdigit()]

# 统一处理两个平台数据
def preprocess_data(df, platform):
    # 清洗核心文本字段
    text_cols = ["关键词", "岗位名称", "工作地址", "公司", "行业"]
    for col in text_cols:
        df[f"{col}_cleaned"] = df[col].apply(clean_text)
    
    # 地址标准化
    df["工作地址_standard"] = df["工作地址_cleaned"].apply(standardize_address)
    
    # 岗位名称分词
    df["岗位名称_seg"] = df["岗位名称_cleaned"].apply(segment_text)
    
    # 过滤无效行（岗位名称为空则删除）
    df = df[df["岗位名称_cleaned"] != ""].reset_index(drop=True)
    print(f"{platform}预处理完成：剩余{len(df)}行")
    return df

liepin_processed = preprocess_data(liepin_align, "猎聘")
job51_processed = preprocess_data(job51_align, "前程无忧")


# -------------------------- 3. 数据合并+去重（避免重复职位干扰分析） --------------------------
def merge_and_deduplicate(liepin_df, job51_df):
    # 1. 纵向合并两个平台数据
    merged_df = pd.concat([liepin_df, job51_df], ignore_index=True)
    print(f"\n 初始合并后：共{len(merged_df)}行数据")

    # 2. 去重（核心：基于“岗位名称+公司+工作地址”联合去重，避免同一职位多平台重复）
    # 先创建去重标识字段（合并关键信息为字符串）
    merged_df["dedupe_key"] = merged_df["岗位名称_cleaned"] + "_" + merged_df["公司_cleaned"] + "_" + merged_df["工作地址_standard"]
    # 去重（保留第一次出现的记录）
    merged_df_deduped = merged_df.drop_duplicates(subset="dedupe_key", keep="first").reset_index(drop=True)
    
    # 3. 删除临时去重字段，保留有用字段
    drop_cols = ["dedupe_key"] + [col for col in merged_df.columns if col not in liepin_df.columns and "_cleaned" not in col and "_standard" not in col and "_seg" not in col]
    merged_df_deduped = merged_df_deduped.drop(columns=drop_cols)

    print(f" 去重完成：合并后{len(merged_df)}行 → 去重后{len(merged_df_deduped)}行，删除重复数据{len(merged_df)-len(merged_df_deduped)}行")
    return merged_df_deduped

merged_df = merge_and_deduplicate(liepin_processed, job51_processed)


# -------------------------- 4. 合并后文本统计（验证合并效果） --------------------------
def merged_text_statistics(df):
    print(f"\n 合并后数据统计结果：")
    # 1. 平台分布
    platform_dist = df["平台来源"].value_counts()
    print(f"1. 平台分布：")
    for platform, count in platform_dist.items():
        print(f"   - {platform}：{count}行（{round(count/len(df)*100,1)}%）")
    
    # 2. 热门城市TOP10
    city_dist = df["工作地址_standard"][df["工作地址_standard"] != ""].value_counts().head(10)
    print(f"2. 热门招聘城市TOP10：")
    for i, (city, count) in enumerate(city_dist.items(), 1):
        print(f"   {i:2d}. {city}：{count}个职位")
    
    # 3. 岗位名称高频关键词TOP20
    all_words = []
    for words in df["岗位名称_seg"]:
        all_words.extend(words)
    word_freq = Counter(all_words).most_common(20)
    print(f"3. 岗位核心关键词TOP20：")
    for i, (word, freq) in enumerate(word_freq, 1):
        print(f"   {i:2d}. {word}：{freq}次")

merged_text_statistics(merged_df)


# -------------------------- 5. 保存合并后数据 --------------------------
save_path = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\招聘数据_合并预处理后.csv"
merged_df.to_csv(save_path, index=False, encoding='utf-8')
print(f"\n合并数据已保存至：{save_path}")
print(f"   数据包含字段：{list(merged_df.columns)}")