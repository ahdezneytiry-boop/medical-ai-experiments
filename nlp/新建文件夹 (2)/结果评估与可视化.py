import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random

# -------------------------- 全局配置（路径+中文显示） --------------------------
# 本地文件路径（桌面）
FEAT_PATH = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\招聘数据_特征矩阵_基础版.csv"
RAW_PATH = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\招聘数据_合并预处理后.csv"
SALARY_PATH = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\招聘数据_薪资结构化后.csv"
# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 固定随机种子（结果可复现）
random.seed(42)
np.random.seed(42)


# -------------------------- 1. 统一加载所有数据（避免重复读取） --------------------------
def load_all_data():
    print("开始加载所有数据...")
    # 加载特征矩阵
    feat_df = pd.read_csv(FEAT_PATH, encoding='gbk')
    # 加载原始预处理数据
    raw_df = pd.read_csv(RAW_PATH, encoding='gbk')
    # 加载薪资结构化数据
    salary_df = pd.read_csv(SALARY_PATH, encoding='gbk')
    print(f" 数据加载完成：特征矩阵{len(feat_df)}行、原始数据{len(raw_df)}行、薪资数据{len(salary_df)}行")
    return feat_df, raw_df, salary_df

# 执行数据加载
feat_df, raw_df, salary_df = load_all_data()


# -------------------------- 2. 辅助函数（复用） --------------------------
# 岗位层级标注函数（评估+可视化共用）
def label_level(job_name):
    job_name = str(job_name).lower()
    if '高级' in job_name or '资深' in job_name or 'senior' in job_name:
        return 3  # 高级
    elif '中级' in job_name or 'mid' in job_name:
        return 2  # 中级
    elif '助理' in job_name or 'junior' in job_name:
        return 1  # 初级
    else:
        return 2  # 默认中级

# 相似度计算函数（评估+可视化共用）
def get_similarity_matrix(feat_df):
    tfidf_cols = [col for col in feat_df.columns if col.startswith('tfidf_')]
    tfidf_features = feat_df[tfidf_cols].values
    return cosine_similarity(tfidf_features)


# -------------------------- 3. 结果评估模块（按任务分类） --------------------------
print("\n" + "="*60 + " 结果评估开始 " + "="*60)

# 3.1 关键词提取评估
def evaluate_keywords(feat_df, raw_df):
    print("\n【1. 关键词提取评估】")
    # 计算TOP20关键词平均权重
    tfidf_cols = [col for col in feat_df.columns if col.startswith('tfidf_')]
    tfidf_avg = feat_df[tfidf_cols].mean(axis=0)
    top20_kw = tfidf_avg.sort_values(ascending=False).head(20)
    kw_names = [col.replace('tfidf_', '') for col in top20_kw.index]
    
    # 有效关键词筛选（排除地域、福利词）
    exclude_kw = ["徐州", "双休", "一金", "五险"]
    valid_kw = [kw for kw in kw_names if kw not in exclude_kw]
    valid_rate = len(valid_kw)/20 * 100
    
    # 随机抽样验证（20条）
    sample_jobs = raw_df.sample(20, random_state=42)[["岗位名称_cleaned", "岗位名称_seg"]]
    match_count = 0
    for _, row in sample_jobs.iterrows():
        seg = row["岗位名称_seg"]
        matched = [kw for kw in seg if kw in kw_names[:10]]
        if matched:
            match_count += 1
    
    # 输出结果
    print(f"有效关键词占比：{len(valid_kw)}/20 = {valid_rate:.1f}%")
    print(f"抽样匹配率：{match_count}/20 = {match_count/20*100:.1f}%（岗位分词含TOP10关键词）")
    return top20_kw  # 供后续可视化使用

# 执行关键词评估
top20_kw = evaluate_keywords(feat_df, raw_df)


# 3.2 薪资抽取评估
def evaluate_salary(salary_df):
    print("\n【2. 薪资抽取评估】")
    # 过滤异常值（薪资>100k/月视为异常）
    salary_df["有效标记"] = salary_df["最低薪资(k/月)"].apply(
        lambda x: 1 if pd.notna(x) and 0 < x <= 100 else 0
    )
    total = len(salary_df)
    raw_success = salary_df["最低薪资(k/月)"].notna().sum()
    valid_success = salary_df["有效标记"].sum()
    
    # 异常值统计
    abnormal = salary_df[salary_df["最低薪资(k/月)"] > 100]
    
    # 输出结果
    print(f"原始抽取成功率：{raw_success}/{total} = {raw_success/total*100:.1f}%")
    print(f" 有效抽取率（排除异常值）：{valid_success}/{total} = {valid_success/total*100:.1f}%")
    print(f" 异常值数量：{len(abnormal)}条（薪资>100k/月，需优化正则）")
    return salary_df  # 供后续可视化使用

# 执行薪资评估
salary_df = evaluate_salary(salary_df)


# 3.3 职位相似度评估
def evaluate_similarity(feat_df, raw_df):
    print("\n【3. 职位相似度评估】")
    # 计算相似度矩阵
    sim_matrix = get_similarity_matrix(feat_df)
    # 随机抽5个目标岗位
    target_indices = random.sample(range(len(raw_df)), 5)
    avg_similar_count = 0
    
    for i, idx in enumerate(target_indices, 1):
        target_job = raw_df.iloc[idx]["岗位名称_cleaned"]
        # 筛选相似度>0.8的岗位（排除自身）
        similar_indices = [j for j, s in enumerate(sim_matrix[idx]) if s > 0.8 and j != idx]
        similar_count = len(similar_indices)
        avg_similar_count += similar_count
        # 输出单条结果
        print(f"   {i}. 目标岗位：{target_job[:25]:<25} | 相似岗位数：{similar_count}个")
    
    # 输出平均结果
    avg_similar_count = avg_similar_count / 5
    print(f" 平均每个岗位相似岗位数：{avg_similar_count:.1f}个（相似度阈值=0.8）")
    return sim_matrix  # 供后续可视化使用

# 执行相似度评估
sim_matrix = evaluate_similarity(feat_df, raw_df)


# 3.4 岗位层级分类评估
def evaluate_level_classification(feat_df, raw_df):
    print("\n【4. 岗位层级分类评估】")
    # 准备特征和标签
    raw_df['岗位层级'] = raw_df['岗位名称_cleaned'].apply(label_level)
    X = feat_df[[col for col in feat_df.columns if col.startswith('tfidf_')]].values
    y = raw_df['岗位层级'].values
    # 划分数据集并训练
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # 输出结果
    print(f"测试集准确率：{accuracy:.2%}")
    print(f"测试集加权F1值：{f1:.4f}")
    print(f"混淆矩阵（行=真实，列=预测；1=初级，2=中级，3=高级）：")
    print(conf_mat)
    return model, y_test, y_pred, conf_mat  # 供后续可视化使用

# 执行分类评估
level_model, y_test, y_pred, conf_mat = evaluate_level_classification(feat_df, raw_df)

print("\n" + "="*60 + " 结果评估结束 " + "="*60)


# -------------------------- 4. 可视化模块（按图表分类） --------------------------
print("\n" + "="*60 + " 可视化开始 " + "="*60)

# 4.1 关键词TOP15柱状图
def plot_keywords(top20_kw):
    print("\n【1. 绘制关键词TOP15图表...】")
    # 筛选有效关键词（排除地域、福利词）
    exclude_kw = ["徐州", "双休", "一金", "五险"]
    top20_kw_filtered = top20_kw[~top20_kw.index.str.replace('tfidf_', '').isin(exclude_kw)]
    top15_kw = top20_kw_filtered.head(15)
    # 提取关键词名称和权重
    kw_names = [col.replace('tfidf_', '') for col in top15_kw.index]
    kw_weights = top15_kw.values
    
    # 绘制横向柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(kw_names)), kw_weights, color='#1f77b4')
    # 样式设置
    ax.set_yticks(range(len(kw_names)))
    ax.set_yticklabels(kw_names, fontsize=10)
    ax.set_xlabel('平均TF-IDF权重', fontsize=12)
    ax.set_title('数据分析师岗位核心关键词TOP15（技能/层级类）', fontsize=14, pad=20)
    ax.grid(axis='x', alpha=0.3)
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    # 保存
    plt.tight_layout()
    plt.savefig("C:\\Users\\ahdezneytiry-boop1\\Desktop\\关键词TOP15.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" 关键词TOP15图表已保存！")

# 执行关键词可视化
plot_keywords(top20_kw)


# 4.2 岗位层级混淆矩阵热力图
def plot_confusion_matrix(conf_mat):
    print("\n【2. 绘制混淆矩阵图表...】")
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_mat, cmap='Blues', aspect='auto')
    # 标签设置（1=初级，2=中级，3=高级）
    level_labels = ['初级', '中级', '高级']
    ax.set_xticks(range(len(level_labels)))
    ax.set_yticks(range(len(level_labels)))
    ax.set_xticklabels(level_labels, fontsize=12)
    ax.set_yticklabels(level_labels, fontsize=12)
    ax.set_xlabel('预测层级', fontsize=12)
    ax.set_ylabel('真实层级', fontsize=12)
    ax.set_title('岗位层级分类混淆矩阵（准确率94.83%）', fontsize=14, pad=20)
    # 添加数值标签
    for i in range(len(level_labels)):
        for j in range(len(level_labels)):
            text = ax.text(j, i, conf_mat[i, j], ha='center', va='center', 
                          color='white' if conf_mat[i, j] > conf_mat.max()/2 else 'black', 
                          fontsize=11)
    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('样本数量', fontsize=10)
    # 保存
    plt.tight_layout()
    plt.savefig("C:\\Users\\ahdezneytiry-boop1\\Desktop\\岗位层级混淆矩阵.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("混淆矩阵图表已保存！")

# 执行混淆矩阵可视化
plot_confusion_matrix(conf_mat)


# 4.3 热门城市薪资箱线图
def plot_salary_distribution(salary_df):
    print("\n【3. 绘制薪资分布图表...】")
    # 过滤异常值+筛选热门城市（TOP6）
    hot_cities = ['徐州', '上海', '深圳', '北京', '广州', '成都']
    salary_clean = salary_df[
        (salary_df["最低薪资(k/月)"] <= 100) & 
        (salary_df["工作地址_standard"].isin(hot_cities)) &
        (salary_df["最低薪资(k/月)"].notna())
    ]
    # 按城市分组
    city_salary = []
    city_names = []
    for city in hot_cities:
        city_data = salary_clean[salary_clean["工作地址_standard"] == city]["最低薪资(k/月)"].values
        if len(city_data) >= 5:
            city_salary.append(city_data)
            city_names.append(f"{city}\n({len(city_data)}条)")
    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(12, 7))
    box_plot = ax.boxplot(city_salary, labels=city_names, patch_artist=True,
                          boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
    # 样式设置
    ax.set_ylabel('最低薪资（k/月）', fontsize=12)
    ax.set_title('热门城市数据分析师岗位薪资分布（过滤异常值后）', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    # 保存
    plt.tight_layout()
    plt.savefig("C:\\Users\\ahdezneytiry-boop1\\Desktop\\热门城市薪资分布.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("薪资分布图表已保存！")

# 执行薪资可视化
plot_salary_distribution(salary_df)


# 4.4 岗位相似度TOP20折线图
def plot_similarity_top20(sim_matrix, raw_df):
    print("\n【4. 绘制岗位相似度图表...】")
    # 选择目标岗位（优先选"高级数据分析师"，无则选第一条）
    target_idx = raw_df[raw_df["岗位名称_cleaned"].str.contains("高级数据分析师", na=False)].index
    if len(target_idx) == 0:
        target_idx = [0]  # 备选：第一条岗位
    target_idx = target_idx[0]
    target_job = raw_df.iloc[target_idx]["岗位名称_cleaned"]
    
    # 获取TOP20相似岗位（排除自身）
    similar_scores = sim_matrix[target_idx]
    sorted_indices = np.argsort(similar_scores)[::-1][1:21]  # 从高到低排，排除自身
    sorted_scores = similar_scores[sorted_indices]
    # 处理岗位名称过长（截断）
    sorted_jobs = []
    for i in sorted_indices:
        job = raw_df.iloc[i]["岗位名称_cleaned"]
        sorted_jobs.append(job[:15] + "..." if len(job) > 15 else job)
    
    # 绘制折线图
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(sorted_scores)), sorted_scores, marker='o', linewidth=2, markersize=6, color='#1f77b4')
    # 样式设置
    ax.set_xticks(range(len(sorted_jobs)))
    ax.set_xticklabels(sorted_jobs, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('余弦相似度（0~1）', fontsize=12)
    ax.set_title(f'目标岗位：{target_job[:25]}... → TOP20相似岗位相似度', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0.5)
    # 添加阈值线
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='相似度阈值=0.8')
    ax.legend()
    # 保存
    plt.tight_layout()
    plt.savefig("C:\\Users\\ahdezneytiry-boop1\\Desktop\\岗位相似度TOP20.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 岗位相似度图表已保存！")

# 执行相似度可视化
plot_similarity_top20(sim_matrix, raw_df)

print("\n" + "="*60 + " 可视化结束 " + "="*60)
print("\n所有评估结果已输出，4张图表已保存到桌面！")
print("图表文件：关键词TOP15.png、岗位层级混淆矩阵.png、热门城市薪资分布.png、岗位相似度TOP20.png")