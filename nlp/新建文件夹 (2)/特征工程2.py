import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 加载数据（特征矩阵+原始文本，用于标注层级）
feat_df = pd.read_csv("data/招聘数据_特征矩阵_基础版.csv", encoding='gbk')
raw_df = pd.read_csv("data/招聘数据_合并预处理后.csv", encoding='gbk')

# 2. 手动标注层级标签（示例规则，可扩展）
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

raw_df['岗位层级'] = raw_df['岗位名称_cleaned'].apply(label_level)
labels = raw_df['岗位层级'].values  # 标签
features = feat_df[ [col for col in feat_df.columns if col.startswith('tfidf_')] ].values  # TF-IDF特征

# 3. 训练+预测+评估
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 预测并输出准确率
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"=== 岗位层级分类结果 ===")
print(f"测试集准确率：{acc:.2%}")
print(f"前5个样本预测结果：{y_pred[:5]}（真实标签：{y_test[:5]}）")