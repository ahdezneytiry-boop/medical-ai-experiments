import os
from Bio import SeqIO
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. 读数据
df = pd.read_csv("data.csv")

# 2. 诊断
print(df.info())
print(df.describe(include="all").T)
print("缺失值：", df.isna().sum())
print("重复值：", df.duplicated().sum())
print(df.head())

# 3. 处理缺失与异常
# 3.1 缺失值（按字段策略）
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# 3.2 异常值：IQR
def clip_iqr(series, k=1.5):
    q1, q3 = series.quantile([0.25,0.75])
    iqr = q3-q1
    low, high = q1 - k*iqr, q3 + k*iqr
    return series.clip(lower=low, upper=high)

for c in num_cols:
    df[c] = clip_iqr(df[c])

# 3.3 处理重复
df = df.drop_duplicates()

# 3.4 统一格式
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip().str.lower().replace({"n/a":"","nan":""})
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 4. 预处理 pipeline
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

X = df.drop(columns=["target"], errors="ignore")
y = df["target"] if "target" in df else None

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y is not None else None)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# 5. 导出
df.to_csv("data_clean.csv", index=False)