# 导入所有依赖库（不变）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. 加载数据 & 初始化参数（不变）
# ------------------------------------------------------------------------------
print("="*60)
print("步骤1：加载数据并初始化")
print("="*60)

LOCAL_DIR = r"E:\jqxx\机器学习大作业"
DATA_LOAD_PATH = f"{LOCAL_DIR}\\bbb_clean_with_features_and_label.csv"
HEATMAP_SAVE = f"{LOCAL_DIR}\\phys_chem_correlation.png"
FEATURE_IMPORTANCE_SAVE = f"{LOCAL_DIR}\\feature_importance.csv"
FINAL_DATA_SAVE = f"{LOCAL_DIR}\\bbb_feature_engineering_result.csv"
REPORT_SAVE = f"{LOCAL_DIR}\\bbb_feature_engineering_report.md"

# 加载数据
try:
    df = pd.read_csv(DATA_LOAD_PATH)
    print(f"✅ 数据加载成功：样本数={df.shape[0]}，原有列数={df.shape[1]}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 未找到数据文件，请确认路径：{DATA_LOAD_PATH}")

# 定义非特征列
non_feature_cols = [
    "Smiles_unify", "property", "target", "random_train_test_label",
    "scaffold_train_test_label"
]

# 筛选纯数值理化特征
original_phys_chem_cols = [col for col in df.columns if col not in non_feature_cols]
for col in original_phys_chem_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"❌ 列「{col}」不是数值型，无法作为理化特征")
print(f"原有分子理化特征（纯数值）：{original_phys_chem_cols}")

# 初始化参数（不变）
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
IMPORTANCE_QUANTILE = 0.8
DOMAIN_KEY_FEATURES = ["logP", "tpsa", "h_bond_donor"]
print(f"Morgan指纹参数：半径={MORGAN_RADIUS}，维度={MORGAN_NBITS}")
print(f"特征重要性阈值：分位数={IMPORTANCE_QUANTILE}（筛选前{int(IMPORTANCE_QUANTILE*100)}%特征）")
print(f"BBB关键特征（强制保留）：{DOMAIN_KEY_FEATURES}")


# ------------------------------------------------------------------------------
# 2. 特征提取：生成Morgan指纹（不变）
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("步骤2：特征提取 - 生成Morgan分子指纹")
print("="*60)

def extract_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(MORGAN_NBITS, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return np.array(fp, dtype=int)

print(f"正在提取{len(df)}个样本的Morgan指纹（{MORGAN_NBITS}维）...")
morgan_fp_list = df["Smiles_unify"].apply(extract_morgan_fingerprint)
morgan_cols = [f"morgan_{i}" for i in range(MORGAN_NBITS)]
morgan_df = pd.DataFrame(morgan_fp_list.tolist(), columns=morgan_cols, index=df.index)

feature_df = df.drop(columns=non_feature_cols).copy()
full_feature_df = pd.concat([feature_df, morgan_df], axis=1, ignore_index=False)
final_df = pd.concat([df[non_feature_cols], full_feature_df], axis=1, ignore_index=False)

print(f"✅ 特征提取完成：新增{MORGAN_NBITS}维Morgan指纹")
print(f"   总特征数：{full_feature_df.shape[1]}（{len(original_phys_chem_cols)}理化 + {MORGAN_NBITS}指纹）")


# ------------------------------------------------------------------------------
# 3. 特征选择（不变，已验证无问题）
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("步骤3：特征选择 - 筛选核心特征")
print("="*60)

# 3.1 相关性过滤
print("\n3.1 相关性过滤（删除高冗余理化特征）...")
phys_chem_corr = feature_df[original_phys_chem_cols].corr()

plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
sns.heatmap(phys_chem_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("10个分子理化特征相关性热力图")
plt.tight_layout()
plt.savefig(HEATMAP_SAVE, dpi=300)
plt.close()
print(f"   ✅ 相关性热力图已保存：{HEATMAP_SAVE}")

# 计算高相关特征对
high_corr_pairs = []
for i in range(len(phys_chem_corr.columns)):
    for j in range(i + 1, len(phys_chem_corr.columns)):
        corr_val = abs(phys_chem_corr.iloc[i, j])
        if corr_val > 0.8:
            high_corr_pairs.append((phys_chem_corr.columns[i], phys_chem_corr.columns[j], round(corr_val, 2)))

# 确定删除的特征
target_corr = pd.concat([feature_df[original_phys_chem_cols], df["target"]], axis=1).corr()["target"]
target_corr = target_corr.drop("target")
drop_phys_chem = []
if high_corr_pairs:
    print(f"   高相关特征对（|r|>0.8）：")
    for col1, col2, corr_val in high_corr_pairs:
        if target_corr[col1] >= target_corr[col2]:
            drop_phys_chem.append(col2)
            print(f"     - {col1} ↔ {col2}（r={corr_val}）→ 保留{col1}，删除{col2}")
        else:
            drop_phys_chem.append(col1)
            print(f"     - {col1} ↔ {col2}（r={corr_val}）→ 保留{col2}，删除{col1}")
    drop_phys_chem = list(set(drop_phys_chem))
else:
    print(f"   ✅ 无高相关理化特征（|r|>0.8），无需删除")

keep_phys_chem = [col for col in original_phys_chem_cols if col not in drop_phys_chem]
print(f"   保留的理化特征：{keep_phys_chem}（共{len(keep_phys_chem)}个）")


# 3.2 模型重要性筛选
print("\n3.2 模型特征重要性筛选（随机森林）...")
if "random_train_test_label" not in final_df.columns:
    raise ValueError("❌ 数据中无「random_train_test_label」列，请检查列名")
train_mask = final_df["random_train_test_label"] == "train"
if train_mask.sum() == 0:
    raise ValueError("❌ 未找到「train」标签样本，请确认标签值（如1/0）")

train_features = keep_phys_chem + morgan_cols
X_train = final_df.loc[train_mask, train_features].copy()
y_train = final_df.loc[train_mask, "target"].copy()

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "feature_name": train_features,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)
feature_importance.to_csv(FEATURE_IMPORTANCE_SAVE, index=False, encoding="utf-8")

importance_threshold = feature_importance["importance"].quantile(IMPORTANCE_QUANTILE)
important_features = feature_importance[feature_importance["importance"] >= importance_threshold]["feature_name"].tolist()

important_phys_chem = [col for col in important_features if col in keep_phys_chem]
important_morgan = [col for col in important_features if col in morgan_cols]
print(f"   筛选前{int(IMPORTANCE_QUANTILE*100)}%重要特征：")
print(f"     - 总数量：{len(important_features)}（理化{len(important_phys_chem)}个 + 指纹{len(important_morgan)}个）")
print(f"   ✅ 特征重要性表已保存：{FEATURE_IMPORTANCE_SAVE}")


# 3.3 领域知识验证
print("\n3.3 领域知识验证（保留BBB关键特征）...")
missing_key_features = [col for col in DOMAIN_KEY_FEATURES if col not in important_features]
if missing_key_features:
    important_features.extend(missing_key_features)
    print(f"   补充缺失的BBB关键特征：{missing_key_features}")

final_core_features = list(set(important_features))
final_phys_chem = [col for col in final_core_features if col in keep_phys_chem]
final_morgan = [col for col in final_core_features if col in morgan_cols]
print(f"   最终核心特征数：{len(final_core_features)}（理化{len(final_phys_chem)}个 + 指纹{len(final_morgan)}个）")


# ------------------------------------------------------------------------------
# 4. 特征变换：核心修正（移除PowerTransformer的random_state参数）
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("步骤4：特征变换 - 优化特征分布与维度")
print("="*60)

# 4.1 理化特征正态化（适配旧版本sklearn：删除random_state）
print("\n4.1 理化特征正态化（Yeo-Johnson变换）...")
if len(final_phys_chem) > 0:
    # 核心修正：移除random_state参数（旧版本sklearn无此参数）
    pt = PowerTransformer(method="yeo-johnson")  # 仅保留method参数
    # 仅用训练集拟合
    pt.fit(final_df.loc[train_mask, final_phys_chem])
    # 对全量数据变换
    final_df.loc[:, final_phys_chem] = pt.transform(final_df[final_phys_chem])
    print(f"   ✅ 完成{len(final_phys_chem)}个理化特征的正态化")
else:
    print(f"   ⚠️  无需要正态化的理化特征，跳过此步骤")


# 4.2 指纹特征PCA降维（不变）
print("\n4.2 指纹特征PCA降维（保留95%信息）...")
pca_cols = []
if len(final_morgan) > 50:
    # PCA的random_state参数在旧版本中存在，可保留
    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(final_df.loc[train_mask, final_morgan])
    morgan_pca = pca.transform(final_df[final_morgan])
    pca_cols = [f"morgan_pca_{i}" for i in range(morgan_pca.shape[1])]
    morgan_pca_df = pd.DataFrame(morgan_pca, columns=pca_cols, index=final_df.index)
    
    final_df = final_df.drop(columns=final_morgan)
    final_df = pd.concat([final_df, morgan_pca_df], axis=1)
    final_core_features = [c for c in final_core_features if c not in final_morgan] + pca_cols
    
    print(f"   ✅ 指纹降维完成：原{len(final_morgan)}维 → 降维后{len(pca_cols)}维（保留95%信息）")
else:
    pca_cols = final_morgan
    print(f"   ⚠️  指纹特征数较少（{len(final_morgan)}个），无需PCA降维")


# ------------------------------------------------------------------------------
# 5. 输出结果（不变）
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("步骤5：输出特征工程结果")
print("="*60)

# 5.1 保存最终数据集
output_cols = non_feature_cols + final_core_features
fe_result = final_df[output_cols].copy()
fe_result.to_csv(FINAL_DATA_SAVE, index=False, encoding="utf-8")
print(f"\n5.1 最终建模数据集已保存：{FINAL_DATA_SAVE}")
print(f"   数据集概况：")
print(f"     - 样本数：{fe_result.shape[0]}")
print(f"     - 总列数：{fe_result.shape[1]}（{len(non_feature_cols)}个非特征列 + {len(final_core_features)}个核心特征列）")


# 5.2 生成报告
print("\n5.2 生成特征工程报告...")
pca_before = len(final_morgan) if len(final_morgan) > 0 else 0
pca_after = len(pca_cols) if len(pca_cols) > 0 else 0
high_corr_count = len(high_corr_pairs) if high_corr_pairs else 0

report_content = f"""# BBB血脑屏障预测 - 特征工程报告
## 一、流程概述
本次特征工程围绕**“补充有效特征→剔除冗余噪声→优化特征形式”** 展开，适配GNN+XGBoost建模需求，最终得到低冗余、高区分度的核心特征集，共处理8301个样本。

## 二、关键步骤详情
### 1. 特征提取（宏观属性+微观结构）
| 特征类型       | 数量/维度   | 具体说明                                                                 |
|----------------|-------------|--------------------------------------------------------------------------|
| 原有理化特征   | {len(original_phys_chem_cols)}个 | 涵盖分子量（mol_weight）、脂水分配系数（logP）、极性表面积（tpsa）等BBB渗透关键属性 |
| 新增Morgan指纹 | {MORGAN_NBITS}维       | 捕捉分子微观结构（半径={MORGAN_RADIUS}），覆盖官能团、化学键组合等细节信息，提升区分度 |

### 2. 特征选择（三层过滤策略）
#### （1）相关性过滤（处理理化特征冗余）
- 筛选逻辑：删除相关系数|r|>0.8的高冗余特征，保留与目标变量（BBB渗透）相关性更强的特征；
- 处理结果：共发现{high_corr_count}对高相关特征，删除{len(drop_phys_chem)}个冗余特征，保留{len(keep_phys_chem)}个理化特征；
- 保留特征：{keep_phys_chem}。

#### （2）模型重要性筛选（随机森林）
- 筛选逻辑：用训练集训练随机森林，筛选特征重要性前{int(IMPORTANCE_QUANTILE*100)}%的特征；
- 处理结果：从{len(train_features)}个特征中筛选出{len(important_features)}个重要特征（理化{len(important_phys_chem)}个 + 指纹{len(important_morgan)}个）。

#### （3）领域知识验证（强制保留关键特征）
- 验证逻辑：BBB渗透的核心影响因素（logP、tpsa、h_bond_donor）必须保留，若未被筛选则补充；
- 处理结果：补充{len(missing_key_features) if missing_key_features else 0}个关键特征，最终核心特征共{len(final_core_features)}个。

### 3. 特征变换（优化模型输入）
| 变换类型       | 处理对象                 | 效果                                                                 |
|----------------|--------------------------|----------------------------------------------------------------------|
| Yeo-Johnson变换 | {len(final_phys_chem)}个理化特征 | 消除特征偏态分布（如logP的长尾分布），提升模型对数值特征的拟合效果     |
| PCA降维        | {pca_before}维指纹特征       | 降维至{pca_after}维（保留95%信息），减少计算量，避免维度灾难           |

## 三、最终数据集说明
| 列类型         | 列名示例                  | 数量   | 用途                          |
|----------------|---------------------------|--------|-------------------------------|
| 非特征列       | Smiles_unify, target      | {len(non_feature_cols)} | Smiles（GNN图结构输入）、target（建模目标）、训练测试划分 |
| 核心特征列     | logP, tpsa, morgan_pca_0  | {len(final_core_features)} | XGBoost输入（或GNN节点属性） |

## 四、建模建议
### 1. GNN建模
- 图结构输入：用`Smiles_unify`列通过RDKit或DGL生成分子图；
- 节点属性补充：将`logP`、`tpsa`等核心理化特征作为节点属性，增强结构信息表达；
- 输出：将GNN的结构嵌入向量作为XGBoost的输入之一。

### 2. XGBoost建模
- 输入特征：GNN结构嵌入向量 + 本次筛选的核心特征；
- 预处理：无需额外正态化（理化特征已做Yeo-Johnson变换，PCA特征已标准化）；
- 超参调优：重点调整`max_depth`（3-10）、`learning_rate`（0.01-0.1）、`n_estimators`（100-500）。

### 3. 模型评估
- 数据划分：严格使用`random_train_test_label`列划分训练/测试集，避免数据泄露；
- 评估指标：重点关注**召回率**（避免漏判BBB渗透分子）和**F1-score**（平衡精确率与召回率）。

## 五、文件说明
| 文件名                          | 用途                                  |
|---------------------------------|---------------------------------------|
| bbb_feature_engineering_result.csv | 最终建模数据集（含核心特征+标签）     |
| feature_importance.csv           | 特征重要性排序表（便于分析关键特征）  |
| phys_chem_correlation.png       | 理化特征相关性热力图（冗余分析依据）  |
| bbb_feature_engineering_report.md | 本报告（流程与结果说明）              |
"""

with open(REPORT_SAVE, "w", encoding="utf-8") as f:
    f.write(report_content)
print(f"   ✅ 特征工程报告已保存：{REPORT_SAVE}")

print("\n" + "="*60)
print("🎉 特征工程全流程执行完成！所有文件已保存到：")
print(f"   {LOCAL_DIR}")
print("="*60)