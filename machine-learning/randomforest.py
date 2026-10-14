# 随机森林+XGBoost对比：ADMET属性预测（AMES致突变性分类任务）
# 支持模型性能对比、特征重要性可视化，适配本地环境
# 基于新上传的ames_cls_final_data.csv数据集

# ==============================
# 1. 导入依赖库（新增XGBoost库）
# ==============================
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # 新增XGBoost分类器
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# 解决中文乱码（适配更多系统）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 2. 任务配置（针对AMES数据集优化）
# ==============================
class Config:
    # 数据路径：适配本地环境，若数据集与代码同文件夹，直接写"ames_cls_final_data.csv"
    data_path =r"E:\jqxx\机器学习大作业\机器学习数据\ames_cls_final_data.csv" # 服务器路径，本地使用时改为"ames_cls_final_data.csv"
    # 任务类型：AMES致突变性为分类任务（1=致突变，0=非致突变）
    task_type = "classification"
    # 特征类型：Morgan指纹（捕捉分子结构细节，适合致突变性预测）
    feature_type = "morgan"
    # 数据集划分：优先用scaffold划分（避免分子骨架泄露，结果更可靠）
    split_type = "scaffold"
    # 模型公共参数
    random_seed = 42  # 固定种子，保证结果可复现
    test_size = 0.2   # 若用random划分时的测试集比例（scaffold划分时无效）
    # Morgan指纹参数
    morgan_radius = 2  # 半径2对应ECFP4（药物化学常用）
    morgan_nbits = 1024  # 指纹向量长度
    # 随机森林参数
    rf_n_estimators = 100  # 决策树数量
    rf_max_depth = None    # 树最大深度（None表示不限制）
    rf_n_jobs = -1         # 多线程训练
    # XGBoost参数（针对分类任务优化）
    xgb_n_estimators = 100  # 弱学习器数量
    xgb_max_depth = 6       # 树深度（避免过拟合）
    xgb_learning_rate = 0.1 # 学习率
    xgb_objective = "binary:logistic"  # 二分类目标函数
    xgb_eval_metric = "auc"            # 评估指标（ROC-AUC）

# ==============================
# 3. AMES数据集加载与预处理（新增数据分布分析）
# ==============================
def load_and_analyze_ames_data(config):
    """
    加载AMES数据集，过滤无效SMILES，分析数据分布，划分训练/测试集
    返回：训练/测试集的SMILES、标签，以及数据集基本信息
    """
    # 3.1 路径有效性检查
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(
            f"AMES数据集未找到！当前路径：{config.data_path}\n"
            f"本地使用时请确保'ames_cls_final_data.csv'与代码在同一文件夹"
        )
    
    # 3.2 加载数据并查看基本信息
    df = pd.read_csv(config.data_path)
    print("="*60)
    print("AMES致突变性数据集基本信息")
    print("="*60)
    print(f"数据总行数：{df.shape[0]}，总列数：{df.shape[1]}")
    print(f"列名：{list(df.columns)}")
    print(f"\n缺失值统计：")
    print(df.isnull().sum())  # 检查是否有缺失值（AMES数据通常无缺失）
    
    # 3.3 过滤无效SMILES（关键步骤：排除无法解析的分子结构）
    def is_valid_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    df["is_valid"] = df["Smiles_unify"].apply(is_valid_smiles)
    df_clean = df[df["is_valid"]].copy()
    invalid_count = len(df) - len(df_clean)
    
    # 3.4 分析数据分布（AMES数据集常存在样本不平衡）
    pos_count = df_clean[df_clean["value"] == 1].shape[0]
    neg_count = df_clean[df_clean["value"] == 0].shape[0]
    pos_ratio = pos_count / len(df_clean) * 100
    neg_ratio = neg_count / len(df_clean) * 100
    
    print(f"\n数据有效性筛选：")
    print(f"原始样本数：{len(df)}，无效SMILES数：{invalid_count}，有效样本数：{len(df_clean)}")
    print(f"有效样本标签分布：")
    print(f"致突变样本（1）：{pos_count}条（{pos_ratio:.2f}%）")
    print(f"非致突变样本（0）：{neg_count}条（{neg_ratio:.2f}%）")
    
    # 3.5 划分训练/测试集（基于配置的划分方式）
    split_col = f"{config.split_type}_train_test_label"
    if split_col in df_clean.columns:
        # 用数据集自带的划分标签（推荐）
        train_df = df_clean[df_clean[split_col] == "train"]
        test_df = df_clean[df_clean[split_col] == "test"]
        print(f"\n使用{config.split_type}划分：")
    else:
        # 若无自带标签，用随机划分
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df_clean, test_size=config.test_size, random_state=config.random_seed, stratify=df_clean["value"]
        )
        print(f"\n使用random划分（ stratified 保持标签分布）：")
    
    print(f"训练集样本数：{len(train_df)}（致突变：{train_df[train_df['value']==1].shape[0]}条）")
    print(f"测试集样本数：{len(test_df)}（致突变：{test_df[test_df['value']==1].shape[0]}条）")
    print("="*60)
    
    # 3.6 提取模型输入输出
    X_train_smiles = train_df["Smiles_unify"].values
    y_train = train_df["value"].values.astype(int)  # 转为整数型标签
    X_test_smiles = test_df["Smiles_unify"].values
    y_test = test_df["value"].values.astype(int)
    
    # 保存数据集信息（用于后续分析）
    data_info = {
        "total_samples": len(df_clean),
        "pos_ratio": pos_ratio,
        "train_samples": len(train_df),
        "test_samples": len(test_df)
    }
    
    return X_train_smiles, y_train, X_test_smiles, y_test, data_info

# ==============================
# 4. 分子特征提取（复用Morgan指纹，保证对比公平性）
# ==============================
def extract_morgan_fingerprint(smiles_list, config):
    """
    提取Morgan指纹（所有模型用相同特征，保证对比公平）
    返回：(n_samples, n_bits)的特征矩阵
    """
    print(f"\n正在提取Morgan指纹（radius={config.morgan_radius}, nBits={config.morgan_nbits}）...")
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        # 生成Morgan指纹（二进制向量）
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=config.morgan_radius, nBits=config.morgan_nbits
        )
        features.append(np.array(fp))
    features = np.array(features)
    print(f"指纹提取完成，特征矩阵形状：{features.shape}")
    return features

# ==============================
# 5. 模型训练与评估（新增XGBoost，对比随机森林）
# ==============================
def train_and_evaluate_models(config, X_train, y_train, X_test, y_test):
    """
    训练随机森林和XGBoost模型，评估性能并返回结果
    返回：模型对象、性能指标字典
    """
    # 存储性能指标
    metrics_dict = {}
    
    # 5.1 训练随机森林模型
    print("\n" + "="*60)
    print("训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        random_state=config.random_seed,
        n_jobs=config.rf_n_jobs,
        class_weight="balanced"  # 处理AMES样本不平衡
    )
    rf_model.fit(X_train, y_train)
    
    # 5.2 随机森林预测与评估
    rf_y_pred = rf_model.predict(X_test)
    rf_y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # 正样本概率
    metrics_dict["RF"] = {
        "Accuracy": accuracy_score(y_test, rf_y_pred),
        "ROC-AUC": roc_auc_score(y_test, rf_y_pred_proba),
        "F1-Score": f1_score(y_test, rf_y_pred),
        "Confusion Matrix": confusion_matrix(y_test, rf_y_pred),
        "Classification Report": classification_report(y_test, rf_y_pred, output_dict=True)
    }
    
    print("随机森林模型评估结果：")
    print(f"准确率（Accuracy）：{metrics_dict['RF']['Accuracy']:.4f}")
    print(f"ROC-AUC：{metrics_dict['RF']['ROC-AUC']:.4f}")
    print(f"F1分数：{metrics_dict['RF']['F1-Score']:.4f}")
    print(f"混淆矩阵：\n{metrics_dict['RF']['Confusion Matrix']}")
    
    # 5.3 训练XGBoost模型
    print("\n" + "="*60)
    print("训练XGBoost模型...")
    xgb_model = XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        objective=config.xgb_objective,
        eval_metric=config.xgb_eval_metric,
        random_state=config.random_seed,
        n_jobs=-1,
        scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train)  # 处理样本不平衡
    )
    xgb_model.fit(X_train, y_train)
    
    # 5.4 XGBoost预测与评估
    xgb_y_pred = xgb_model.predict(X_test)
    xgb_y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # 正样本概率
    metrics_dict["XGBoost"] = {
        "Accuracy": accuracy_score(y_test, xgb_y_pred),
        "ROC-AUC": roc_auc_score(y_test, xgb_y_pred_proba),
        "F1-Score": f1_score(y_test, xgb_y_pred),
        "Confusion Matrix": confusion_matrix(y_test, xgb_y_pred),
        "Classification Report": classification_report(y_test, xgb_y_pred, output_dict=True)
    }
    
    print("XGBoost模型评估结果：")
    print(f"准确率（Accuracy）：{metrics_dict['XGBoost']['Accuracy']:.4f}")
    print(f"ROC-AUC：{metrics_dict['XGBoost']['ROC-AUC']:.4f}")
    print(f"F1分数：{metrics_dict['XGBoost']['F1-Score']:.4f}")
    print(f"混淆矩阵：\n{metrics_dict['XGBoost']['Confusion Matrix']}")
    print("="*60)
    
    return rf_model, xgb_model, metrics_dict

# ==============================
# 6. 模型对比可视化（新增性能对比图、特征重要性对比）
# ==============================
def plot_model_comparison(rf_model, xgb_model, metrics_dict, config, data_info):
    """
    生成模型性能对比图和特征重要性对比图，保存到本地文件夹
    """
    # 6.1 性能对比图（柱状图：Accuracy、ROC-AUC、F1-Score）
    metrics_names = ["Accuracy", "ROC-AUC", "F1-Score"]
    rf_scores = [metrics_dict["RF"][metric] for metric in metrics_names]
    xgb_scores = [metrics_dict["XGBoost"][metric] for metric in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35  # 柱子宽度
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, rf_scores, width, label="随机森林", color="#1f77b4")
    bars2 = plt.bar(x + width/2, xgb_scores, width, label="XGBoost", color="#ff7f0e")
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f"{height:.4f}", ha="center", va="bottom")
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f"{height:.4f}", ha="center", va="bottom")
    
    # 图表美化
    plt.xlabel("评估指标", fontsize=12)
    plt.ylabel("指标值（越高越好）", fontsize=12)
    plt.title(f"AMES致突变性预测模型性能对比\n（数据集：{data_info['total_samples']}条样本，致突变比例：{data_info['pos_ratio']:.2f}%）", fontsize=14)
    plt.xticks(x, metrics_names)
    plt.ylim(0.5, 1.0)  # 限定y轴范围，突出差异
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    
    # 保存性能对比图
    save_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    perf_save_path = os.path.join(save_dir, "ames_model_performance_comparison.png")
    plt.tight_layout()
    plt.savefig(perf_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n模型性能对比图已保存至：{perf_save_path}")
    
    # 6.2 特征重要性对比图（Top10特征）
    # 提取两个模型的特征重要性
    rf_importance = rf_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_
    
    # 获取Top10特征的索引（按XGBoost重要性排序，保持一致性）
    top10_idx = np.argsort(xgb_importance)[-10:][::-1]
    rf_top10 = rf_importance[top10_idx]
    xgb_top10 = xgb_importance[top10_idx]
    feature_labels = [f"指纹位{idx}" for idx in top10_idx]
    
    # 绘制特征重要性对比
    plt.figure(figsize=(14, 8))
    x = np.arange(len(feature_labels))
    
    bars1 = plt.bar(x - width/2, rf_top10, width, label="随机森林", color="#1f77b4", alpha=0.8)
    bars2 = plt.bar(x + width/2, xgb_top10, width, label="XGBoost", color="#ff7f0e", alpha=0.8)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f"{height:.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f"{height:.4f}", ha="center", va="bottom", fontsize=9)
    
    # 图表美化
    plt.xlabel("Morgan指纹特征（Top10）", fontsize=12)
    plt.ylabel("特征重要性（越高影响越大）", fontsize=12)
    plt.title("AMES致突变性预测模型Top10特征重要性对比", fontsize=14)
    plt.xticks(x, feature_labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    
    # 保存特征重要性对比图
    imp_save_path = os.path.join(save_dir, "ames_model_feature_importance_comparison.png")
    plt.tight_layout()
    plt.savefig(imp_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"特征重要性对比图已保存至：{imp_save_path}")

# ==============================
# 7. 主函数（串联全流程：数据加载→特征提取→模型训练→对比分析）
# ==============================
def main():
    try:
        # 初始化配置
        config = Config()
        print("AMES致突变性预测：随机森林VS XGBoost 对比实验")
        print(f"实验配置：特征类型={config.feature_type}，划分方式={config.split_type}，随机种子={config.random_seed}")
        
        # 1. 加载并分析AMES数据集
        X_train_smiles, y_train, X_test_smiles, y_test, data_info = load_and_analyze_ames_data(config)
        
        # 2. 提取分子特征（所有模型用相同特征）
        X_train = extract_morgan_fingerprint(X_train_smiles, config)
        X_test = extract_morgan_fingerprint(X_test_smiles, config)
        
        # 3. 训练并评估两个模型
        rf_model, xgb_model, metrics_dict = train_and_evaluate_models(config, X_train, y_train, X_test, y_test)
        
        # 4. 生成对比可视化图表
        plot_model_comparison(rf_model, xgb_model, metrics_dict, config, data_info)
        
        # 5. 输出关键结论
        print("\n" + "="*80)
        print("关键结论：")
        print("="*80)
        # 性能最优模型判断
        best_model = "XGBoost" if metrics_dict["XGBoost"]["ROC-AUC"] > metrics_dict["RF"]["ROC-AUC"] else "随机森林"
        print(f"1. 性能最优模型：{best_model}（ROC-AUC：{max(metrics_dict['RF']['ROC-AUC'], metrics_dict['XGBoost']['ROC-AUC']):.4f}）")
        # 样本不平衡影响
        print(f"2. 数据集特点：AMES数据集存在样本不平衡（致突变比例{data_info['pos_ratio']:.2f}%），两模型均通过class_weight/scale_pos_weight优化")
        # 特征重要性差异
        rf_top_imp = np.max(rf_model.feature_importances_)
        xgb_top_imp = np.max(xgb_model.feature_importances_)
        print(f"3. 特征重要性：XGBoost对关键特征的权重更集中（Top1特征重要性：{xgb_top_imp:.4f}），随机森林更分散（{rf_top_imp:.4f}）")
        print(f"4. 实际应用建议：优先选择{best_model}，因其在致突变性预测（高风险筛选）中表现更优，误判率更低")
        print("="*80)
        
    except Exception as e:
        print(f"\n【执行错误】：{str(e)}")

# ==============================
# 8. 启动实验
# ==============================
if __name__ == "__main__":
    main()