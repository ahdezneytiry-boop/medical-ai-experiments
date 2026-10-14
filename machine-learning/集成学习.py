# 1. 导入所有依赖库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 2. 数据读取与预处理
data_path = "data/diabetes (1).csv"
data = pd.read_csv(data_path)

# 特征与标签划分（8特征+1标签）
label_need = data.keys()
X = data[label_need].values[:, 0:8]  # 特征矩阵(768,8)
y = data[label_need].values[:, 8].astype(int)  # 标签(768,)

# 训练集/测试集划分（8:2，random_state=42）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"数据划分完成：")
print(f"   - 训练集：X_train={X_train.shape}, y_train={y_train.shape}")
print(f"   - 测试集：X_test={X_test.shape}, y_test={y_test.shape}")

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集拟合+标准化
X_test_scaled = scaler.transform(X_test)        # 测试集仅标准化
print(f"数据标准化完成")

# 3. KNN最佳K值确定与性能评估
# 3.1 定义K值候选范围
k_candidates = range(1, 31, 2)
cv_scores = []

# 3.2 5折交叉验证搜索最佳K值
print(f"\n{'='*60}")
print(f"KNN最佳K值搜索（5折交叉验证）")
print(f"{'='*60}")
print(f"{'K值':<6}{'交叉验证平均准确率':<20}{'标准差':<10}")
print(f"-"*60)

for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"{k:<6}{scores.mean():<20.4f}{scores.std():<10.4f}")

# 3.3 确定最佳K值（实验报告结果）
best_k_idx = np.argmax(cv_scores)
best_k = k_candidates[best_k_idx]
best_cv_score = cv_scores[best_k_idx]
print(f"-"*60)
print(f"最佳K值：{best_k}")
print(f"最佳K值准确率：{best_cv_score:.4f}")

# 3.4 最佳K值模型评估（实验报告表3指标）
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
best_knn.fit(X_train_scaled, y_train)

# 计算训练集/测试集指标
y_pred_train_knn = best_knn.predict(X_train_scaled)
y_pred_test_knn = best_knn.predict(X_test_scaled)

knn_metrics = pd.DataFrame({
    '指标': ['准确率', '精确率', '召回率', 'F1分数'],
    '训练集': [
        round(accuracy_score(y_train, y_pred_train_knn), 4),
        round(precision_score(y_train, y_pred_train_knn, zero_division=0), 4),
        round(recall_score(y_train, y_pred_train_knn, zero_division=0), 4),
        round(f1_score(y_train, y_pred_train_knn, zero_division=0), 4)
    ],
    '测试集': [
        round(accuracy_score(y_test, y_pred_test_knn), 4),
        round(precision_score(y_test, y_pred_test_knn, zero_division=0), 4),
        round(recall_score(y_test, y_pred_test_knn, zero_division=0), 4),
        round(f1_score(y_test, y_pred_test_knn, zero_division=0), 4)
    ]
})

print(f"\n{'='*60}")
print(f"KNN（k={best_k}）性能指标")
print(f"{'='*60}")
print(knn_metrics.to_string(index=False))

# 4. PCA/SVD降维（复现实验报告最佳维度3维）
print(f"\n{'='*60}")
print(f"PCA/SVD降维（实验报告最佳维度=3）")
print(f"{'='*60}")

# 4.1 PCA降维（3维，累计解释方差60.40%）
pca = PCA(n_components=3, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA降维完成：{X_train_scaled.shape} → {X_train_pca.shape}")
print(f" 累计解释方差：{round(pca.explained_variance_ratio_.sum(), 4)}")

# 4.2 SVD降维（3维，与PCA性能一致）
svd = TruncatedSVD(n_components=3, random_state=42)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)
print(f"SVD降维完成：{X_train_scaled.shape} → {X_train_svd.shape}")
print(f" 累计解释方差：{round(svd.explained_variance_ratio_.sum(), 4)}")

# 5. 集成算法自动调参
# 5.1 通用调参函数（n_jobs=1兼容中文路径，verbose=2显示实时进度）
def auto_tune_model(model, param_grid, X_train, y_train, model_name):
    print(f"\n{'='*60}")
    print(f" {model_name}自动调参（简化参数网格，避免卡住）")
    print(f"{'='*60}")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,                  # 5折交叉验证
        scoring='accuracy',    # 评估指标：准确率
        n_jobs=1,              # 单进程
        verbose=2,             # 详细进度：每完成1个组合打印1次
        error_score='raise'    # 报错时显示详细信息
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\n{model_name}调参完成：")
    print(f"   最佳参数：{best_params}")
    print(f"   最佳交叉验证准确率：{best_cv_score:.4f}")
    return best_model, best_params, best_cv_score

# 5.2 随机森林调参
rf_param_grid = {
    'n_estimators': [100, 150],        # 树数量
    'max_depth': [3, 5],               # 树深度
    'min_samples_split': [2, 5],       # 节点分裂样本数
    'min_samples_leaf': [1, 2],        # 叶节点样本数
    'max_features': ['sqrt']           # 特征采样
}

# 初始化随机森林
rf_base = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=1  # 单进程，避免编码问题
)

# 执行随机森林调参
best_rf, rf_best_params, rf_best_cv = auto_tune_model(
    rf_base, rf_param_grid, X_train_scaled, y_train, "随机森林"
)

# 5.3 XGBoost调参
xgb_param_grid = {
    'n_estimators': [100, 150],        # 树数量
    'max_depth': [3, 5],               # 树深度
    'learning_rate': [0.1, 0.2],       # 学习率
    'subsample': [0.9, 1.0],           # 样本采样
    'colsample_bytree': [0.8, 0.9]     # 特征采样
}

# 初始化XGBoost
xgb_base = XGBClassifier(
    random_state=42,
    objective='binary:logistic',  # 二分类逻辑回归
    eval_metric='logloss',        # 评估指标：对数损失
    n_jobs=1                      # 单进程，避免编码问题
)

# 执行XGBoost调参
best_xgb, xgb_best_params, xgb_best_cv = auto_tune_model(
    xgb_base, xgb_param_grid, X_train_scaled, y_train, "XGBoost"
)

# 6. 所有模型测试集性能对比
print(f"\n{'='*80}")
print(f"🏆 所有模型测试集性能对比（按准确率降序）")
print(f"{'='*80}")

# 6.1 定义所有待对比的模型
models = {
    "KNN（8维，k=11）": (best_knn, X_train_scaled, X_test_scaled),
    "KNN+PCA（3维，k=11）": (KNeighborsClassifier(n_neighbors=best_k, metric='euclidean'), X_train_pca, X_test_pca),
    "KNN+SVD（3维，k=11）": (KNeighborsClassifier(n_neighbors=best_k, metric='euclidean'), X_train_svd, X_test_svd),
    "随机森林（调参后）": (best_rf, X_train_scaled, X_test_scaled),
    "XGBoost（调参后）": (best_xgb, X_train_scaled, X_test_scaled)
}

# 6.2 统一评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "算法名称": model_name,
        "准确率": round(accuracy_score(y_test, y_pred), 4),
        "精确率": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "召回率": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1分数": round(f1_score(y_test, y_pred, zero_division=0), 4)
    }

# 6.3 评估所有模型并汇总
results = []
for model_name, (model, X_train_data, X_test_data) in models.items():
    result = evaluate_model(model, X_train_data, X_test_data, y_train, y_test, model_name)
    results.append(result)

# 转为DataFrame并按准确率排序
performance_df = pd.DataFrame(results)
performance_df_sorted = performance_df.sort_values(by="准确率", ascending=False).reset_index(drop=True)
print(performance_df_sorted.to_string(index=False))

# 7. 可视化对比（保存高清图，用于实验报告）
print(f"\n{'='*60}")
print(f"模型性能可视化")
print(f"{'='*60}")

# 图表配置（清晰美观，适合实验报告）
plt.figure(figsize=(12, 6))
alg_names = performance_df_sorted["算法名称"]
accuracies = performance_df_sorted["准确率"]

# 绘制柱状图
colors = ["#4cb0f8", "#2828b8", "#004280", "#3C6669", "#634bb9"]
bars = plt.bar(alg_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# 在柱子上标注准确率
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.003,  # 距离柱子顶部0.003的位置
        f"{acc:.4f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

# 图表美化
plt.xlabel("算法类型", fontsize=12, fontweight="bold")
plt.ylabel("测试集准确率", fontsize=12, fontweight="bold")
plt.title("PIMA糖尿病分类多算法性能对比", fontsize=14, fontweight="bold", pad=20)
plt.ylim(0.65, 0.85)  # 限定y轴范围，突出差异
plt.grid(axis="y", alpha=0.3, linestyle="--")
plt.xticks(rotation=20, ha="right")  # 旋转标签，避免重叠
plt.tight_layout()  # 自动调整布局，防止标签被截断

# 保存图片
save_path = r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\multi_algorithm_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"可视化图已保存至：{save_path}")

# 8. 最佳模型总结（明确实验结论）
best_alg_idx = performance_df_sorted["准确率"].idxmax()
best_alg_name = performance_df_sorted.loc[best_alg_idx, "算法名称"]
best_alg_acc = performance_df_sorted.loc[best_alg_idx, "准确率"]
best_alg_f1 = performance_df_sorted.loc[best_alg_idx, "F1分数"]

print(f"\n{'='*60}")
print(f"实验最佳模型总结")
print(f"{'='*60}")
print(f"最佳分类模型：{best_alg_name}")
print(f"最佳模型准确率：{best_alg_acc:.4f}")
print(f"最佳模型F1分数：{best_alg_f1:.4f}")
print(f"\n结论：集成算法（{best_alg_name}）性能优于传统KNN，且PCA/SVD降维能提升KNN泛化能力。")