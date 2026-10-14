# 1. 导入相关库
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 2. 读取数据
data = pd.read_csv(r"C:\Users\ahdezneytiry-boop1\Desktop\实验三\diabetes (1).csv")  
print("数据集预览：")
print(data.head())

# 3. 读取变量名
label_need = data.keys()
print("\n变量名：")
print(label_need.tolist())

# 4. 定义X（特征）和y（标签）数据
X = data[label_need].values[:, 0:8]  # 前8列为特征
y = data[label_need].values[:, 8].astype(int)  # 第9列为标签（转为整数类型）
print("\n特征矩阵形状：", X.shape)
print("标签数组形状：", y.shape)

# 5. 分离训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n训练集大小：X_train=", X_train.shape, " y_train=", y_train.shape)
print("测试集大小：X_test=", X_test.shape, " y_test=", y_test.shape)

# 6. 定义欧式距离计算函数
def compute_distances(X_test, X_train):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    # 初始化距离矩阵
    dists = np.zeros((num_test, num_train))
    # 矩阵运算优化欧式距离计算
    M = np.dot(X_test, X_train.T)
    te = np.square(X_test).sum(axis=1)
    tr = np.square(X_train).sum(axis=1)
    dists = np.sqrt(-2 * M + tr + np.matrix(te).T)
    return dists

# 7. 计算欧式距离
dists = compute_distances(X_test, X_train)
print("\n距离矩阵形状：", dists.shape)
print("距离矩阵前5行前5列：")
print(dists[:5, :5])

# 8. 可视化距离矩阵
plt.figure(figsize=(8, 6))
plt.imshow(dists, interpolation='none', cmap='viridis')
plt.xlabel('训练集样本索引')
plt.ylabel('测试集样本索引')
plt.title('测试集与训练集的欧式距离矩阵')
plt.colorbar(label='欧式距离')
plt.show()

# 9. 手动实现KNN预测（设置k值，可根据需求调整）
k = 5  # 近邻数量，可修改优化
def knn_predict(dists, y_train, k):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        # 对距离排序，取前k个近邻的标签
        labels = y_train[np.argsort(dists[i, :])].flatten()
        closest_y = labels[:k]
        # 投票统计最多的类别
        c = Counter(closest_y)
        y_pred[i] = c.most_common(1)[0][0]
    return y_pred.astype(int)

# 执行预测
y_pred_manual = knn_predict(dists, y_train, k)
print("\n手动实现KNN预测结果（前10个样本）：", y_pred_manual[:10])

# 10. 查看手动实现的正确率
accuracy_manual = accuracy_score(y_test, y_pred_manual)
print(f"\n手动实现KNN正确率（k={k}）：{accuracy_manual:.4f}")

# 11. 手动实现交叉验证（验证k值合理性）
def cross_validate_knn(X, y, k, cv=5):
    scores = []
    # 分割为cv个折
    fold_size = X.shape[0] // cv
    for i in range(cv):
        # 划分训练折和验证折
        val_start = i * fold_size
        val_end = val_start + fold_size
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_fold_train = np.concatenate([X[:val_start], X[val_end:]])
        y_fold_train = np.concatenate([y[:val_start], y[val_end:]])
        # 计算距离并预测
        dists_fold = compute_distances(X_val, X_fold_train)
        y_pred_fold = knn_predict(dists_fold, y_fold_train, k)
        # 计算正确率
        score = accuracy_score(y_val, y_pred_fold)
        scores.append(score)
    return np.mean(scores), np.std(scores)

cv_mean, cv_std = cross_validate_knn(X, y, k)
print(f"\n手动实现5折交叉验证正确率：{cv_mean:.4f}（±{cv_std:.4f}）")

# 12. scikit-learn实现KNN
knn_sklearn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"\nscikit-learn KNN正确率（k={k}）：{accuracy_sklearn:.4f}")

# 13. scikit-learn交叉验证
cv_scores = cross_val_score(knn_sklearn, X, y, cv=5)
print(f"scikit-learn 5折交叉验证正确率：{cv_scores.mean():.4f}（±{cv_scores.std():.4f}）")

# 14. scikit-learn实现KNN变种（带权重的KNN）
knn_weighted = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
knn_weighted.fit(X_train, y_train)
y_pred_weighted = knn_weighted.predict(X_test)
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
print(f"\n带距离权重的KNN正确率（k={k}）：{accuracy_weighted:.4f}")