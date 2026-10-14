import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 用简单易上手的GCN作为GNN基础模型
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# -------------------------- 1. 数据输入（按你的数据格式调整）--------------------------
# 假设预处理后的数据为CSV格式，包含：
# - compound_id：化合物ID（用于结果追溯）
# - mol_struct_vec_0~mol_struct_vec_99：分子结构向量（共100维，预处理后的数据）
# - logP、molecular_weight、h_bond_donor：标准化后的理化特征
# - oral_bioavailability：真实口服生物利用度（目标变量，回归任务）
data = pd.read_csv("data/bbb_cls_final_data.csv")

# 分离特征与目标变量
# 分子结构向量（假设100维，根据实际维度调整列名）
mol_struct_cols = [col for col in data.columns if col.startswith("mol_struct_vec_")]
mol_struct_data = data[mol_struct_cols].values  # 形状：(样本数, 100)
# 标准化理化特征（已预处理，直接使用）
phys_chem_cols = ["logP", "molecular_weight", "h_bond_donor"]
phys_chem_data = data[phys_chem_cols].values  # 形状：(样本数, 3)
# 目标变量（口服生物利用度）
y = data["oral_bioavailability"].values  # 形状：(样本数,)
# 化合物ID（用于结果输出）
compound_ids = data["compound_id"].values

# 8:2划分训练集/测试集（回归任务用shuffle保持分布一致，无需分层抽样）
train_idx, test_idx = train_test_split(
    range(len(data)), test_size=0.2, random_state=42, shuffle=True
)

# 拆分训练/测试集的结构特征、理化特征、目标变量、ID
# 训练集
train_mol_struct = mol_struct_data[train_idx]
train_phys_chem = phys_chem_data[train_idx]
train_y = y[train_idx]
train_ids = compound_ids[train_idx]
# 测试集
test_mol_struct = mol_struct_data[test_idx]
test_phys_chem = phys_chem_data[test_idx]
test_y = y[test_idx]
test_ids = compound_ids[test_idx]

# -------------------------- 2. 参数初始化（按你指定的参数设置）--------------------------
# GNN参数（隐藏层256，学习率0.001，轮次100，批大小32）
gnn_params = {
    "input_dim": mol_struct_data.shape[1],  # 分子结构向量维度（100）
    "hidden_dim": 256,
    "output_dim": 128,  # GNN输出的结构特征维度（后续与理化特征融合）
    "lr": 0.001,
    "epochs": 100,
    "batch_size": 32
}
# XGBoost参数（树数量100，学习率0.1，最大深度5，正则化λ=0.1）
xgb_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "reg_lambda": 0.1,
    "objective": "reg:squarederror",  # 回归任务目标函数（MSE）
    "random_state": 42,
    "early_stopping_rounds": 5  # 早停防过拟合
}

# -------------------------- 3. GNN模型定义与训练（提取分子结构特征）--------------------------
# 定义简单GCN模型（适合首次建模，易理解且效果稳定）
class GNN_MolFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # 第一层GCN
        self.conv2 = GCNConv(hidden_dim, output_dim)  # 第二层GCN

    def forward(self, x):
        # 注：此处简化处理——假设输入的mol_struct_vec已为“节点特征聚合后的分子级向量”
        # 若原始数据是分子图（含节点、边），需补充邻接矩阵处理（见代码末尾备注）
        x = F.relu(self.conv1(x, torch.eye(x.shape[0]).nonzero().t()))  # 用单位矩阵模拟全连接（简化）
        x = self.conv2(x, torch.eye(x.shape[0]).nonzero().t())
        return x  # 输出：(样本数, 128)——GNN提取的结构特征

# 初始化GNN模型，判断GPU是否可用（必须用GPU加速，否则训练慢）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise Warning("未检测到GPU，GNN训练会极慢！建议使用支持CUDA的GPU。")
gnn_model = GNN_MolFeatureExtractor(
    gnn_params["input_dim"],
    gnn_params["hidden_dim"],
    gnn_params["output_dim"]
).to(device)

# 准备GNN输入数据（转为PyTorch张量）
train_mol_tensor = torch.tensor(train_mol_struct, dtype=torch.float32).to(device)
test_mol_tensor = torch.tensor(test_mol_struct, dtype=torch.float32).to(device)

# GNN训练配置（优化器、损失函数）
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=gnn_params["lr"])
criterion = torch.nn.MSELoss()  # GNN以“重构分子特征”为目标（简化训练，实际可结合目标变量）

# 训练GNN模型
gnn_model.train()
for epoch in range(gnn_params["epochs"]):
    optimizer.zero_grad()
    # 前向传播：输入分子结构向量，输出GNN特征
    gnn_train_features = gnn_model(train_mol_tensor)
    # 损失计算（简化：让GNN输出逼近原始结构向量，实际可替换为“与目标变量相关的损失”）
    loss = criterion(gnn_train_features, train_mol_tensor)
    # 反向传播与优化
    loss.backward()
    optimizer.step()
    # 每10轮打印训练损失（监控训练稳定性）
    if (epoch + 1) % 10 == 0:
        print(f"GNN训练轮次 {epoch+1}/{gnn_params['epochs']}，损失：{loss.item():.4f}")

# 提取训练/测试集的GNN结构特征（关闭梯度计算，提高效率）
with torch.no_grad():
    gnn_train_features = gnn_model(train_mol_tensor).cpu().numpy()  # (训练样本数, 128)
    gnn_test_features = gnn_model(test_mol_tensor).cpu().numpy()    # (测试样本数, 128)

# -------------------------- 4. 特征融合（GNN结构特征 + 理化特征）--------------------------
# 拼接特征：GNN输出的128维结构特征 + 3维理化特征 → 131维融合特征
train_fused_features = np.hstack([gnn_train_features, train_phys_chem])  # (训练样本数, 131)
test_fused_features = np.hstack([gnn_test_features, test_phys_chem])    # (测试样本数, 131)

# -------------------------- 5. XGBoost训练与预测--------------------------
# 准备XGBoost输入格式（DMatrix，提高训练效率）
dtrain = xgb.DMatrix(train_fused_features, label=train_y)
dtest = xgb.DMatrix(test_fused_features, label=test_y)

# 划分XGBoost验证集（用于早停）
val_idx = train_test_split(range(len(train_fused_features)), test_size=0.2, random_state=42)[1]
dval = xgb.DMatrix(train_fused_features[val_idx], label=train_y[val_idx])

# 训练XGBoost模型（早停基于验证集损失）
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    evals=[(dval, "validation")],  # 验证集监控
    verbose_eval=10  # 每10轮打印验证集损失
)

# 在测试集预测口服生物利用度
test_pred = xgb_model.predict(dtest)  # 预测值：(测试样本数,)

# -------------------------- 6. 结果输出（生成表格 + 计算评估指标）--------------------------
# 计算误差（MAE、RMSE，与目标函数MSE一致）
test_mae = mean_absolute_error(test_y, test_pred)
test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
print(f"\n测试集评估结果：MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

# 生成结果表格（含ID、真实值、预测值、误差）
result_df = pd.DataFrame({
    "compound_id": test_ids,
    "真实口服生物利用度": test_y,
    "预测口服生物利用度": test_pred.round(4),
    "绝对误差（MAE）": np.abs(test_y - test_pred).round(4),
    "平方误差（MSE）": (test_y - test_pred)**2.round(4)
})

# 保存结果到Excel（方便后续分析）
result_df.to_excel("oral_bioavailability_prediction_results.xlsx", index=False)
print("\n预测结果已保存至 oral_bioavailability_prediction_results.xlsx")