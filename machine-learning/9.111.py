# 1. 导入必备库（修复：添加import warnings）
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes  # 开源糖尿病数据集
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # 提前导入，避免函数内导入的不规范（非报错原因，仅优化）

# 解决中文显示+忽略无关警告（修复：已导入warnings）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 现在可正常使用


# 2. 加载并探索开源糖尿病数据集
def load_explore_diabetes():
    """加载sklearn开源糖尿病数据集，返回特征、目标及原始DataFrame"""
    diabetes = load_diabetes()
    # 转换为DataFrame（便于探索）
    df = pd.DataFrame(
        data=np.c_[diabetes.data, diabetes.target],
        columns=list(diabetes.feature_names) + ['血糖水平']  # 特征列+目标列
    )
    X = df.drop(columns=['血糖水平'])  # 特征矩阵（10个生理特征）
    y = df['血糖水平']  # 目标变量（血糖水平）

    # 打印数据集基本信息
    print("="*60)
    print("【开源糖尿病数据集基本信息】")
    print("="*60)
    print(f"样本数量：{df.shape[0]} | 特征数量：{df.shape[1]-1}（含1个目标列）")
    print(f"\n特征名称及含义：")
    feature_info = {
        'age': '年龄', 'sex': '性别', 'bmi': '身体质量指数（BMI）',
        'bp': '平均血压', 's1': '血清总胆固醇', 's2': '低密度脂蛋白',
        's3': '高密度脂蛋白', 's4': '总胆固醇/高密度脂蛋白', 's5': '血清甘油三酯',
        's6': '血糖水平的对数（非目标）'
    }
    for feat in diabetes.feature_names:
        print(f"  {feat}：{feature_info[feat]}")
    print(f"\n目标变量（血糖水平）统计：")
    print(df['血糖水平'].describe().round(2))

    # 分析特征与血糖的相关性（选择简单回归特征）
    corr_with_target = df.corr()['血糖水平'].sort_values(ascending=False)
    print(f"\n【特征与血糖水平的相关性（降序）】")
    print(corr_with_target.round(4))

    # 选择相关性最高的特征（BMI，相关系数≈0.58）
    best_feat = corr_with_target.index[1]  # index[0]是目标自身
    X_simple = X[[best_feat]]  # 简单回归的单特征
    print(f"\n选择简单线性回归的特征：{best_feat}（与血糖相关性：{corr_with_target[best_feat]:.4f}）")
    return X, X_simple, y, df, best_feat


# 3. 数据拆分与模型训练
def train_models(X, X_simple, y):
    """拆分训练集/测试集，训练3类线性回归模型"""
    # 拆分数据（80%训练，20%测试，随机种子保证可复现）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_simple_train, X_simple_test, _, _ = train_test_split(
        X_simple, y, test_size=0.2, random_state=42  # 简单回归数据
    )

    print(f"\n" + "="*60)
    print("【数据集拆分结果】")
    print("="*60)
    print(f"训练集：{X_train.shape} | 测试集：{X_test.shape}")

    # ---------------------- 模型1：简单线性回归（BMI → 血糖） ----------------------
    simple_model = LinearRegression()
    simple_model.fit(X_simple_train, y_train)
    # 预测
    y_simple_train_pred = simple_model.predict(X_simple_train)
    y_simple_test_pred = simple_model.predict(X_simple_test)
    # 评估指标
    simple_metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_simple_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_simple_test_pred)),
        'test_r2': r2_score(y_test, y_simple_test_pred),
        'equation': f"血糖 = {simple_model.intercept_:.2f} + {simple_model.coef_[0]:.2f} × BMI"
    }

    # ---------------------- 模型2：多元线性回归（全10特征） ----------------------
    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    y_multi_test_pred = multi_model.predict(X_test)
    multi_metrics = {
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_multi_test_pred)),
        'test_r2': r2_score(y_test, y_multi_test_pred)
    }

    # ---------------------- 模型3：Ridge正则化回归（优化过拟合） ----------------------
    ridge_model = Ridge(alpha=10.0)  # alpha=正则化强度（经调优）
    ridge_model.fit(X_train, y_train)
    y_ridge_test_pred = ridge_model.predict(X_test)
    ridge_metrics = {
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_ridge_test_pred)),
        'test_r2': r2_score(y_test, y_ridge_test_pred)
    }

    # 打印模型评估结果
    print(f"\n" + "="*60)
    print("【模型评估结果（测试集）】")
    print("="*60)
    print(f"1. 简单线性回归（BMI）：")
    print(f"   - RMSE：{simple_metrics['test_rmse']:.2f} | R²：{simple_metrics['test_r2']:.4f}")
    print(f"   - 回归方程：{simple_metrics['equation']}")
    print(f"2. 多元线性回归（全10特征）：")
    print(f"   - RMSE：{multi_metrics['test_rmse']:.2f} | R²：{multi_metrics['test_r2']:.4f}")
    print(f"3. Ridge正则化回归：")
    print(f"   - RMSE：{ridge_metrics['test_rmse']:.2f} | R²：{ridge_metrics['test_r2']:.4f}")

    return {
        # 数据
        'X_simple_train': X_simple_train, 'X_simple_test': X_simple_test,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        # 预测结果
        'y_simple_test_pred': y_simple_test_pred, 'y_multi_test_pred': y_multi_test_pred,
        # 模型与指标
        'simple_model': simple_model, 'multi_model': multi_model, 'ridge_model': ridge_model,
        'simple_metrics': simple_metrics, 'multi_metrics': multi_metrics, 'ridge_metrics': ridge_metrics
    }


# 4. 模型可视化（4类核心图表，修复：seaborn kdeplot参数）
def visualize_models(results, df, best_feat):
    """生成4合一可视化图表，展示模型效果"""
    # 提取关键数据
    X_simple_train = results['X_simple_train']
    y_train = results['y_train']
    y_test = results['y_test']
    y_simple_test_pred = results['y_simple_test_pred']
    y_multi_test_pred = results['y_multi_test_pred']
    simple_model = results['simple_model']
    multi_model = results['multi_model']
    simple_metrics = results['simple_metrics']
    multi_metrics = results['multi_metrics']

    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('开源糖尿病数据集线性回归模型可视化', fontsize=12, fontweight='bold')

    # ---------------------- 图1：简单回归 - 训练集拟合线 ----------------------
    ax1 = axes[0, 0]
    ax1.scatter(X_simple_train, y_train, color='#1f77b4', alpha=0.6, s=30, label='训练集样本')
    # 拟合线（覆盖特征取值范围）
    x_line = np.linspace(X_simple_train.min(), X_simple_train.max(), 100).reshape(-1, 1)
    y_line = simple_model.predict(x_line)
    ax1.plot(x_line, y_line, color="#255e17", linewidth=2.5, label=f'拟合线（R²={simple_metrics["test_r2"]:.4f}）')
    ax1.set_xlabel(f'特征：{best_feat}（标准化后）', fontsize=10)
    ax1.set_ylabel('血糖水平', fontsize=10)
    ax1.set_title(f'简单线性回归：BMI与血糖的拟合效果', fontsize=10, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------------------- 图2：简单回归 - 测试集预测vs实际 ----------------------
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_simple_test_pred, color="#082777", alpha=0.6, s=30, label='测试集样本')
    # 理想预测线（y=x）
    min_val, max_val = min(y_test.min(), y_simple_test_pred.min()), max(y_test.max(), y_simple_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], color="#14933a", linestyle='--', linewidth=2.5, label='理想预测线（y=x）')
    ax2.set_xlabel('实际血糖水平', fontsize=10)
    ax2.set_ylabel('预测血糖水平', fontsize=10)
    ax2.set_title(f'简单回归：测试集预测 vs 实际（RMSE={simple_metrics["test_rmse"]:.2f}）', fontsize=10, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ---------------------- 图3：多元回归 - 残差分布（修复：kdeplot添加data参数） ----------------------
    ax3 = axes[1, 0]
    residuals = y_test - y_multi_test_pred  # 残差 = 实际值 - 预测值
    # 绘制残差直方图+核密度曲线（修复：sns.kdeplot(data=residuals)）
    ax3.hist(residuals, bins=15, color="#434ccc", alpha=0.6, density=True, label='残差分布')
    sns.kdeplot(data=residuals, color="#1b0255", linewidth=2.5, ax=ax3, label='核密度曲线')
    ax3.axvline(x=0, color="#0f93cc", linestyle='--', linewidth=2.5, label='残差=0（理想值）')
    ax3.set_xlabel('残差（实际血糖 - 预测血糖）', fontsize=10)
    ax3.set_ylabel('密度', fontsize=10)
    ax3.set_title(f'多元回归：残差分布（RMSE={multi_metrics["test_rmse"]:.2f}）', fontsize=10, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # ---------------------- 图4：多元回归 - 特征重要性 ----------------------
    ax4 = axes[1, 1]
    coef_abs = np.abs(multi_model.coef_)  # 特征重要性=系数绝对值
    feat_names = df.drop(columns=['血糖水平']).columns
    # 按重要性降序排列
    sorted_idx = np.argsort(coef_abs)[::-1]
    sorted_feats = [feat_names[i] for i in sorted_idx]
    sorted_coef = coef_abs[sorted_idx]
    # 绘制水平条形图
    bars = ax4.barh(sorted_feats, sorted_coef, color="#4568b3", alpha=0.7)
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', ha='left', va='center', fontsize=10)
    ax4.set_xlabel('系数绝对值', fontsize=10)
    ax4.set_title('多元回归：各特征对血糖的影响程度', fontsize=10, fontweight='bold')
    ax4.grid(alpha=0.3, axis='x')

    # 调整间距：先自动布局，再手动扩大垂直间距（解决图注重叠）
    plt.tight_layout()  # 自动优化基本布局
    plt.subplots_adjust(hspace=0.3)  # 关键：扩大子图间垂直间距（值越大间距越宽）
    # 保存并显示图片
    plt.savefig('diabetes_regression_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可视化图表已保存为：diabetes_regression_visualization.png")


# 5. 主函数（整合所有步骤，修复：joblib提前导入）
def main():
    # 步骤1：加载并探索开源糖尿病数据集
    X, X_simple, y, df, best_feat = load_explore_diabetes()
    # 步骤2：训练模型并评估
    results = train_models(X, X_simple, y)
    # 步骤3：可视化模型结果
    visualize_models(results, df, best_feat)
    # 保存最佳模型
    joblib.dump(results['ridge_model'], 'best_diabetes_ridge_model.pkl')
    print("最佳Ridge模型已保存为：best_diabetes_ridge_model.pkl")


# 运行代码
if __name__ == "__main__":
    main()