# 导入数据处理库pandas，用于数据框的创建、操作和分析
import pandas as pd
# 导入matplotlib的pyplot模块，用于基础绘图功能
import matplotlib.pyplot as plt
# 导入seaborn库，基于matplotlib的高级可视化库，提供更美观的图表样式
import seaborn as sns
# 导入missingno库，专门用于缺失值的可视化分析
import missingno as msno
# 从sklearn.datasets中导入load_diabetes函数，用于加载糖尿病数据集
from sklearn.datasets import load_diabetes
# 导入scipy的stats模块，用于进行统计分析（如Z-score计算）
from scipy import stats
# 重复导入matplotlib.pyplot（可删除，此处保留原代码结构，注释说明冗余）
import matplotlib.pyplot as plt

# 中文显示设置：解决matplotlib绘图时中文乱码问题
# 设置支持的中文字体列表（兼容不同系统，优先SimHei，其次是开源中文字体）
plt.rcParams["font.family"] = ["SimHei"]
# 解决坐标轴负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 1. 加载并探索数据集 ----------------------
# 调用load_diabetes()函数加载糖尿病数据集，返回的是Bunch对象（类似字典）
diabetes = load_diabetes()
# 将数据集中的特征数据（diabetes.data）转换为DataFrame，列名使用数据集自带的特征名（diabetes.feature_names）
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# 向DataFrame中添加目标变量列（diabetes.target，代表糖尿病进展指标），列名为'target'
df['target'] = diabetes.target

# 打印数据集基本信息的标题，用于区分输出内容
print('=== 数据集基本信息 ===')
# 调用df.info()，查看数据集的行数、列数、各列数据类型及非空值数量
df.info()
# 打印数据集形状（行数，列数），直观展示数据规模
print(f'\n数据集形状：{df.shape}')
# 打印描述性统计的标题
print('=== 各列描述性统计 ===')
# 调用df.describe()计算各列的描述性统计量（均值、标准差、四分位数等），保留2位小数
print(df.describe().round(2))

# ---------------------- 2. 缺失值分析（含可视化） ----------------------
# 打印缺失值统计的标题
print('\n=== 缺失值统计 ===')
# 调用df.isnull().sum()，统计每列的缺失值数量（isnull()标记缺失值为True，sum()计算True的数量）
missing = df.isnull().sum()
# 打印各列缺失值统计结果
print(missing)

# （1）绘制missingno矩阵图，可视化缺失值分布
# 创建画布，设置画布大小为12英寸（宽）×8英寸（高）
plt.figure(figsize=(12, 8))
# 调用msno.matrix()绘制缺失值矩阵图，颜色设置为蓝灰色（RGB值），字体大小10
msno.matrix(df, color=(0.2, 0.4, 0.6), fontsize=10)
# 设置图表标题，字体大小14，标题与图表顶部间距20
plt.title('缺失数据矩阵图', fontsize=14, pad=20)
# 添加背景网格，线型为虚线（--），透明度0.5（避免遮挡数据）
plt.grid(linestyle='--', alpha=0.5)
# 旋转x轴刻度标签45度，避免标签重叠
plt.xticks(rotation=45)
# 自动调整子图间距，避免元素重叠
plt.tight_layout()
# 显示绘制好的图表
plt.show()

# （2）绘制missingno柱状图，展示各列非缺失值占比
# 创建新画布，大小12×8英寸
plt.figure(figsize=(12, 8))
# 调用msno.bar()绘制缺失值柱状图，颜色设置为深一点的蓝灰色
msno.bar(df, color=(0.3, 0.5, 0.7))
# 设置图表标题，字体大小14，顶部间距20
plt.title('缺失数据柱状图', fontsize=14, pad=20)
# 添加背景网格，仅显示y轴方向，线型虚线，透明度0.5
plt.grid(linestyle='--', alpha=0.5, axis='y')
# 旋转x轴刻度标签45度，避免重叠
plt.xticks(rotation=45)
# 自动调整布局
plt.tight_layout()
# 显示图表
plt.show()

# ---------------------- 3. 异常值/离群值分析（多方法+可视化） ----------------------
# （1）绘制箱线图，可视化各特征的离群值
# 创建画布，大小14×10英寸（水平箱线图需要更宽的画布）
plt.figure(figsize=(14, 10))
# 调用sns.boxplot()绘制水平箱线图（orient='h'），使用viridis配色方案（渐变且易区分）
sns.boxplot(data=df, orient='h', palette='viridis')
# 设置图表标题，字体大小16，顶部间距20
plt.title('各特征箱线图（离群值可视化）', fontsize=16, pad=20)
# 设置x轴标签（特征值），字体大小12
plt.xlabel('特征值', fontsize=12)
# 设置y轴标签（特征名称），字体大小12
plt.ylabel('特征名称', fontsize=12)
# 添加背景网格，线型虚线，透明度0.7
plt.grid(linestyle='--', alpha=0.7)
# 自动调整布局
plt.tight_layout()
# 显示图表
plt.show()

# （2）定义函数，分别用IQR法和Z-score法量化离群值
# 定义IQR法离群值检测函数，参数为单个特征数据
def find_outliers_iqr(data):
    # 计算数据的下四分位数（Q1，25%分位数）
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    # 计算四分位距（IQR = Q3 - Q1）
    iqr = q3 - q1
    # 返回超出[Q1-1.5*IQR, Q3+1.5*IQR]范围的数据（即离群值）
    return data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]

# 定义Z-score法离群值检测函数，参数为特征数据和阈值（默认3，超出3个标准差为离群值）
def find_outliers_zscore(data, threshold=3):
    # 计算数据的Z-score（(数据值-均值)/标准差）
    z_scores = stats.zscore(data)
    # 返回Z-score绝对值大于阈值的数据（离群值）
    return data[abs(z_scores) > threshold]

# 创建空列表，用于存储各特征的离群值统计结果
outliers_stats = []
# 遍历DataFrame的每一列（每个特征+目标变量）
for col in df.columns:
    # 用IQR法计算当前列的离群值
    iqr_out = find_outliers_iqr(df[col])
    # 用Z-score法计算当前列的离群值
    z_out = find_outliers_zscore(df[col])
    # 将当前特征名、IQR离群值数量、Z-score离群值数量组成字典，添加到列表中
    outliers_stats.append({
        '特征': col,
        'IQR离群值数量': len(iqr_out),
        'Z-score离群值数量': len(z_out)
    })
# 将列表转换为DataFrame，便于结构化展示
outliers_stats = pd.DataFrame(outliers_stats)

# 打印离群值统计结果的标题
print('\n=== 离群值统计（IQR vs Z-score） ===')
# 打印离群值统计DataFrame
print(outliers_stats)

# ---------------------- 4. 特征相关性分析（热力图） ----------------------
# 创建画布，大小12×10英寸（热力图需要足够空间显示数值）
plt.figure(figsize=(12, 10))
# 计算DataFrame中各列之间的皮尔逊相关系数（默认）
corr = df.corr()
# 绘制相关性热力图：annot=True显示相关系数数值，cmap='coolwarm'用红蓝渐变表示正负相关，
# fmt='.2f'保留2位小数，linewidths=0.5设置格子间距
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# 设置图表标题，字体大小16，顶部间距20
plt.title('特征相关性热力图', fontsize=16, pad=20)
# 自动调整布局
plt.tight_layout()
# 显示图表
plt.show()

# ---------------------- 5. 特征与目标变量关系（散点图） ----------------------
# 创建画布，大小16×12英寸（用于容纳多个子图）
plt.figure(figsize=(16, 12))
# 遍历糖尿病数据集的特征名，enumerate获取索引（i）和特征名（col），索引从1开始（便于子图定位）
for i, col in enumerate(diabetes.feature_names, 1):
    # 创建子图：3行4列的网格，当前子图位置为i（按行优先排列）
    plt.subplot(3, 4, i)
    # 绘制散点图：x轴为当前特征，y轴为目标变量（糖尿病进展），alpha=0.6设置点的透明度（避免重叠遮挡）
    sns.scatterplot(x=df[col], y=df['target'], alpha=0.6)
    # 设置子图标题，显示当前特征与目标变量的关系
    plt.title(f'{col}与糖尿病进展的关系')
    # 添加子图背景网格，线型虚线，透明度0.5
    plt.grid(linestyle='--', alpha=0.5)
# 自动调整所有子图的间距，避免标题、标签重叠
plt.tight_layout()
# 显示整个包含多个子图的画布
plt.show()