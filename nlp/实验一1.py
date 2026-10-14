# 导入pandas库
import pandas as pd

# 使用pandas的DataFrame函数创建一个数据集
df = pd.DataFrame(
    {
        # 定义表格的列名和对应数据，包含4列：名称、数量、金额、合计
        '名称': ['产品1', '产品2', '产品3', '产品4', '产品5', '产品6', '产品7', '产品8'],
        '数量': ['A', '0.7', '0.8', '0.4', '0.7', 'B', '0.76', '0.28'],
        '金额': ['0', '0.48', '0.33', 'C', '0.74', '0', '0', '0.22'],
        '合计': ['D', '0.37', '0.28', 'E', '0.57', 'F', '0', '0.06'],
    }
)

# 打印原始的DataFrame
print("df:\n{}\n".format(df))

# 使用replace方法将数据中的'A'替换为0.1，结果保存到新的变量df1

# 打印替换后的df1，查看替换效果
print("df1:\n{}\n".format(df))

# 对DataFrame的'名称'列使用str.replace方法，只替换该列中字符串的部分内容
df2 = df['名称'].str.replace('产品', 'product')

# 打印替换后的名称列数据df2
print("df2:\n{}\n".format(df2))

# 对DataFrame的'合计'列使用replace方法，通过字典指定替换规则
df['合计'].replace({'D': 0.11111, 'F': 0.22222}, inplace=True)

# 打印修改后的原DataFrame df，查看'合计'列的替换效果
print("df:\n{}\n".format(df))