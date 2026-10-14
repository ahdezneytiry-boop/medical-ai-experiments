import pandas as pd
import re

# -------------------------- 修复核心：薪资抽取函数（新增异常处理） --------------------------
def extract_salary(salary_text):
    # 1. 先处理异常文本（空值、纯文本、重复格式）
    salary_text = str(salary_text).strip()
    # 处理空值或无数字的纯文本
    if not salary_text or not any(char.isdigit() for char in salary_text):
        return None, None, None
    # 修复重复格式（如"1.51.5"→"1.5"，"9-9k"→"9k"）
    # 去除连续重复的数字+小数点（如"1.51.5"→"1.5"）
    salary_text = re.sub(r'(\d+\.\d+)\1+', r'\1', salary_text)
    # 去除重复的范围符号（如"9-9k"→"9k"）
    salary_text = re.sub(r'(\d+)-\1', r'\1', salary_text)

    # 2. 正则匹配（分3种常见格式，按优先级匹配）
    try:
        # 格式1：9-15k、10-20k·15薪（最常见）
        k_match = re.search(r'(\d+\.?\d*)-?(\d+\.?\d*)?k', salary_text)
        if k_match:
            min_sal = float(k_match.group(1))
            # 处理无范围的情况（如"12k"→最低最高都是12）
            max_sal = float(k_match.group(2)) if k_match.group(2) else min_sal
            return round(min_sal, 1), round(max_sal, 1), 'k/月'
        
        # 格式2：1-2万、1.5-2.5万/月（转为k单位，1万=10k）
        month_match = re.search(r'(\d+\.?\d*)-?(\d+\.?\d*)?万(?!/年)', salary_text)
        if month_match:
            min_sal = float(month_match.group(1)) * 10
            max_sal = float(month_match.group(2)) * 10 if month_match.group(2) else min_sal
            return round(min_sal, 1), round(max_sal, 1), 'k/月'
        
        # 格式3：12-24万/年（转为月薪资，年薪资÷12）
        year_match = re.search(r'(\d+\.?\d*)-?(\d+\.?\d*)?万/年', salary_text)
        if year_match:
            min_sal = (float(year_match.group(1)) * 10) / 12
            max_sal = (float(year_match.group(2)) * 10) / 12 if year_match.group(2) else min_sal
            return round(min_sal, 1), round(max_sal, 1), 'k/月'
        
        # 格式4：单值薪资（如"15k"、"2万"）
        single_match = re.search(r'(\d+\.?\d*)(k|万)', salary_text)
        if single_match:
            sal = float(single_match.group(1))
            unit = single_match.group(2)
            if unit == 'k':
                return round(sal, 1), round(sal, 1), 'k/月'
            else:  # 万
                sal_k = sal * 10
                return round(sal_k, 1), round(sal_k, 1), 'k/月'
        
        # 无匹配格式
        return None, None, None
    
    # 3. 容错处理：即使某条数据报错，也返回空值，不中断批量处理
    except Exception as e:
        print(f"⚠️  单条薪资处理失败（文本：{salary_text}，错误：{str(e)}），跳过该条")
        return None, None, None


# -------------------------- 加载数据+批量抽取 --------------------------
# 加载含薪资文本的原始数据（确保路径正确）
raw_df = pd.read_csv("data/招聘数据_合并预处理后.csv", encoding='gbk')

# 批量抽取薪资信息（新增进度提示）
print("🔄 开始批量抽取薪资信息...")
salary_result = raw_df['薪资'].apply(extract_salary)

# 添加结构化薪资列到原始数据
raw_df[['最低薪资(k/月)', '最高薪资(k/月)', '薪资单位']] = pd.DataFrame(
    salary_result.tolist(), 
    index=raw_df.index
)

# -------------------------- 结果输出与保存 --------------------------
# 统计抽取成功率
valid_count = raw_df['最低薪资(k/月)'].notna().sum()
total_count = len(raw_df)
success_rate = round(valid_count / total_count * 100, 2)

print(f"\n=== 薪资结构化抽取结果汇总 ===")
print(f"总处理条数：{total_count}条")
print(f"成功抽取条数：{valid_count}条")
print(f"抽取成功率：{success_rate}%")

# 显示前15条结果（含正常+异常处理案例）
print(f"\n前15条抽取结果示例：")
display_cols = ['岗位名称_cleaned', '薪资', '最低薪资(k/月)', '最高薪资(k/月)', '薪资单位']
print(raw_df[display_cols].head(15).to_string(index=False))

# 保存结果到桌面（覆盖原文件或新建，避免丢失）
save_path = "data/招聘数据_薪资结构化后.csv"
raw_df.to_csv(save_path, index=False, encoding='gbk')
print(f"\n💾 已保存结构化结果到：{save_path}")