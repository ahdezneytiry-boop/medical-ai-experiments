import csv
import os
import time
from collections import Counter
import jieba.analyse
from pyecharts import options as opts
from pyecharts.charts import WordCloud
import pandas as pd
# 用于浏览器自动化截图（HTML转图片）
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# -------------------------- 1. 读取数据分析师岗位数据 --------------------------
def read_job_data():
    csv_path = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\招聘数据_合并预处理后.csv"
    df = pd.read_csv(csv_path, encoding="gbk")
    
    all_words = []
    for seg_list in df["岗位名称_seg"].apply(eval):
        if seg_list:
            all_words.extend(seg_list)
    
    # 词频≥3，过滤低频词
    word_freq = Counter(all_words)
    return {word: count for word, count in word_freq.items() if count >= 3}

# -------------------------- 2. 精准停用词过滤 --------------------------
def filter_stop_words(word_freq):
    stop_words = {
        # 地域词
        "徐州", "上海", "深圳", "北京", "广州", "成都", "杭州", "南京",
        # 福利词
        "双休", "五险", "一金", "社保", "公积金", "福利", "年终奖", "带薪假",
        # 无意义词
        "的", "一般", "普通", "相关", "负责", "主要", "以上", "以下", "左右"
    }
    
    filtered_freq = {
        word: count for word, count in word_freq.items() 
        if word not in stop_words and len(word) >= 2
    }
    print(f"✅ 过滤后保留核心词数量：{len(filtered_freq)}个")
    return filtered_freq

# -------------------------- 3. 生成无旋转文字的词云（输出为图片） --------------------------
def generate_job_wordcloud(filtered_freq):
    # 取前80个核心词（按词频排序）
    word_items = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:80]
    
    # 1. 配置词云（核心修改：添加rotation_range=0 实现文字不旋转）
    c = (
        WordCloud(
            init_opts=opts.InitOpts(width="1000px", height="1000px", bg_color="white")
        )
        .add(
            series_name="数据分析师岗位核心词",
            data_pair=word_items,
            word_size_range=[15, 120],  # 字体大小范围
            shape="circle",  # 圆形词云
            textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑"),  # 中文显示
            rotation_range=0,  # 关键配置：旋转角度范围设为0，所有文字水平不旋转
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="数据分析师岗位词云（无旋转图片版）",
                title_textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="微软雅黑")
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
    )
    
    # 2. 先生成临时HTML文件（用于浏览器渲染）
    html_path = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\临时词云文件.html"
    img_path = "C:\\Users\\ahdezneytiry-boop1\\Desktop\\数据分析师无旋转词云.png"  # 最终图片保存路径
    c.render(html_path)
    print(f"✅ 已生成临时HTML文件：{html_path}")
    
    # 3. 配置Chrome浏览器，静默渲染并截图
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # 无头模式（不显示浏览器窗口）
    chrome_options.add_argument("--window-size=1000,1000")  # 窗口大小匹配词云大小
    chrome_options.add_argument("--disable-gpu")  # 禁用GPU加速，避免兼容性问题
    
    # 启动浏览器，加载HTML并截图
    with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) as driver:
        try:
            # 修复：先单独处理路径，避免f-string内直接使用反斜杠
            standard_html_path = os.path.abspath(html_path).replace('\\', '/')
            driver.get(f"file:///{standard_html_path}")
            
            # 等待词云渲染完成（ECharts渲染需要一定时间，可根据电脑配置调整）
            time.sleep(3)  
            # 截图并保存为图片
            driver.save_screenshot(img_path)
            print(f"✅ 词云图片已保存：{img_path}（直接打开即可查看）")
        finally:
            # 4. 清理临时HTML文件（可选，如需保留可注释此行）
            if os.path.exists(html_path):
                os.remove(html_path)
                print(f"✅ 已清理临时HTML文件：{html_path}")

# -------------------------- 执行流程 --------------------------
if __name__ == "__main__":
    job_word_freq = read_job_data()
    filtered_word_freq = filter_stop_words(job_word_freq)
    generate_job_wordcloud(filtered_word_freq)