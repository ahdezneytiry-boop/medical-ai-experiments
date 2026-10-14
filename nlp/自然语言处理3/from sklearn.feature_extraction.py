import jieba
import numpy as np
from docx import Document
import re
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽tensorflow无关日志

def read_and_clean_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            para_text = para.text.strip()
            if para_text:
                full_text.append(para_text)
        raw_text = ' '.join(full_text)
        clean_text = re.sub(r'[\n\t\r]', ' ', raw_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        return clean_text
    except Exception as e:
        print(f"Read {file_path} error: {str(e)}")
        return ""

def segment_with_filter(text):
    jieba.add_word("智能医学工程")
    jieba.add_word("医药产品经理")
    jieba.add_word("数据分析师")
    jieba.add_word("数据挖掘")
    jieba.add_word("机器学习")
    stop_words = set(["的", "和", "是", "在", "对", "与", "为", "等", "及", "或", "了", "着", "过",
                      "基于", "使用", "采用", "显示", "包括", "涵盖", "提供", "关注", "进行", "分析",
                      "研究", "结果", "需求", "岗位", "数据", "技能", "学历", "经验", "福利", "薪资",
                      "摘要", "关键词", "引言", "参考文献", "致谢", "总结", "展望", "一", "二", "三", "四",
                      "图", "表", "如图", "如表", "所示", "来源", "作者", "单位", "年份", "卷", "期", "页码"])
    words = jieba.lcut(text, cut_all=False)
    filtered_words = [word.strip() for word in words if word.strip() not in stop_words 
                      and len(word.strip()) > 1 and not word.strip().isdigit()
                      and not re.match(r'^[^\u4e00-\u9fa5a-zA-Z]+$', word.strip())]
    return filtered_words

# 读取文档
doc1_path = "data/基于网络爬虫的智能医学工程专业岗位的社会需求分析.docx"
doc2_path = "data/对招聘网站中数据分析师数据的爬取和分析.docx"
doc1_text = read_and_clean_docx(doc1_path)
doc2_text = read_and_clean_docx(doc2_path)
doc1_words = segment_with_filter(doc1_text)
doc2_words = segment_with_filter(doc2_text)

# 1. Jaccard相似度（无外部依赖）
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return round(intersection / union if union != 0 else 0.0, 4)
doc1_set = set(doc1_words)
doc2_set = set(doc2_words)
jaccard_sim = jaccard_similarity(doc1_set, doc2_set)

# 2. Word2Vec相似度（自动训练小型模型，无需外部文件）
def train_small_word2vec(corpus, vec_dim=100):
    model = Word2Vec(sentences=corpus, vector_size=vec_dim, window=3, min_count=1, workers=2)
    return model
def get_w2v_vec(words, model, vec_dim=100):
    vecs = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(vec_dim)
# 用两篇文档的分词结果构建语料库
w2v_corpus = [doc1_words, doc2_words]
w2v_model = train_small_word2vec(w2v_corpus)
doc1_w2v_vec = get_w2v_vec(doc1_words, w2v_model)
doc2_w2v_vec = get_w2v_vec(doc2_words, w2v_model)

# 3. 余弦相似度计算（通用）
def cosine_sim(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return round(np.dot(vec1, vec2) / (norm1 * norm2), 4)
word2vec_sim = cosine_sim(doc1_w2v_vec, doc2_w2v_vec)

# 4. BERT相似度（适配国内网络，失败则提示）
def get_bert_sim(text1, text2):
    try:
        # 国内镜像源（避免超时）
        tokenizer = BertTokenizer.from_pretrained("https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/bert-base-chinese/")
        model = BertModel.from_pretrained("https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/bert-base-chinese/")
        model.eval()
        def get_vec(text):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(** inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        vec1 = get_vec(text1)
        vec2 = get_vec(text2)
        return cosine_sim(vec1, vec2)
    except Exception as e:
        print(f"\nBERT模型加载失败（网络/路径问题）：{str(e)[:100]}...")
        return "加载失败"

bert_sim = get_bert_sim(doc1_text, doc2_text)

# 输出结果
print("="*60)
print("【多方法文本相似度计算结果】")
print(f"1. Jaccard相似度（关键词交集）：{jaccard_sim}")
print(f"2. Word2Vec相似度（自动训练模型）：{word2vec_sim}")
print(f"3. BERT相似度（动态语义）：{bert_sim}")
print("\n注：若BERT加载失败，可手动下载模型到本地后修改路径")
print("="*60)