# 1. 导入依赖库
import pandas as pd
import numpy as np
import re
import nltk
# nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# 2. 定义语料和标签
corpus = [
    'The sky is blue and beautiful.',
    'Love this blue and beautiful sky!',
    'The quick brown fox jumps over the lazy dog.',
    'The brown fox is quick and the blue dog is lazy!',
    'The sky is very blue and the sky is very beautiful today',
    'The dog is lazy but the brown fox is quick!'
]
labels = ['weather', 'weather', 'animals', 'animals', 'weather', 'animals']


# 3. 构建DataFrame数据
corpus = np.array(corpus)
# 修正原代码中的拼写错误（catoray → category）
corpus_df = pd.DataFrame({'Document': corpus, 'category': labels})


# 4. 分词和停止词去除
# 手动定义英文停用词列表（替代NLTK下载）
stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]
# 建立词分隔模型
cut_model = nltk.WordPunctTokenizer()

# 定义文本标准化函数
def Normalize_corpus(doc):
    # 去除标点符号
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', string=doc)
    # 转为小写
    doc = doc.lower()
    # 去除两端空格
    doc = doc.strip()
    # 分词
    tokens = cut_model.tokenize(doc)
    # 去除停止词
    doc = [token for token in tokens if token not in stopwords]
    # 拼接为字符串（用于后续TF-IDF处理）
    doc = ' '.join(doc)
    return doc


# 5. 向量化处理（批量应用文本标准化函数）
# 避免函数名与变量名冲突，重命名为Normalize_corpus_vec
Normalize_corpus_vec = np.vectorize(Normalize_corpus)
corpus_norm = Normalize_corpus_vec(corpus)


# 6. 构建TF-IDF模型（词袋模型）
Tf = TfidfVectorizer(use_idf=True)
Tf.fit(corpus_norm)
# 获取特征词列表（新版本sklearn可替换为get_feature_names_out()）
vocs = Tf.get_feature_names_out()
# 转换为TF-IDF矩阵
corpus_array = Tf.transform(corpus_norm).toarray()
# 转为DataFrame格式
corpus_norm_df = pd.DataFrame(corpus_array, columns=vocs)


# 7. 计算余弦相似度
similarity_matrix = cosine_similarity(corpus_array)
similarity_matrix_df = pd.DataFrame(similarity_matrix)


# 8. K-Means聚类
model = KMeans(n_clusters=2, n_init='auto', random_state=42)
model.fit(np.array(similarity_matrix))
# 输出聚类标签
print("聚类标签：", model.labels_)
# 将聚类标签加入结果DataFrame
corpus_norm_df['k_labels'] = np.array(model.labels_)
# 输出最终结果
print("\n聚类结果DataFrame：")
print(corpus_norm_df)