from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    "我 爱 学习",    
    "学习 使 我 快乐", 
    "我 热爱 编程"  
]

vectorizer = CountVectorizer(token_pattern='\w+')
X = vectorizer.fit_transform(corpus)

print("1. 词汇表索引：", vectorizer.vocabulary_)
print("2. 词频矩阵：")
print(X.toarray()) 