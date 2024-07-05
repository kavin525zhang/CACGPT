from rank_bm25 import BM25Okapi
import jieba
import numpy as np

# BM25初始化（构建搜索语料库）
corpus = [
    "上半年哪些券商的董事长发生了变更，以及头部券商高管变动的主要原因是什么？",
    "海力风电目前所处行业市场竞争格局？",
    "国家主席习近平在对哈萨克斯坦进行国事访问之际，在当地媒体发表的署名文章标题是什么？",
    "宁德时代利润情况？",
    "盟升电子生产模式是怎样，有没有什么制约产能的主要因素？",
    "达嘉维康的战略布局？"
]

tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

#文档搜索得分（新词条搜索）
query = "宁德时代的战略布局"
tokenized_query = list(jieba.cut(query))
doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)
# [0.33168176 0.0  0.2176771  3.5478576  0.25386726 3.92819009]

# 文档最高得分n个句子，参数n可以调选择top几数据
print(bm25.get_top_n(tokenized_query, corpus, n=1))
# ['达嘉维康的战略布局？']  
# 这就是Bm25一个比较大的弊端，人眼看肯定宁德时代这个词更重要。

# 文档最高得分n个句子的id，参数n可以调选择top几数据
top_n = np.argsort(doc_scores)[::-1][:3]
print(top_n)
# [5 3 0]