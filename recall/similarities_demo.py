from similarities import BM25Similarity, EnsembleSimilarity
import numpy as np
import requests
import json
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

model = BM25Similarity()
corpus = [
    "上半年哪些券商的董事长发生了变更，以及头部券商高管变动的主要原因是什么？",
    "海力风电目前所处行业市场竞争格局？",
    "国家主席习近平在对哈萨克斯坦进行国事访问之际，在当地媒体发表的署名文章标题是什么？",
    "宁德时代利润情况？",
    "盟升电子生产模式是怎样，有没有什么制约产能的主要因素？",
    "达嘉维康的战略布局？"
]

model.add_corpus(corpus)
res = model.most_similar(["宁德时代的战略布局？"])[0]

# 分数归一化
res = dict(sorted(res.items(), key=lambda x: x[0]))
scores = list(res.values())
min_bm25, max_bm25 = min(scores), max(scores)
scores = [score - min_bm25 / (max_bm25 - min_bm25 + 0.01)  for score in scores]
bm25_norm_score = [sigmoid(score) for score in scores]
print("norm scores:{}".format(bm25_norm_score))

def get_text_embedding(text_or_text_list: str | List[str],
                       model: str = "bge") -> List[List[float]] | None:
    try:
        res = requests.post(
            "http://172.17.120.200:8003/embedding",
            json={"text": text_or_text_list, "type": model},
            timeout=60
        )
        return res.json()
    except (requests.Timeout, json.JSONDecodeError):
        return None
    
emb_scores = cosine_similarity(get_text_embedding(["宁德时代的战略布局？"]), get_text_embedding(corpus))[0]
print("emb scores:{}".format(emb_scores))
final_scores = [0.5 * bm25_norm_score[i] + 0.5 * emb_scores[i] for i in range(len(bm25_norm_score))]
print("final scores:{}".format(final_scores))

