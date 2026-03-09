import datetime
import sys
import os
import wikipedia
import requests
from RAG.Embeddings import OpenAIEmbedding
from RAG.VectorBase import VectorStore
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""
定义Agent可以使用的工具函数
"""


class HybridRetriever:
    """混合检索器：集成 BM25 与 向量 RRF 融合算法"""
    def __init__(self, documents: list):
        self.documents = documents
        # 预处理：中文分词
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]

        # 初始化 BM25 实例
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, embedder, vector_base, top_k=3, k_rrf=60):
        query_vector = embedder.get_embedding(query)
        # 计算所有文档的相似度
        vec_similarities = [OpenAIEmbedding.cosine_similarity(query_vector, vec) for vec in vector_base.vectors]
        # 获取前 10 个最相关的索引
        dense_top_indices = np.argsort(vec_similarities)[-10:][::-1]

        # --- 步骤 B: 关键词检索 (Sparse/BM25) ---
        tokenized_query = list(jieba.cut(query))
        # 获取 BM25 分数
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 获取前 10 个最相关的索引
        sparse_top_indices = np.argsort(bm25_scores)[-10:][::-1]

        # --- RRF 融合逻辑 (稍微修改返回值) ---
        all_indices = set(dense_top_indices) | set(sparse_top_indices)
        rrf_scores = {}

        for idx in all_indices:
            score = 0.0
            if idx in dense_top_indices:
                rank = np.where(dense_top_indices == idx)[0][0] + 1
                score += 1.0 / (k_rrf + rank)
            if idx in sparse_top_indices:
                rank = np.where(sparse_top_indices == idx)[0][0] + 1
                score += 1.0 / (k_rrf + rank)
            rrf_scores[idx] = score

        # 排序
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_indices = sorted_indices[:top_k]

        # 返回 top_indices 和 完整的评分字典
        return top_indices, rrf_scores


# 获取当前的日期和时间
def get_current_datetime():

    current_time = datetime.datetime.now()
    formatted_datetime = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


# 在维基百科中搜索指定查询的前三个页面摘要
def search_wikipedia(query: str):

    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"页面：{page_title}\n摘要：{wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "维基百科没有搜索到合适的结果"
    return "\n\n".join(summaries)


# 获取指定经纬度的当前位置温度
def get_current_temperature(latitude: float, longitude: float):

    # Open Meteo API 的URL
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"

    # 请求参数
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': "temperature_2m",
        'forecast_days': 1,

    }

    # 发送API请求
    response = requests.get(open_meteo_url, params=params)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON响应
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    # 获取当前的UTC时间
    current_utc_time = datetime.datetime.now(datetime.UTC)

    # 将时间字符串转换为datetime对象
    # results的结构为：
    # {
    #     "hourly": {
    #         "time": ["2023-10-01T12:00:00", "2023-10-01T13:00:00", ...],
    #         "temperature_2m": [15.2, 16.1, ...],
    #         ...
    #     }
    # }
    # .fromisoformat 将符合ISO 8601格式的字符串解析为datetime对象
    # datetime的replace方法，可替换对象的属性，返回一个新的datetime对象
    # tzinfo将时区信息设置为UTC
    time_list = [datetime.datetime.fromisoformat(time_str).replace(tzinfo=datetime.timezone.utc)
                 for time_str in results['hourly']['time']]

    # 获取温度列表
    temperature_list = results['hourly']['temperature_2m']

    # 找到最接近当前时间的索引
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))

    # 获取当前温度
    current_temperature = temperature_list[closest_time_index]

    # 返回当前温度的字符串格式
    return f'现在温度是{current_temperature}℃'


# 初始化RAG组件（全局单例，避免重复加载）
_embedder = None
_vector_base = None
_retriever = None


def _init_rag():
    global _embedder, _vector_base,  _retriever

    # 只有在变量为 None 时才初始化
    if _embedder is None:
        print(">>> 正在初始化 Embedding...")
        _embedder = OpenAIEmbedding(path="BAAI/bge-m3", is_api=True)

    if _vector_base is None:
        _vector_base = VectorStore()
        _vector_base.load_vector("./storage")

    if _retriever is None:
        _retriever = HybridRetriever(_vector_base.document)


def rag_search(query: str):
    """
    检索知识库。
    """
    _init_rag()
    # 获取 Top-3 最相关分块
    indices, _ = _retriever.search(query, _embedder, _vector_base, top_k=3)
    final_texts = [_vector_base.document[i] for i in indices]
    return "\n\n".join(final_texts)
