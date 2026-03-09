import os
import re
import sys
import numpy as np
import time
import random
from Qwen.Agent.src.tools import HybridRetriever
from RAG.Embeddings import OpenAIEmbedding
from RAG.VectorBase import VectorStore
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def generate_test_cases_improved(vector_base, num_cases=50):
    new_test_data = []
    for _ in range(num_cases):
        idx = random.randint(0, len(vector_base.document) - 1)
        full_text = vector_base.document[idx]
        body_text = full_text.split("] ")[-1]

        # 尝试取第二句或者中间的一段，避开开头的“第一条”、“、”等
        sentences = re.split(r'[。！？](?!\d)', body_text)
        # 挑一个长度适中（15-30字）的句子作为 Query
        valid_sentences = [s.strip() for s in sentences if 15 < len(s.strip()) < 50]

        if valid_sentences:
            query = random.choice(valid_sentences)
            # 再次清洗干扰符
            query = re.sub(r'^[、\d\s]+', '', query)
            new_test_data.append({"query": query, "expected_doc_idx": idx})
    return new_test_data


def evaluate_retrieval(retriever, embedder, vector_base, test_cases):
    methods = ["Vector Only", "Hybrid (RRF)"]
    results = {method: {"hits": 0, "total": 0, "time": 0} for method in methods}

    # --- 记录特定案例的容器 ---
    failed_both = []  # 两者都匹配失败
    hybrid_only_hits = []  # 只有 Hybrid 成功

    print(f"开始评估 {len(test_cases)} 个案例...")

    for i, case in enumerate(tqdm(test_cases, desc="Evaluating")):
        query = case["query"]
        expected_idx = case["expected_doc_idx"]
        expected_content = vector_base.document[expected_idx]

        try:
            # 1. 测试纯向量检索
            start = time.time()
            query_vector = embedder.get_embedding(query)
            # 计算余弦相似度并取 Top 3
            sims = [OpenAIEmbedding.cosine_similarity(query_vector, v) for v in vector_base.vectors]
            vector_top_3 = np.argsort(sims)[-3:][::-1].tolist()

            v_hit = expected_idx in vector_top_3
            results["Vector Only"]["hits"] += 1 if v_hit else 0
            results["Vector Only"]["total"] += 1
            results["Vector Only"]["time"] += (time.time() - start)

            # 2. 测试混合检索
            start = time.time()
            hybrid_top_3, _ = retriever.search(query, embedder, vector_base, top_k=3)

            h_hit = expected_idx in hybrid_top_3
            results["Hybrid (RRF)"]["hits"] += 1 if h_hit else 0
            results["Hybrid (RRF)"]["total"] += 1
            results["Hybrid (RRF)"]["time"] += (time.time() - start)

            # --- 3. 核心逻辑：记录失败详情 ---
            case_info = {
                "query": query,
                "expected_content": expected_content,
                "vector_top_3": [vector_base.document[idx][:50] + "..." for idx in vector_top_3],
                "hybrid_top_3": [vector_base.document[idx][:50] + "..." for idx in hybrid_top_3]
            }

            if not v_hit and not h_hit:
                failed_both.append(case_info)
            elif h_hit and not v_hit:
                hybrid_only_hits.append(case_info)

        except Exception as e:
            print(f"\n第 {i} 个案例出错: {e}")
            continue

    # --- 4. 输出报表 (恢复并优化了百分比展示) ---
    print("\n" + "=" * 20 + " 评估结果报表 " + "=" * 20)
    print(f"{'Method':<15} | {'Hit Rate@3':<12} | {'Avg Latency':<12}")
    print("-" * 50)
    for method, data in results.items():
        total = data["total"] if data["total"] > 0 else 1
        hit_rate = (data["hits"] / total) * 100
        avg_time = (data["time"] / total) * 1000  # 转为毫秒
        print(f"{method:<15} | {hit_rate:>10.2f}% | {avg_time:>10.2f}ms")

    # --- 5. 详细案例分析 ---
    print("\n" + "!" * 20 + " 深度案例分析 " + "!" * 20)

    if hybrid_only_hits:
        print(f"\n[1] Hybrid 独有的成功案例 (共{len(hybrid_only_hits)}个，展示前3个):")
        for item in hybrid_only_hits[:3]:
            print(f"-> 问题: {item['query']}")
            print(f"   正确答案首部: {item['expected_content'][:80]}...")
            print("-" * 20)

    if failed_both:
        print(f"\n[2] 两者全部失败的顽固案例 (共{len(failed_both)}个，展示前3个):")
        for item in failed_both[:3]:
            print(f"-> 问题: {item['query']}")
            print(f"   Vector 第一候选: {item['vector_top_3'][0]}")
            print(f"   Hybrid 第一候选: {item['hybrid_top_3'][0]}")
            print("-" * 20)

    sys.stdout.flush()


def start_evaluation():
    # 1. 初始化
    _embedder = OpenAIEmbedding(
        path="BAAI/bge-m3",
        is_api=True
    )
    _vector_base = VectorStore()
    _vector_base.load_vector("./storage")

    if not _vector_base.document:
        print("错误：未加载到向量库内容，请检查 ./storage 路径。")
        return

    # 2. 初始化检索器
    retriever = HybridRetriever(_vector_base.document)

    # 3. 核心选择：
    test_data = generate_test_cases_improved(_vector_base, num_cases=50)

    # 4. 运行评估
    evaluate_retrieval(retriever, _embedder, _vector_base, test_data)


if __name__ == "__main__":
    start_evaluation()
