import os
import json
import numpy as np
from tqdm import tqdm
from typing import List


class VectorStore:
    def __init__(self, document: List[str] = None) -> None:
        """
        初始化向量数据库
        """
        self.document = document if document is not None else []
        self.vectors = []

    def get_vector(self, EmbeddingModel) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc='Calculating embeddings'):
            # 确保传入正确的参数名或默认值已在基类处理
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "document.json"), 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        with open(os.path.join(path, "vectors.json"), 'w', encoding='utf-8') as f:
            json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        doc_path = os.path.join(path, "document.json")
        vec_path = os.path.join(path, "vectors.json")

        if os.path.exists(vec_path) and os.path.exists(doc_path):
            with open(vec_path, 'r', encoding='utf-8') as f:
                self.vectors = json.load(f)
            with open(doc_path, 'r', encoding='utf-8') as f:
                self.document = json.load(f)
        else:
            print(f"警告：在 {path} 未找到完整的数据库文件。")

    def query(self, query: str, EmbeddingModel, k: int = 1) -> List[str]:
        if not self.vectors:
            return []

        query_vector = EmbeddingModel.get_embedding(query)
        # 简单的向量化相似度计算
        vec_matrix = np.array(self.vectors)
        q_vec = np.array(query_vector)

        # 计算相似度并排序
        scores = np.dot(vec_matrix, q_vec) / (np.linalg.norm(vec_matrix, axis=1) * np.linalg.norm(q_vec) + 1e-9)
        best_indices = np.argsort(scores)[-k:][::-1]

        return [self.document[i] for i in best_indices]
