import os
from typing import List
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:

    def __init__(self, path: str, is_api: bool) -> None:
        """
        初始化嵌入基类
        Args:
            path: 模型或数据的路径
            is_api: 是否使用API方式

        """
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text, model: str = "BAAI/bge-m3") -> List[float]:
        """
        获取文本的嵌入向量表示
        Args:
            text(str): 输入文本
            model(str): 使用的模型名称
        Returns:
            List[float]: 文本的嵌入向量
        Raises:
            NotImplementedError: 该方法需要在子类中实现

        """
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        Args:
            vector1(List[float]): 第一个向量
            vector2(List[float]): 第二个向量
        Returns:
            float: 两个向量的余弦相似度，范围在[-1, 1]之前

        """
        # 将输入列表转换为numpy数组，并指定数据类型为float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # 检查向量中是否包含无穷大或NaN值
        # np.isfinite()计算是否为有限数
        if not np.all(np.isfinite(v1)) and not np.all(np.isfinite(v2)):
            return 0.0

        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        # 计算向量的L2范数
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 计算分母，为两个向量范数的乘积
        magnitude = norm_v1 * norm_v2
        # 分母为0：
        if magnitude == 0:
            return 0.0

        # 返回余弦相似度
        return dot_product / magnitude


class OpenAIEmbedding(BaseEmbeddings):

    def __init__(self, path: str, is_api: bool) -> None:
        super().__init__(path, is_api)
        if self.is_api:

            self.client = OpenAI(timeout=30.0)
            # 从环境变量中获取硅基流动的密钥
            self.client.api_key = os.getenv('OPENAI_API_KEY')
            # 获取硅基流动的基础URL
            self.client.base_url = os.getenv('OPENAI_BASE_URL')

    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:

        if self.is_api:
            # 将文本行中的换行符替换为空格
            text = text.replace("\n", " ")
            # 访问客户端的embeddings属性
            # .create 调用create方法，发送请求创建嵌入
            # .data[0].embedding 返回数据对象中列表的第一个结果的embedding属性，即嵌入向量
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
