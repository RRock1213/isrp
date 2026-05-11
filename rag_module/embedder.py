"""
Embedding 客户端封装

使用阿里云百炼 text-embedding-v4 API
"""

import hashlib
import time
from typing import List, Optional
import json


class Embedder:
    """Embedding 客户端"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimension: int = 1024,
        batch_size: int = 10
    ):
        # 如果没有传入 api_key，从 config.py 获取
        if api_key is None:
            try:
                from config import API_KEYS
                api_key = API_KEYS.get("dashscope", "")
            except ImportError:
                print("[Embedder] Warning: 无法导入 config.API_KEYS，请在初始化时传入 api_key")
                api_key = ""

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

        # 延迟导入 OpenAI 客户端
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client

    def embed(self, text: str) -> List[float]:
        """单个文本 embedding"""
        if not text or not text.strip():
            return [0.0] * self.dimension

        print(f"[Embedder] 调用真实API生成嵌入，文本长度: {len(text)} 字符")
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension,
                encoding_format="float"
            )
            print(f"[Embedder] API调用成功，返回向量维度: {len(response.data[0].embedding)}")
            return response.data[0].embedding
        except Exception as e:
            print(f"[Embedder] Error: {e}")
            return [0.0] * self.dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding"""
        if not texts:
            return []

        results = []
        print(f"[Embedder] 开始批量生成嵌入，总文本数: {len(texts)}, 批次大小: {self.batch_size}")

        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"[Embedder] 处理批次 {i//self.batch_size + 1}/{(len(texts)+self.batch_size-1)//self.batch_size}, 文本数: {len(batch)}")

            # 过滤空文本
            batch = [t if t and t.strip() else " " for t in batch]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension,
                    encoding_format="float"
                )
                print(f"[Embedder] 批次API调用成功，返回 {len(response.data)} 个向量")

                # 按 index 排序
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                results.extend(batch_embeddings)

            except Exception as e:
                print(f"[Embedder] Batch error: {e}")
                # 失败时返回零向量
                results.extend([[0.0] * self.dimension] * len(batch))

        print(f"[Embedder] 批量处理完成，共生成 {len(results)} 个向量")
        return results

    def get_cache_key(self, text: str) -> str:
        """生成缓存 key"""
        content = f"{self.model}:{self.dimension}:{text}"
        return hashlib.md5(content.encode()).hexdigest()


class MockEmbedder(Embedder):
    """
    Mock Embedder（用于测试）

    不调用真实 API，基于文本特征生成确定性向量
    相似文本会生成相似向量
    """

    def __init__(self, dimension: int = 1024):
        super().__init__(api_key="mock", dimension=dimension)
        self._use_mock = True
        self._cache = {}  # 缓存已生成的向量

    def embed(self, text: str) -> List[float]:
        """返回基于文本特征的确定性向量"""
        import random
        import hashlib

        # 使用缓存
        if text in self._cache:
            return self._cache[text]

        # 基于文本 hash 生成确定性随机种子
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        random.seed(seed)

        # 生成基础向量
        base_vector = [random.gauss(0, 1) for _ in range(self.dimension)]

        # 根据文本特征添加偏移，使相似文本有相似向量
        words = text.lower().split()
        word_features = {}

        for word in words:
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:4], 16)
            word_features[word] = word_hash / 65535.0

        # 调整向量：相似词会影响向量方向
        for i, word in enumerate(words[:10]):  # 只考虑前10个词
            offset = word_features[word] * 0.5
            idx = i * (self.dimension // 10) % self.dimension
            base_vector[idx] += offset

        # 归一化
        norm = sum(x * x for x in base_vector) ** 0.5
        if norm > 0:
            base_vector = [x / norm for x in base_vector]

        self._cache[text] = base_vector
        return base_vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量返回向量"""
        return [self.embed(t) for t in texts]