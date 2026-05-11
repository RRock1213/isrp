"""
检索器

整合切片、向量化、检索流程
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .chunker import Chunk, BaseChunker, ChunkerFactory
from .embedder import Embedder
from .vector_store import VectorStore, VectorDocument, SearchResult
from .config import RAGConfig


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_context_string(self, include_score: bool = False) -> str:
        """转换为上下文字符串"""
        if include_score:
            return f"[相关度: {self.score:.3f}]\n{self.text}"
        return self.text


class Retriever:
    """
    检索器

    负责整合切片、embedding、向量存储和检索
    """

    def __init__(
        self,
        config: RAGConfig,
        chunker: BaseChunker = None,
        embedder: Embedder = None,
        vector_store: VectorStore = None
    ):
        self.config = config

        # 初始化组件
        self.chunker = chunker or ChunkerFactory.create(
            config.chunker_type,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        self.embedder = embedder or Embedder(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.embedding_model,
            dimension=config.embedding_dimension,
            batch_size=config.embedding_batch_size
        )

        self.vector_store = vector_store or VectorStore(
            db_path=config.vector_db_path,
            dimension=config.embedding_dimension
        )

        # 缓存原文（用于检索结果填充）
        self._text_cache: Dict[str, Dict[str, str]] = {}

    def index(self, sample_id: str, text: str) -> bool:
        """
        索引文本

        Args:
            sample_id: 样本 ID（作为 collection 名称）
            text: 待索引的文本

        Returns:
            是否成功
        """
        if not text or len(text) < self.config.min_prompt_length:
            return False

        # 1. 切片
        chunks = self.chunker.chunk(text)
        if not chunks:
            return False

        # 2. 提取文本列表
        texts = [chunk.text for chunk in chunks]

        # 3. 批量 embedding
        embeddings = self.embedder.embed_batch(texts)

        # 4. 构建文档列表
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append(VectorDocument(
                doc_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                metadata={
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "token_count": chunk.token_count,
                    **chunk.metadata
                }
            ))

        # 5. 缓存原文
        self._text_cache[sample_id] = {chunk.chunk_id: chunk.text for chunk in chunks}

        # 6. 插入向量存储
        return self.vector_store.insert(sample_id, documents)

    def retrieve(
        self,
        sample_id: str,
        query: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        检索相关内容

        Args:
            sample_id: 样本 ID
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = self.config.top_k

        if not query:
            return []

        # 1. Query embedding
        query_embedding = self.embedder.embed(query)

        # 2. 向量检索
        search_results = self.vector_store.search(
            collection_name=sample_id,
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 3. 填充原文并转换结果
        results = []
        text_cache = self._text_cache.get(sample_id, {})

        for sr in search_results:
            # 过滤低分结果
            if sr.score < self.config.similarity_threshold:
                continue

            # 获取原文
            text = sr.text or text_cache.get(sr.doc_id, "")

            results.append(RetrievalResult(
                chunk_id=sr.doc_id,
                text=text,
                score=sr.score,
                metadata=sr.metadata
            ))

        return results

    def retrieve_for_node(
        self,
        sample_id: str,
        node_title: str,
        node_description: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        为 DFS 节点检索相关内容

        将节点标题和描述组合为查询

        Args:
            sample_id: 样本 ID
            node_title: 节点标题
            node_description: 节点描述
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 组合查询：标题 + 描述
        query_parts = []
        if node_title:
            query_parts.append(node_title)
        if node_description:
            # 描述可能很长，截取关键部分
            desc = node_description[:500] if len(node_description) > 500 else node_description
            query_parts.append(desc)

        query = "\n".join(query_parts)

        return self.retrieve(sample_id, query, top_k)

    def format_results_as_context(
        self,
        results: List[RetrievalResult],
        max_length: int = 2000
    ) -> str:
        """
        将检索结果格式化为上下文字符串

        Args:
            results: 检索结果列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return ""

        parts = ["【检索到的相关资料】"]
        current_length = len(parts[0])

        for i, result in enumerate(results, 1):
            chunk_text = f"\n\n[{i}] {result.text}"

            if current_length + len(chunk_text) > max_length:
                # 截断
                remaining = max_length - current_length - 10
                if remaining > 100:
                    chunk_text = f"\n\n[{i}] {result.text[:remaining]}..."
                else:
                    break

            parts.append(chunk_text)
            current_length += len(chunk_text)

        return "".join(parts)

    def clear_cache(self, sample_id: str = None):
        """清除缓存"""
        if sample_id:
            self._text_cache.pop(sample_id, None)
            self.vector_store.delete_collection(sample_id)
        else:
            self._text_cache.clear()