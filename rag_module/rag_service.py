"""
RAG 服务统一入口

提供简洁的 API 接口，整合切片、向量化、检索流程
支持缓存复用和详细日志
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import os
import json
import time
from datetime import datetime
from pathlib import Path

from .config import RAGConfig, DEFAULT_RAG_CONFIG
from .chunker import Chunk, BaseChunker, ChunkerFactory
from .embedder import Embedder, MockEmbedder
from .vector_store import VectorStore
from .retriever import Retriever, RetrievalResult
from .cache_manager import CacheManager


@dataclass
class IndexResult:
    """索引结果"""
    success: bool
    sample_id: str
    chunk_count: int
    cached: bool
    message: str = ""


@dataclass
class RetrievalLog:
    """检索日志"""
    node_title: str
    node_level: int
    query: str
    results: List[Dict]
    context_length: int


class RAGService:
    """
    RAG 服务统一入口

    使用示例：
    ```python
    # 初始化
    rag = RAGService(config)

    # 索引样本
    rag.index_prompt("sample_1", long_prompt)

    # 检索
    results = rag.retrieve("sample_1", "查询内容")

    # 为节点检索
    results = rag.retrieve_for_node("sample_1", node_title, node_desc)
    ```
    """

    def __init__(self, config: RAGConfig = None, use_mock: bool = False, log_dir: str = None):
        """
        初始化 RAG 服务

        Args:
            config: RAG 配置，为 None 时使用默认配置
            use_mock: 是否使用 Mock Embedder（用于测试）
            log_dir: RAG 日志目录（可选）
        """
        self.config = config or DEFAULT_RAG_CONFIG
        self.use_mock = use_mock

        # 日志系统
        self.log_dir = log_dir
        self._sample_logs: Dict[str, Dict] = {}  # 样本级日志缓存
        self._max_cached_logs = 100  # 最大缓存日志数量，防止内存泄漏

        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self._init_components()

        # 状态追踪
        self._indexed_samples: Dict[str, bool] = {}

    def _init_components(self):
        """初始化组件"""
        # 缓存管理器
        self.cache_manager = CacheManager(cache_dir=self.config.cache_dir)

        # 切片器
        self.chunker = ChunkerFactory.create(
            self.config.chunker_type,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Embedder
        if self.use_mock:
            self.embedder = MockEmbedder(dimension=self.config.embedding_dimension)
        else:
            self.embedder = Embedder(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.embedding_model,
                dimension=self.config.embedding_dimension,
                batch_size=self.config.embedding_batch_size
            )

        # 向量存储
        self.vector_store = VectorStore(
            db_path=self.config.vector_db_path,
            dimension=self.config.embedding_dimension
        )

        # 检索器
        self.retriever = Retriever(
            config=self.config,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store
        )

    # ========================================================================
    # 日志方法
    # ========================================================================

    def _init_sample_log(self, sample_id: str, prompt_length: int):
        """初始化样本日志"""
        # 自动清理超出限制的旧日志
        if len(self._sample_logs) >= self._max_cached_logs:
            # 移除最早的一半日志（假设按插入顺序）
            keys_to_remove = list(self._sample_logs.keys())[:self._max_cached_logs // 2]
            for key in keys_to_remove:
                del self._sample_logs[key]

        if sample_id not in self._sample_logs:
            self._sample_logs[sample_id] = {
                "sample_id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_length": prompt_length,
                "rag_enabled": True,
                "config": {
                    "chunker_type": self.config.chunker_type,
                    "chunk_size": self.config.chunk_size,
                    "top_k": self.config.top_k,
                    "min_prompt_length": self.config.min_prompt_length,
                    "similarity_threshold": self.config.similarity_threshold,
                    "embedding_model": self.config.embedding_model,
                    "embedding_dimension": self.config.embedding_dimension
                },
                "chunking": None,
                "indexing": None,
                "retrievals": [],
                "summary": {
                    "total_retrievals": 0,
                    "total_context_length": 0,
                    "avg_score": 0.0
                }
            }

    def log_shortcut_path_sample(self, sample_id: str, prompt_length: int,
                                  confidence: float = None, chapters: list = None):
        """
        为快捷路径样本创建简化日志

        当 prompt 长度足够触发 RAG 但因快捷路径跳过索引时，
        创建简化日志记录快捷路径信息

        Args:
            sample_id: 样本 ID
            prompt_length: prompt 长度
            confidence: 快捷路径置信度
            chapters: 检测到的章节列表
        """
        if not self.log_dir:
            return

        # 初始化基础日志
        self._init_sample_log(sample_id, prompt_length)

        # 更新日志内容
        log_data = self._sample_logs[sample_id]
        log_data["rag_triggered"] = prompt_length >= self.config.min_prompt_length
        log_data["shortcut_path"] = True
        log_data["shortcut_reason"] = "explicit_chapter_frame_detected"

        if confidence is not None:
            log_data["shortcut_confidence"] = confidence
        if chapters:
            log_data["detected_chapters"] = chapters

        # 更新摘要说明
        log_data["summary"]["note"] = "快捷路径样本：检测到明确章节框架，跳过 RAG 索引"

        # 保存日志
        self._save_sample_log(sample_id)

        print(f"[RAG Log] Shortcut sample logged: {sample_id}")

    def _log_chunking(self, sample_id: str, chunker_type: str, chunk_size: int,
                      total_chunks: int, cached: bool, chunk_details: List[Dict] = None):
        """记录切片日志"""
        if sample_id in self._sample_logs:
            chunking_log = {
                "chunker_type": chunker_type,
                "chunk_size": chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "total_chunks": total_chunks,
                "cached": cached
            }
            # 可选：记录每个切片的详细信息
            if chunk_details and not cached:
                chunking_log["chunks_preview"] = [
                    {
                        "chunk_id": i,
                        "text_preview": c.get("text", "")[:100] + "..." if len(c.get("text", "")) > 100 else c.get("text", ""),
                        "token_count": c.get("token_count", 0)
                    }
                    for i, c in enumerate(chunk_details[:5])  # 只记录前5个
                ]

            self._sample_logs[sample_id]["chunking"] = chunking_log

    def _log_indexing(self, sample_id: str, success: bool, embedding_dimension: int = None,
                       time_ms: float = None, cached: bool = False):
        """记录索引日志"""
        if sample_id in self._sample_logs:
            self._sample_logs[sample_id]["indexing"] = {
                "success": success,
                "embedding_dimension": embedding_dimension or self.config.embedding_dimension,
                "time_ms": round(time_ms, 2) if time_ms else None,
                "cached": cached
            }

    def _log_retrieval(self, sample_id: str, node_title: str, node_level: int,
                       query: str, results: List[RetrievalResult], context_length: int):
        """记录检索日志"""
        if sample_id not in self._sample_logs:
            return

        # 提取详细的检索结果
        detailed_results = []
        total_score = 0.0
        for i, r in enumerate(results):
            score = r.score if hasattr(r, 'score') else 0.0
            total_score += score

            result_entry = {
                "chunk_id": r.chunk_id if hasattr(r, 'chunk_id') else i,
                "score": round(score, 4),
                "text_preview": ""
            }

            # 提取文本预览
            if hasattr(r, 'text') and r.text:
                text = r.text
                result_entry["text_preview"] = text[:150] + "..." if len(text) > 150 else text
                result_entry["text_length"] = len(text)

            detailed_results.append(result_entry)

        retrieval_log = {
            "timestamp": datetime.now().isoformat(),
            "node_title": node_title,
            "node_level": node_level,
            "query": query[:200] + "..." if len(query) > 200 else query,
            "top_k": self.config.top_k,
            "result_count": len(results),
            "results": detailed_results,
            "context_length": context_length,
            "avg_score": round(total_score / len(results), 4) if results else 0.0
        }

        self._sample_logs[sample_id]["retrievals"].append(retrieval_log)

        # 更新汇总信息
        summary = self._sample_logs[sample_id]["summary"]
        summary["total_retrievals"] += 1
        summary["total_context_length"] += context_length

        # 计算平均分数
        all_scores = [r["score"] for r in detailed_results if r["score"] > 0]
        if all_scores:
            summary["avg_score"] = round(sum(all_scores) / len(all_scores), 4)

    def _save_sample_log(self, sample_id: str):
        """保存样本日志到文件"""
        if not self.log_dir or sample_id not in self._sample_logs:
            return

        log_data = self._sample_logs[sample_id]

        # 根据实际检索情况添加/移除说明
        retrievals = log_data.get("retrievals", [])
        if retrievals and len(retrievals) > 0:
            # 有检索记录，移除可能的错误 note
            log_data.pop("retrieval_note", None)
            # 更新 summary.note 为准确描述
            summary = log_data.get("summary", {})
            if summary.get("note", "").startswith("快捷路径样本"):
                summary["note"] = f"快捷路径样本：已完成 RAG 索引和 {len(retrievals)} 次检索"
        else:
            # 无检索记录，添加说明
            log_data["retrieval_note"] = "检索未触发： 快捷路径直接创建了章节节点，绕过了 DFS 扩展中的 RAG 检索步骤"

        log_file = Path(self.log_dir) / f"{sample_id}.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            print(f"[RAG Log] Saved: {log_file}")
        except Exception as e:
            print(f"[RAG Log] Failed to save log for {sample_id}: {e}")

    def get_sample_log(self, sample_id: str) -> Optional[Dict]:
        """获取样本日志"""
        return self._sample_logs.get(sample_id)

    def clear_sample_log(self, sample_id: str):
        """清除样本日志缓存"""
        self._sample_logs.pop(sample_id, None)

    # ========================================================================
    # 核心方法
    # ========================================================================

    def is_enabled(self) -> bool:
        """检查 RAG 是否启用"""
        return self.config.enabled

    def should_use_rag(self, prompt: str) -> bool:
        """检查是否应该使用 RAG"""
        if not self.config.enabled:
            return False
        if not prompt or len(prompt) < self.config.min_prompt_length:
            return False
        return True

    def index_prompt(
        self,
        sample_id: str,
        prompt: str,
        force_reindex: bool = False
    ) -> IndexResult:
        """
        索引样本 prompt

        自动检查缓存，如已存在有效缓存则跳过

        Args:
            sample_id: 样本 ID
            prompt: 用户 prompt
            force_reindex: 是否强制重新索引

        Returns:
            索引结果
        """
        if not self.should_use_rag(prompt):
            return IndexResult(
                success=False,
                sample_id=sample_id,
                chunk_count=0,
                cached=False,
                message="Prompt 太短或 RAG 未启用"
            )

        # 初始化样本日志
        self._init_sample_log(sample_id, len(prompt))
        index_start_time = time.time()  # 记录开始时间

        # 优先检查向量索引是否存在（支持断点续传复用）
        if not force_reindex and self.cache_manager.has_vector_index(sample_id):
            # 向量索引存在，直接加载
            self.vector_store.load_index(sample_id)
            self._indexed_samples[sample_id] = True

            # 尝试加载切片缓存获取 chunk 数量
            chunks_text = self.cache_manager.get_cached_chunks_text(sample_id)
            chunk_count = len(chunks_text) if chunks_text else 0

            # 不再为获取数量而重新切片，避免冗余计算
            # 如果切片缓存不存在，返回 0 作为占位符（向量化已完成，切片数仅用于日志）
            # 切片缓存会在下次实际需要切片时重建

            # 记录日志（从缓存加载）
            self._log_chunking(sample_id, self.config.chunker_type,
                              self.config.chunk_size, chunk_count, cached=True)
            self._log_indexing(sample_id, True, cached=True)
            self._save_sample_log(sample_id)

            print(f"[RAG] Loaded from cache: {sample_id}, chunks: {chunk_count}")

            return IndexResult(
                success=True,
                sample_id=sample_id,
                chunk_count=chunk_count,
                cached=True,
                message="从向量缓存加载成功"
            )

        # 检查完整缓存（切片 + 向量）
        if not force_reindex and self.cache_manager.is_cache_valid(
            sample_id, prompt,
            self.config.chunker_type,
            self.config.chunk_size
        ):
            # 加载缓存的切片文本
            chunks_text = self.cache_manager.get_cached_chunks_text(sample_id)

            # 检查向量索引是否存在
            if self.cache_manager.has_vector_index(sample_id):
                # 加载向量索引
                self.vector_store.load_index(sample_id)
                self._indexed_samples[sample_id] = True

                # 记录缓存日志
                self._log_chunking(sample_id, self.config.chunker_type,
                                  self.config.chunk_size, len(chunks_text), cached=True)
                self._log_indexing(sample_id, True, cached=True)
                self._save_sample_log(sample_id)

                print(f"[RAG] Loaded from full cache: {sample_id}, chunks: {len(chunks_text)}")

                return IndexResult(
                    success=True,
                    sample_id=sample_id,
                    chunk_count=len(chunks_text),
                    cached=True,
                    message="从缓存加载成功"
                )

        # 执行切片
        print(f"[RAG] Chunking prompt for {sample_id}...")
        chunk_start_time = time.time()
        chunks = self.chunker.chunk(prompt)
        chunk_time = (time.time() - chunk_start_time) * 1000  # ms

        if not chunks:
            self._log_indexing(sample_id, False)
            self._save_sample_log(sample_id)
            return IndexResult(
                success=False,
                sample_id=sample_id,
                chunk_count=0,
                cached=False,
                message="切片失败"
            )

        print(f"[RAG] Chunked into {len(chunks)} chunks in {chunk_time:.1f}ms")

        # 记录切片日志（包含切片详情）
        chunk_details = [{"text": c.text if hasattr(c, 'text') else str(c),
                          "token_count": c.token_count if hasattr(c, 'token_count') else 0}
                         for c in chunks]
        self._log_chunking(sample_id, self.config.chunker_type,
                          self.config.chunk_size, len(chunks), cached=False, chunk_details=chunk_details)

        # 保存切片缓存
        self.cache_manager.save_chunks(
            sample_id=sample_id,
            prompt=prompt,
            chunker_type=self.config.chunker_type,
            chunk_size=self.config.chunk_size,
            chunks=chunks
        )

        # 执行索引（向量化）
        print(f"[RAG] Indexing {len(chunks)} chunks for {sample_id}...")
        index_start = time.time()
        success = self.retriever.index(sample_id, prompt)
        index_time = (time.time() - index_start) * 1000  # ms

        total_time = (time.time() - index_start_time) * 1000  # 总耗时

        if success:
            # 保存向量索引
            self.vector_store.save_index(sample_id)
            self._indexed_samples[sample_id] = True

            # 记录索引成功日志（包含时间）
            self._log_indexing(sample_id, True, time_ms=index_time, cached=False)
            self._save_sample_log(sample_id)

            print(f"[RAG] Indexed {len(chunks)} chunks in {index_time:.1f}ms, total: {total_time:.1f}ms")

            return IndexResult(
                success=True,
                sample_id=sample_id,
                chunk_count=len(chunks),
                cached=False,
                message=f"索引成功，耗时 {total_time:.1f}ms"
            )
        else:
            self._log_indexing(sample_id, False, time_ms=index_time, cached=False)
            self._save_sample_log(sample_id)
            return IndexResult(
                success=False,
                sample_id=sample_id,
                chunk_count=0,
                cached=False,
                message="向量化失败"
            )

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
        if not self.config.enabled:
            return []

        if sample_id not in self._indexed_samples:
            # 尝试加载缓存
            if self.cache_manager.has_vector_index(sample_id):
                self.vector_store.load_index(sample_id)
                self._indexed_samples[sample_id] = True
            else:
                return []

        return self.retriever.retrieve(sample_id, query, top_k)

    def retrieve_for_node(
        self,
        sample_id: str,
        node_title: str,
        node_description: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        为 DFS 节点检索相关内容

        Args:
            sample_id: 样本 ID
            node_title: 节点标题
            node_description: 节点描述
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if not self.config.enabled:
            return []

        if sample_id not in self._indexed_samples:
            if self.cache_manager.has_vector_index(sample_id):
                self.vector_store.load_index(sample_id)
                self._indexed_samples[sample_id] = True
            else:
                return []

        return self.retriever.retrieve_for_node(
            sample_id, node_title, node_description, top_k
        )

    def format_as_context(
        self,
        results: List[RetrievalResult],
        max_length: int = 2000
    ) -> str:
        """
        将检索结果格式化为上下文

        Args:
            results: 检索结果
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        return self.retriever.format_results_as_context(results, max_length)

    def get_rag_context_for_node(
        self,
        sample_id: str,
        node_title: str,
        node_description: str,
        top_k: int = None,
        max_length: int = 2000,
        node_level: int = 1  # 节点层级，用于日志
    ) -> str:
        """
        一站式获取节点的 RAG 上下文

        组合检索 + 格式化

        Args:
            sample_id: 样本 ID
            node_title: 节点标题
            node_description: 节点描述
            top_k: 检索数量
            max_length: 最大长度
            node_level: 节点层级（用于日志）

        Returns:
            RAG 上下文字符串（如无结果返回空字符串）
        """
        results = self.retrieve_for_node(sample_id, node_title, node_description, top_k)

        if not results:
            return ""

        context = self.format_as_context(results, max_length)

        # 记录检索日志
        query = f"{node_title}: {node_description}"
        self._log_retrieval(sample_id, node_title, node_level, query, results, len(context))

        return context

    def save_all_logs(self):
        """保存所有样本日志（在处理完成后调用）"""
        if not self.log_dir:
            return

        for sample_id in list(self._sample_logs.keys()):
            self._save_sample_log(sample_id)

    def clear_sample(self, sample_id: str, clear_cache: bool = True):
        """
        清除指定样本的缓存

        Args:
            sample_id: 样本 ID
            clear_cache: 是否清除磁盘缓存（默认 True）。设为 False 仅保存日志和释放内存。
        """
        # 先保存日志
        self._save_sample_log(sample_id)

        # 关闭向量存储，释放文件句柄
        self.vector_store.close()

        if clear_cache:
            # 清除磁盘缓存（包括切片缓存和向量缓存）
            self.cache_manager.clear_sample_cache(sample_id)

        # 清除内存中的索引记录
        self._indexed_samples.pop(sample_id, None)
        # 清除检索器的内存缓存
        self.retriever._text_cache.pop(sample_id, None)
        # 清除日志缓存
        self.clear_sample_log(sample_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取服务状态"""
        cache_stats = self.cache_manager.get_cache_stats()

        return {
            "enabled": self.config.enabled,
            "chunker_type": self.config.chunker_type,
            "embedding_model": self.config.embedding_model,
            "indexed_samples": list(self._indexed_samples.keys()),
            "cache_stats": cache_stats
        }


# 便捷工厂函数
def create_rag_service(
    enabled: bool = True,
    chunker_type: str = "recursive",
    top_k: int = 5,
    min_prompt_length: int = 2000,
    api_key: str = None,
    log_dir: str = None,  # 日志目录
    **kwargs
) -> RAGService:
    """
    创建 RAG 服务的便捷函数

    Args:
        enabled: 是否启用
        chunker_type: 切片器类型
        top_k: 检索数量
        min_prompt_length: 最小 prompt 长度
        api_key: API 密钥
        log_dir: RAG 日志目录
        **kwargs: 其他配置参数

    Returns:
        RAGService 实例
    """
    config = RAGConfig(
        enabled=enabled,
        chunker_type=chunker_type,
        top_k=top_k,
        min_prompt_length=min_prompt_length,
        api_key=api_key,
        **kwargs
    )

    return RAGService(config, log_dir=log_dir)