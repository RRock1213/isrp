"""
缓存管理器

管理切片结果和向量索引的持久化
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time


@dataclass
class ChunkCache:
    """切片缓存结构"""
    sample_id: str
    prompt_hash: str
    chunker_type: str
    chunk_size: int
    chunks: List[Dict[str, Any]]
    created_at: float
    metadata: Dict[str, Any] = None


class CacheManager:
    """
    缓存管理器

    负责：
    1. 切片结果的持久化和加载
    2. 向量索引的持久化状态管理
    3. 缓存有效性检查
    """

    def __init__(self, cache_dir: str = "rag_cache"):
        self.cache_dir = cache_dir
        self.chunks_dir = os.path.join(cache_dir, "chunks")
        self.vectors_dir = os.path.join(cache_dir, "vectors")

        # 确保目录存在
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)

    def compute_prompt_hash(self, prompt: str) -> str:
        """计算 prompt 哈希"""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()

    def get_chunk_cache_path(self, sample_id: str) -> str:
        """获取切片缓存文件路径"""
        return os.path.join(self.chunks_dir, f"{sample_id}.json")

    def get_vector_cache_path(self, sample_id: str) -> str:
        """获取向量缓存目录路径"""
        return os.path.join(self.vectors_dir, sample_id)

    def has_chunks(self, sample_id: str) -> bool:
        """检查切片缓存是否存在"""
        path = self.get_chunk_cache_path(sample_id)
        return os.path.exists(path)

    def has_vector_index(self, sample_id: str) -> bool:
        """检查向量索引是否存在"""
        # zvec 目录或 fallback json
        vec_path = self.get_vector_cache_path(sample_id)
        fallback_path = os.path.join(self.vectors_dir, f"{sample_id}_fallback.json")
        return os.path.exists(vec_path) or os.path.exists(fallback_path)

    def is_cache_valid(
        self,
        sample_id: str,
        prompt: str,
        chunker_type: str,
        chunk_size: int
    ) -> bool:
        """
        检查缓存是否有效

        验证：
        1. 缓存文件存在
        2. prompt hash 匹配
        3. 切片参数匹配
        """
        if not self.has_chunks(sample_id):
            return False

        cache = self.load_chunks(sample_id)
        if cache is None:
            return False

        # 验证 hash
        current_hash = self.compute_prompt_hash(prompt)
        if cache.prompt_hash != current_hash:
            return False

        # 验证参数
        if cache.chunker_type != chunker_type:
            return False
        if cache.chunk_size != chunk_size:
            return False

        return True

    def save_chunks(
        self,
        sample_id: str,
        prompt: str,
        chunker_type: str,
        chunk_size: int,
        chunks: List[Any],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        保存切片结果到缓存

        Args:
            sample_id: 样本 ID
            prompt: 原始 prompt
            chunker_type: 切片器类型
            chunk_size: 切片大小
            chunks: 切片列表
            metadata: 额外元数据

        Returns:
            是否成功
        """
        try:
            # 转换 chunks 为可序列化格式
            chunks_data = []
            for chunk in chunks:
                if hasattr(chunk, '__dataclass_fields__'):
                    chunks_data.append(asdict(chunk))
                elif hasattr(chunk, 'to_dict'):
                    chunks_data.append(chunk.to_dict())
                else:
                    chunks_data.append({
                        "chunk_id": getattr(chunk, 'chunk_id', ''),
                        "text": getattr(chunk, 'text', ''),
                        "start_idx": getattr(chunk, 'start_idx', 0),
                        "end_idx": getattr(chunk, 'end_idx', 0),
                        "token_count": getattr(chunk, 'token_count', 0),
                        "metadata": getattr(chunk, 'metadata', {})
                    })

            cache = ChunkCache(
                sample_id=sample_id,
                prompt_hash=self.compute_prompt_hash(prompt),
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunks=chunks_data,
                created_at=time.time(),
                metadata=metadata
            )

            path = self.get_chunk_cache_path(sample_id)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(asdict(cache), f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"[CacheManager] Save chunks error: {e}")
            return False

    def load_chunks(self, sample_id: str) -> Optional[ChunkCache]:
        """加载切片缓存"""
        path = self.get_chunk_cache_path(sample_id)

        if not os.path.exists(path):
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return ChunkCache(**data)

        except Exception as e:
            print(f"[CacheManager] Load chunks error: {e}")
            return None

    def get_cached_chunks_text(self, sample_id: str) -> List[str]:
        """获取缓存的切片文本列表"""
        cache = self.load_chunks(sample_id)
        if cache is None:
            return []

        return [chunk.get("text", "") for chunk in cache.chunks]

    def clear_sample_cache(self, sample_id: str) -> bool:
        """清除指定样本的缓存"""
        def _retry_delete(path, max_retries=3, delay=0.5):
            """带重试的删除操作"""
            for i in range(max_retries):
                try:
                    if os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    return True
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"[CacheManager] 删除 {path} 失败，{delay}秒后重试: {e}")
                        time.sleep(delay)
                    else:
                        print(f"[CacheManager] 最终删除 {path} 失败: {e}")
                        return False

        try:
            # 切片缓存
            chunk_path = self.get_chunk_cache_path(sample_id)
            if os.path.exists(chunk_path):
                _retry_delete(chunk_path)

            # 向量缓存目录
            vec_path = self.get_vector_cache_path(sample_id)
            if os.path.exists(vec_path):
                _retry_delete(vec_path)

            # Fallback json
            fallback_path = os.path.join(self.vectors_dir, f"{sample_id}_fallback.json")
            if os.path.exists(fallback_path):
                _retry_delete(fallback_path)

            return True

        except Exception as e:
            print(f"[CacheManager] Clear cache error: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        chunk_files = []
        if os.path.exists(self.chunks_dir):
            chunk_files = [f for f in os.listdir(self.chunks_dir) if f.endswith('.json')]

        vector_dirs = []
        if os.path.exists(self.vectors_dir):
            vector_dirs = [d for d in os.listdir(self.vectors_dir)
                          if os.path.isdir(os.path.join(self.vectors_dir, d))
                          or d.endswith('_fallback.json')]

        return {
            "chunk_cache_count": len(chunk_files),
            "vector_cache_count": len(vector_dirs),
            "cache_dir": self.cache_dir
        }