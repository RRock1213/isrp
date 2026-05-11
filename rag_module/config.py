"""
RAG 模块配置
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import os
import sys
from pathlib import Path


def _get_dashscope_api_key() -> str:
    """
    获取 dashscope API key

    优先级：
    1. 环境变量 DASHSCOPE_API_KEY
    2. 父目录的 config.py 中的 API_KEYS["dashscope"]
    """
    # 优先从环境变量获取
    env_key = os.getenv("DASHSCOPE_API_KEY", "")
    if env_key:
        return env_key

    # 尝试从父目录的 config.py 导入
    try:
        # rag_module 的父目录是项目根目录
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from config import API_KEYS
        return API_KEYS.get("dashscope", "")
    except ImportError:
        pass
    except Exception as e:
        print(f"[RAG Config] Warning: 无法从父目录 config.py 获取 API key: {e}")

    return ""


@dataclass
class RAGConfig:
    """RAG 模块配置"""

    # 基础开关
    enabled: bool = True

    # 切片器配置
    chunker_type: Literal["recursive", "semantic", "sentence", "token", "fast", "code"] = "recursive"
    chunk_size: int = 512  # 目标切片大小（tokens）
    chunk_overlap: int = 50  # 切片重叠

    # Embedding 配置
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 10  # 每批处理数量

    # 向量存储配置
    vector_db_path: str = "rag_cache/vectors"
    cache_dir: str = "rag_cache"

    # 检索配置
    top_k: int = 3  # 检索返回数量
    retrieval_level: Literal["all", "level1", "level0"] = "all"
    min_prompt_length: int = 2000  # 触发 RAG 的最小 prompt 长度
    similarity_threshold: float = 0.3  # 相似度阈值

    # API 配置
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __post_init__(self):
        """初始化后处理"""
        # 优先从父目录 config.py 获取 API key
        if self.api_key is None:
            self.api_key = _get_dashscope_api_key()

        # 确保目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "chunks"), exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)


# 默认配置实例
DEFAULT_RAG_CONFIG = RAGConfig()