"""
ISRP — 哈希工具

统一样本 ID 生成：
  - generate_hash:    基于 MD5 的短哈希（默认 8 位）
  - generate_sample_id: 格式化的样本 ID（sample_XXXX_xxxxxxxx）
在断点续传和 RAG 日志中使用，确保不同运行中的样本可追溯。

"""
import hashlib
from typing import Optional

def generate_hash(text: str, length: Optional[int] = None) -> str:
    """
    生成文本的 MD5 哈希值

    Args:
        text: 要哈希的文本
        length: 可选，截取前 N 个字符。None 表示返回完整哈希

    Returns:
        哈希字符串（32 字符或截取后）

    Examples:
        >>> generate_hash("hello")
        '5d41402abc4b2a76b9719d911017c592'
        >>> generate_hash("hello", length=8)
        '5d41402a'
    """
    hash_value = hashlib.md5(text.encode('utf-8')).hexdigest()
    return hash_value[:length] if length else hash_value

def generate_sample_id(prompt: str, index: Optional[int] = None) -> str:
    """
    生成标准化的样本 ID

    统一格式：sample_{index:04d}_{hash8} 或 sample_{hash8}

    Args:
        prompt: 用户 prompt
        index: 可选，样本索引

    Returns:
        标准化的样本 ID

    Examples:
        >>> generate_sample_id("写一篇文章", index=0)
        'sample_0000_a1b2c3d4'
        >>> generate_sample_id("写一篇文章")
        'sample_a1b2c3d4'
    """
    hash_part = generate_hash(prompt, length=8)
    if index is not None:
        return f"sample_{index:04d}_{hash_part}"
    return f"sample_{hash_part}"