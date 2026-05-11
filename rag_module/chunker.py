"""
切片器封装

使用 Chonkie 库提供多种切片策略：
- RecursiveChunker: 层级规则切片（推荐）
- SemanticChunker: 语义相似度切片
- SentenceChunker: 句子切片
- TokenChunker: Token 级别切片
- FastChunker: SIMD 加速字节切片
- CodeChunker: 代码结构切片
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import re

try:
    from chonkie import (
        RecursiveChunker as ChonkieRecursiveChunker,
        SemanticChunker as ChonkieSemanticChunker,
        SentenceChunker as ChonkieSentenceChunker,
        TokenChunker as ChonkieTokenChunker,
        FastChunker as ChonkieFastChunker,
        CodeChunker as ChonkieCodeChunker
    )
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False


@dataclass
class Chunk:
    """切片结果"""
    chunk_id: str
    text: str
    start_idx: int  # 原文起始位置
    end_idx: int    # 原文结束位置
    token_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseChunker(ABC):
    """切片器基类"""

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """切片文本"""
        pass


class ChonkieChunkerWrapper(BaseChunker):
    """
    Chonkie 切片器封装

    使用 chonkie 库提供的切片器
    """

    def __init__(
        self,
        chunker_type: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ):
        if not CHONKIE_AVAILABLE:
            raise ImportError(
                "chonkie 库未安装，请运行: pip install chonkie"
            )
        
        self.chunker_type = chunker_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs
        
        self._init_chunker()

    def _init_chunker(self):
        """初始化 chonkie 切片器"""
        if self.chunker_type == "recursive":
            # RecursiveChunker 不接受 chunk_overlap 参数
            self.chunker = ChonkieRecursiveChunker(
                chunk_size=self.chunk_size,
                **self.kwargs
            )
        elif self.chunker_type == "semantic":
            self.chunker = ChonkieSemanticChunker(
                chunk_size=self.chunk_size,
                **self.kwargs
            )
        elif self.chunker_type == "sentence":
            # 检查 SentenceChunker 是否接受 chunk_overlap
            try:
                self.chunker = ChonkieSentenceChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    **self.kwargs
                )
            except TypeError:
                # 如果不接受，只传递 chunk_size
                self.chunker = ChonkieSentenceChunker(
                    chunk_size=self.chunk_size,
                    **self.kwargs
                )
        elif self.chunker_type == "token":
            # 检查 TokenChunker 是否接受 chunk_overlap
            try:
                self.chunker = ChonkieTokenChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    **self.kwargs
                )
            except TypeError:
                self.chunker = ChonkieTokenChunker(
                    chunk_size=self.chunk_size,
                    **self.kwargs
                )
        elif self.chunker_type == "fast":
            # 检查 FastChunker 是否接受 chunk_overlap
            try:
                self.chunker = ChonkieFastChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    **self.kwargs
                )
            except TypeError:
                self.chunker = ChonkieFastChunker(
                    chunk_size=self.chunk_size,
                    **self.kwargs
                )
        elif self.chunker_type == "code":
            # 检查 CodeChunker 是否接受 chunk_overlap
            try:
                self.chunker = ChonkieCodeChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    **self.kwargs
                )
            except TypeError:
                self.chunker = ChonkieCodeChunker(
                    chunk_size=self.chunk_size,
                    **self.kwargs
                )
        else:
            raise ValueError(f"未知的 chonkie 切片器类型: {self.chunker_type}")

    def chunk(self, text: str) -> List[Chunk]:
        """使用 chonkie 切片文本"""
        if not text or not text.strip():
            return []

        try:
            chonkie_chunks = self.chunker(text)
        except Exception as e:
            print(f"Chonkie 切片失败，使用备用方案: {e}")
            return self._fallback_chunk(text)

        chunks = []
        current_pos = 0

        for idx, chonkie_chunk in enumerate(chonkie_chunks):
            chunk_text = chonkie_chunk.text
            start_idx = text.find(chunk_text, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(chunk_text)
            current_pos = end_idx

            chunks.append(Chunk(
                chunk_id=f"chunk_{idx+1:04d}",
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                token_count=getattr(chonkie_chunk, 'token_count', 0),
                metadata={
                    "length": len(chunk_text),
                    "chunker": "chonkie",
                    "type": self.chunker_type
                }
            ))

        return chunks

    def _fallback_chunk(self, text: str) -> List[Chunk]:
        """备用切片方案（简单的段落分割）"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        
        for idx, para in enumerate(paragraphs):
            start_idx = text.find(para)
            chunks.append(Chunk(
                chunk_id=f"chunk_{idx+1:04d}",
                text=para,
                start_idx=start_idx,
                end_idx=start_idx + len(para),
                token_count=len(para) // 2,
                metadata={
                    "length": len(para),
                    "chunker": "fallback"
                }
            ))
        
        return chunks


class ChunkerFactory:
    """切片器工厂"""

    @staticmethod
    def create(chunker_type: str, **kwargs) -> BaseChunker:
        """创建切片器实例"""
        if CHONKIE_AVAILABLE:
            try:
                return ChonkieChunkerWrapper(chunker_type=chunker_type, **kwargs)
            except Exception as e:
                print(f"使用 chonkie 失败，将使用自定义实现: {e}")
        
        if chunker_type == "recursive":
            return RecursiveChunkerWrapper(**kwargs)
        elif chunker_type == "sentence":
            return SentenceChunkerWrapper(**kwargs)
        elif chunker_type == "semantic":
            raise NotImplementedError("SemanticChunker 需要安装 chonkie 库")
        else:
            raise ValueError(f"未知的切片器类型: {chunker_type}")


class RecursiveChunkerWrapper(BaseChunker):
    """
    RecursiveChunker 封装（备用实现）

    使用层级规则切片：段落 → 句子 → Token
    无需额外模型，速度快，语义保持良好
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n", "\n", "。", ".", "！", "!", "？", "?",
            "；", ";", "，", ",", " ", ""
        ]

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        chunks = []
        self._chunk_counter = 0
        paragraphs = self._split_by_separator(text, "\n\n")
        current_chunk_text = ""
        current_start = 0

        for para in paragraphs:
            para_text = para.strip()
            if not para_text:
                continue

            if self._estimate_tokens(para_text) > self.chunk_size:
                if current_chunk_text:
                    chunks.append(self._create_chunk(current_chunk_text, current_start, text))
                    current_chunk_text = ""
                sub_chunks = self._recursive_split(para_text, self.separators[1:])
                for sub in sub_chunks:
                    sub_start = text.find(sub, current_start)
                    chunks.append(self._create_chunk(sub, sub_start, text))
                current_start = text.find(para_text, current_start) + len(para_text)
            else:
                combined = current_chunk_text + "\n\n" + para_text if current_chunk_text else para_text
                if self._estimate_tokens(combined) <= self.chunk_size:
                    if not current_chunk_text:
                        current_start = text.find(para_text)
                    current_chunk_text = combined
                else:
                    if current_chunk_text:
                        chunks.append(self._create_chunk(current_chunk_text, current_start, text))
                    current_start = text.find(para_text)
                    current_chunk_text = para_text

        if current_chunk_text:
            chunks.append(self._create_chunk(current_chunk_text, current_start, text))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators or self._estimate_tokens(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]
        parts = text.split(separator)
        result = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if self._estimate_tokens(part) > self.chunk_size:
                if current:
                    result.append(current)
                    current = ""
                sub_parts = self._recursive_split(part, remaining_separators)
                result.extend(sub_parts)
            else:
                combined = current + separator + part if current else part
                if self._estimate_tokens(combined) <= self.chunk_size:
                    current = combined
                else:
                    if current:
                        result.append(current)
                    current = part

        if current:
            result.append(current)

        return result

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        return [p.strip() for p in text.split(separator) if p.strip()]

    def _estimate_tokens(self, text: str) -> int:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
        return chinese_chars + english_words + other_chars // 4

    def _create_chunk(self, text: str, start_idx: int, original_text: str) -> Chunk:
        self._chunk_counter += 1
        end_idx = start_idx + len(text)
        return Chunk(
            chunk_id=f"chunk_{self._chunk_counter:04d}",
            text=text,
            start_idx=start_idx,
            end_idx=end_idx,
            token_count=self._estimate_tokens(text),
            metadata={"length": len(text)}
        )


class SentenceChunkerWrapper(BaseChunker):
    """
    SentenceChunker 封装（备用实现）

    按句子切分，简单可靠
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 1):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sentence_endings = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_text = ""
        current_start = 0
        chunk_counter = 0

        for sent in sentences:
            if not current_text:
                current_start = text.find(sent)

            combined = current_text + " " + sent if current_text else sent

            if len(combined) > self.chunk_size * 2:
                if current_text:
                    chunk_counter += 1
                    chunks.append(Chunk(
                        chunk_id=f"chunk_{chunk_counter:04d}",
                        text=current_text,
                        start_idx=current_start,
                        end_idx=current_start + len(current_text),
                        token_count=len(current_text) // 2
                    ))
                current_text = sent
                current_start = text.find(sent)
            else:
                current_text = combined

        if current_text:
            chunk_counter += 1
            chunks.append(Chunk(
                chunk_id=f"chunk_{chunk_counter:04d}",
                text=current_text,
                start_idx=current_start,
                end_idx=current_start + len(current_text),
                token_count=len(current_text) // 2
            ))

        return chunks


# 为向后兼容提供别名
# 如果 chonkie 可用，RecursiveChunker 指向 ChonkieChunkerWrapper
# 否则使用备用的 RecursiveChunkerWrapper
if CHONKIE_AVAILABLE:
    RecursiveChunker = ChonkieChunkerWrapper
else:
    RecursiveChunker = RecursiveChunkerWrapper