"""
ISRP — Prompt 智能压缩器

当用户指令超过长度阈值（默认 2000 字符）时触发压缩：
  - 提炼核心主题 tau（一句话）
  - 提取关键需求列表 kappa（显式约束 + 格式要求）
  - 保留强制信息 I（必须出现在生成内容中的实体、数据、事件）
  - 压缩至目标长度，保留所有关键信息

压缩后的表示 (tau, kappa, I) 在后续每次 LLM 调用中作为全局记忆注入。

"""
from typing import Optional

from ..schemas import CompressedPrompt, detect_language
from .llm_client import LLMClient

class PromptCompressor:
    """
    Prompt 智能压缩器

    当用户指令超过阈值时，使用 LLM 进行信息提纯和压缩，
    保留所有关键信息的同时减少上下文长度。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompts,  # PromptTemplates 实例
        enable_logging: bool = True
    ):
        """
        初始化压缩器

        Args:
            llm_client: LLM 客户端
            prompts: Prompt 模板实例
            enable_logging: 是否启用日志
        """
        self.llm = llm_client
        self.prompts = prompts
        self.enable_logging = enable_logging

    def compress(
        self,
        user_prompt: str,
        target_length: int = 600,
        language: str = None
    ) -> Optional[CompressedPrompt]:
        """
        压缩用户 prompt

        Args:
            user_prompt: 原始用户指令
            target_length: 压缩目标长度（字符数）
            language: 语言代码（可选，自动检测）

        Returns:
            CompressedPrompt 包含压缩结果，或 None（如果压缩失败）
        """
        # 检测语言
        lang = language or detect_language(user_prompt)

        # 构建压缩 Prompt
        compress_template = self.prompts.get_template("PROMPT_COMPRESS", lang)
        compress_prompt = compress_template.format(
            user_prompt=user_prompt,
            target_length=target_length
        )

        try:
            # 定义响应 Schema
            from pydantic import BaseModel, Field
            from typing import List, Dict, Any

            class CompressResult(BaseModel):
                """压缩结果 Schema"""
                core_topic: str = Field(default="", description="核心主题")
                key_requirements: List[str] = Field(default_factory=list, description="关键需求")
                constraints: Dict[str, Any] = Field(default_factory=dict, description="约束条件")
                must_include: List[str] = Field(default_factory=list, description="必须包含的信息")
                compressed_prompt: str = Field(description="压缩后的指令")

            # 调用 LLM（低温度确保稳定输出）
            result = self.llm.call_with_schema(
                compress_prompt,
                CompressResult,
                temperature=0.2,
                max_tokens=2048
            )

            # 构建 CompressedPrompt 对象
            compressed = CompressedPrompt(
                original_prompt=user_prompt,
                compressed_prompt=result.compressed_prompt,
                core_topic=result.core_topic,
                key_requirements=result.key_requirements,
                constraints=result.constraints,
                must_include=result.must_include
            )

            if self.enable_logging:
                original_len = len(user_prompt)
                compressed_len = len(result.compressed_prompt)
                ratio = compressed.compression_ratio
                print(f"    Original: {original_len} chars")
                print(f"    Compressed: {compressed_len} chars")
                print(f"    Compression ratio: {ratio:.1%}")
                if result.key_requirements:
                    print(f"    Key requirements: {len(result.key_requirements)} items")
                if result.must_include:
                    print(f"    Must include: {len(result.must_include)} items")

            return compressed

        except Exception as e:
            if self.enable_logging:
                print(f"    Compression failed: {e}")
            return None

    def should_compress(self, user_prompt: str, threshold: int = 800) -> bool:
        """
        判断是否需要压缩

        Args:
            user_prompt: 用户指令
            threshold: 压缩阈值

        Returns:
            是否需要压缩
        """
        return len(user_prompt) > threshold