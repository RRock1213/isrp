"""
Phase II — 全局摘要提取

基于已建立的根节点和用户意图，生成约 L_a 字的全局摘要 A。
摘要提供全文的整体视角，在后续深层节点生成时作为全局上下文注入，
防止局部优化偏离全局最优方向。同时根据目标字数 W_t 初步确认字数预算。

"""
import sys
from ..schemas import GlobalAbstractResult
from ..utils.llm_client import LLMClient
from ..prompts import PromptTemplates
from ..engines import get_temperature

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout


class GlobalAbstract:
    """Global Abstract"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        enable_logging: bool = True
    ):
        """
        初始化摘要提取器

        Args:
            llm_client: LLM 客户端
            prompts: Prompt 模板实例
            enable_logging: 是否启用日志
        """
        self.llm = llm_client
        self.prompts = prompts
        self.enable_logging = enable_logging

    def execute(
        self,
        core_angle: str,
        core_structure: str,
        total_words: int,
        style_context: str = ""
    ) -> str:
        """
        执行全局摘要提取

        Args:
            core_angle: 核心执行角度
            core_structure: 核心执行架构
            total_words: 目标总字数
            style_context: 风格上下文

        Returns:
            全局摘要字符串
        """
        if self.enable_logging:
            print_and_flush("\n[Global Abstract] Extract Global Abstract")
            print_and_flush("=" * 60)

        # 获取语言
        lang = self.prompts._language

        # 构建 Prompt
        prompt = self.prompts.get_template("PROMPT_2_ABSTRACT", lang).format(
            style_context=style_context,
            core_angle=core_angle,
            core_structure=core_structure,
            total_words=total_words
        )

        # 调用 LLM
        result = self.llm.call_with_schema(
            prompt,
            GlobalAbstractResult,
            temperature=get_temperature("global_abstract", "summarize")
        )

        if self.enable_logging:
            abstract_preview = result.global_abstract[:200] + "..." if len(result.global_abstract) > 200 else result.global_abstract
            print_and_flush(f"  Abstract: {abstract_preview}")

        return result.global_abstract