"""
Phase I — 查询风格识别

分析用户指令，提取写作需求的结构化表示：
  - 写作类型识别（论文/报告/自媒体/小说等）
  - 风格关键词与语言基调
  - 目标读者群体推断
  - 关键实体、强制约束、用户提供的事实信息提取
  - 显式章节框架检测：当用户提供了明确章节结构且置信度高于阈值时，
    触发快捷路径，跳过意图发散阶段直接建立节点树，节省计算资源。

"""
import sys
from typing import Optional

from ..schemas import StyleExtractionResult, detect_language
from ..utils.llm_client import LLMClient
from ..prompts import PromptTemplates
from ..engines import get_temperature

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout


class QueryStyleRecognition:
    """Query Style Recognition"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        enable_logging: bool = True
    ):
        """
        初始化风格提取器

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
        user_prompt: str,
        language: str = None
    ) -> Optional[StyleExtractionResult]:
        """
        执行风格提取

        Args:
            user_prompt: 用户指令
            language: 语言代码（可选，自动检测）

        Returns:
            StyleExtractionResult 或 None（如果提取失败）
        """
        if self.enable_logging:
            print_and_flush("  [Query Style Recognition] Extracting writing style...")

        # 检测语言
        lang = language or detect_language(user_prompt)

        # 构建风格提取 Prompt
        style_prompt = self.prompts.get_template("PROMPT_0_STYLE", lang).format(
            user_prompt=user_prompt
        )

        try:
            # 调用 LLM
            style_result = self.llm.call_with_schema(
                style_prompt,
                StyleExtractionResult,
                temperature=get_temperature("root_establishment", "generate")
            )

            if self.enable_logging:
                print_and_flush(f"    Writing type: {style_result.writing_type}")
                print_and_flush(f"    Style: {', '.join(style_result.style_keywords)}")
                if style_result.key_entities:
                    print_and_flush(f"    Key entities: {', '.join(style_result.key_entities[:5])}")
                if style_result.core_subject:
                    print_and_flush(f"    Core subject: {style_result.core_subject}")
                if style_result.theme_keywords:
                    print_and_flush(f"    Theme keywords: {', '.join(style_result.theme_keywords[:5])}")
                if style_result.subject_domain:
                    print_and_flush(f"    Subject domain: {style_result.subject_domain}")
                if style_result.user_provided_facts:
                    print_and_flush(f"    User facts: {', '.join(style_result.user_provided_facts[:5])}")
                if style_result.explicit_chapter_frame and style_result.explicit_chapter_frame.has_explicit_frame:
                    frame = style_result.explicit_chapter_frame
                    print_and_flush(f"    Explicit chapter frame detected!")
                    print_and_flush(f"    Confidence: {frame.confidence:.2f}")
                    print_and_flush(f"    Chapters: {[c.title for c in frame.chapters]}")

            return style_result

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"    Style extraction failed: {e}, using default")
            return None

    def get_style_context(self, style_result: StyleExtractionResult) -> str:
        """
        获取风格上下文字符串

        Args:
            style_result: 风格提取结果

        Returns:
            格式化的风格上下文
        """
        if not style_result:
            return ""

        parts = []
        parts.append(f"【写作类型】{style_result.writing_type}")

        if style_result.style_keywords:
            parts.append(f"【风格关键词】{', '.join(style_result.style_keywords)}")

        if style_result.tone_guidance:
            parts.append(f"【语气语调】{style_result.tone_guidance}")

        if style_result.language_style:
            parts.append(f"【语言风格】{style_result.language_style}")

        if style_result.target_audience:
            parts.append(f"【目标读者】{style_result.target_audience}")

        if style_result.requires_real_facts:
            parts.append(f"【事实类型】{style_result.fact_type or '需要真实事实'}")

        # 主题信息注入
        if style_result.core_subject:
            parts.append(f"【核心主题】{style_result.core_subject}")

        if style_result.theme_keywords:
            parts.append(f"【主题关键词】{', '.join(style_result.theme_keywords)}")

        if style_result.subject_domain:
            parts.append(f"【学科领域】{style_result.subject_domain}")

        # 用户事实信息注入
        if style_result.user_provided_facts:
            parts.append(f"【用户提供的事实】{'; '.join(style_result.user_provided_facts)}")

        return "\n".join(parts)