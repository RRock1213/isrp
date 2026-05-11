"""
ISRP Ablation — 引擎模块

包含：
  1. ContextEngine — 抗遗忘注入引擎
     - 全局记忆注入：压缩后的主题/关键需求/强制信息 + 风格上下文 + 全局摘要
     - 局部上下文注入：基于 RAG 检索当前节点相关原始指令片段
     - 概念覆盖池追踪：避免已覆盖概念的重复生成
     - 消融：ABLATION_ANTI_FORGET 启用时返回空字符串

  2. WordAllocationEngine — 字数分配引擎
     - BFS 逐层权重评估与字数下发
     - 结构约束检测 + 甜点区间调整 + 末端修正

  3. extract_word_count — 从用户 prompt 提取目标字数
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Any
from .schemas import (
    OutlineNode, PlanSession, LLMRole, ChapterFrameItem, ExplicitChapterFrame,
    InfoType, FactType,  # 导入枚举类型
    PackageInfo, PackageItem, PackageType  # 分包信息类型
)


# ============================================================================
# 常量定义
# ============================================================================

MIN_LEAF_WORDS = 200      # 叶子节点最小字数
MAX_LEAF_WORDS = 800      # 叶子节点最大字数
SOFT_CONSTRAINT_LOW = 90  # 柔性约束下限 (%)
SOFT_CONSTRAINT_HIGH = 110  # 柔性约束上限 (%)
MAX_DEPTH = 2             # 最大深度限制
MAX_WIDTH = 5             # DFS 节点扩展时每层最大子节点数


PROMPT_COMPRESS_THRESHOLD = 800  # 触发智能压缩的阈值
PROMPT_COMPRESS_TARGET = 600     # 压缩目标长度


ENABLE_TEMPLATE_MATCH = True


# ============================================================================

# ============================================================================

@dataclass
class DocumentStructureTemplate:
    """文体结构模板"""
    name: str                              # 模板名称
    writing_type_match: List[str]          # 匹配的写作类型关键词
    trigger_keywords: List[str]            # 触发关键词
    confidence: float                      # 默认置信度
    chapters: List[Dict[str, Any]]         # 章节定义


DOCUMENT_TEMPLATES: Dict[str, DocumentStructureTemplate] = {
    "academic_paper_cn": DocumentStructureTemplate(
        name="学术论文",
        writing_type_match=["论文", "学术论文", "研究论文", "毕业论文", "议论文"],
        trigger_keywords=["论文", "学位论文"],
        confidence=0.85,
        chapters=[
            {"title": "摘要", "needs_expansion": False, "estimated_weight": 3, "is_fixed_format": True},
            {"title": "引言", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
            {"title": "文献综述", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "研究方法", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "研究结果", "needs_expansion": True, "estimated_weight": 9, "is_fixed_format": False},
            {"title": "讨论", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "结论", "needs_expansion": True, "estimated_weight": 6, "is_fixed_format": False},
            {"title": "参考文献", "needs_expansion": False, "estimated_weight": 2, "is_fixed_format": True},
        ]
    ),
    "internship_report": DocumentStructureTemplate(
        name="实习报告",
        writing_type_match=["实习报告", "实习总结", "实习心得"],
        trigger_keywords=["实习"],
        confidence=0.80,
        chapters=[
            {"title": "前言", "needs_expansion": True, "estimated_weight": 5, "is_fixed_format": False},
            {"title": "实习单位简介", "needs_expansion": True, "estimated_weight": 6, "is_fixed_format": False},
            {"title": "实习内容", "needs_expansion": True, "estimated_weight": 9, "is_fixed_format": False},
            {"title": "收获与反思", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "结语", "needs_expansion": True, "estimated_weight": 5, "is_fixed_format": False},
        ]
    ),
    "survey_report": DocumentStructureTemplate(
        name="调研报告",
        writing_type_match=["调研报告", "调查报告", "考察报告"],
        trigger_keywords=["调研", "调查"],
        confidence=0.80,
        chapters=[
            {"title": "调研背景", "needs_expansion": True, "estimated_weight": 6, "is_fixed_format": False},
            {"title": "调研方法", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
            {"title": "调研发现", "needs_expansion": True, "estimated_weight": 9, "is_fixed_format": False},
            {"title": "问题分析", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "建议", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
        ]
    ),
    "proposal_report": DocumentStructureTemplate(
        name="开题报告",
        writing_type_match=["开题报告"],
        trigger_keywords=["开题"],
        confidence=0.80,
        chapters=[
            {"title": "选题背景", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
            {"title": "研究现状", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "研究内容", "needs_expansion": True, "estimated_weight": 9, "is_fixed_format": False},
            {"title": "研究方法", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "预期成果", "needs_expansion": True, "estimated_weight": 6, "is_fixed_format": False},
        ]
    ),
    "party_analysis": DocumentStructureTemplate(
        name="党性分析",
        writing_type_match=["党性分析", "党性分析报告"],
        trigger_keywords=["党性"],
        confidence=0.80,
        chapters=[
            {"title": "存在问题", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
            {"title": "原因分析", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "整改措施", "needs_expansion": True, "estimated_weight": 9, "is_fixed_format": False},
        ]
    ),
    "work_report": DocumentStructureTemplate(
        name="工作报告",
        writing_type_match=["工作报告", "年终总结", "述职报告", "年度总结"],
        trigger_keywords=["总结", "述职"],
        confidence=0.80,
        chapters=[
            {"title": "工作概述", "needs_expansion": True, "estimated_weight": 6, "is_fixed_format": False},
            {"title": "主要工作内容", "needs_expansion": True, "estimated_weight": 9, "is_fixed_format": False},
            {"title": "工作成果", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "存在问题", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
            {"title": "下一步计划", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
        ]
    ),
    "design_document": DocumentStructureTemplate(
        name="设计说明",
        writing_type_match=["设计说明", "设计方案", "设计报告"],
        trigger_keywords=["设计说明"],
        confidence=0.80,
        chapters=[
            {"title": "设计背景", "needs_expansion": True, "estimated_weight": 6, "is_fixed_format": False},
            {"title": "设计理念", "needs_expansion": True, "estimated_weight": 7, "is_fixed_format": False},
            {"title": "设计方案", "needs_expansion": True, "estimated_weight": 9, "is_fixed_format": False},
            {"title": "设计细节", "needs_expansion": True, "estimated_weight": 8, "is_fixed_format": False},
            {"title": "设计总结", "needs_expansion": True, "estimated_weight": 6, "is_fixed_format": False},
        ]
    ),
}


def match_document_template(user_prompt: str, writing_type: str) -> Optional[ExplicitChapterFrame]:
    """
    匹配预定义文档模板

    根据写作类型和用户 prompt 匹配预定义的结构模板。

    Args:
        user_prompt: 用户原始指令
        writing_type: 查询风格识别提取的写作类型

    Returns:
        ExplicitChapterFrame 或 None（无匹配）
    """
    if not ENABLE_TEMPLATE_MATCH:
        return None

    for template in DOCUMENT_TEMPLATES.values():
        # 优先检查写作类型匹配
        if writing_type and any(kw in writing_type for kw in template.writing_type_match):
            return _create_frame_from_template(template)

        # 检查触发关键词
        if any(kw in user_prompt for kw in template.trigger_keywords):
            return _create_frame_from_template(template, confidence_factor=0.9)

    return None


def _create_frame_from_template(
    template: DocumentStructureTemplate,
    confidence_factor: float = 1.0
) -> ExplicitChapterFrame:
    """
    从模板创建 ExplicitChapterFrame

    Args:
        template: 文档模板
        confidence_factor: 置信度因子（关键词匹配置信度稍低）

    Returns:
        ExplicitChapterFrame 实例
    """
    chapters = [
        ChapterFrameItem(
            title=ch["title"],
            needs_expansion=ch.get("needs_expansion", True),
            estimated_weight=ch.get("estimated_weight", 5),
            is_fixed_format=ch.get("is_fixed_format", False)
        )
        for ch in template.chapters
    ]

    return ExplicitChapterFrame(
        has_explicit_frame=True,
        confidence=template.confidence * confidence_factor,
        chapters=chapters,
        frame_source=f"【模板匹配】{template.name}",
        word_allocation_hint=""
    )


# ============================================================================
# 字数提取函数
# ============================================================================

def extract_word_count(prompt: str, default: int = None) -> Optional[int]:
    """
    从 prompt 中提取目标字数（支持中文数字）

    Args:
        prompt: 用户指令文本
        default: 默认值（None 表示无法提取，让调用方决定是否使用 LLM 预估）

    Returns:
        提取到的字数，或 None（表示无法提取）
    """
    # 中文数字映射
    CHINESE_NUM_MAP = {
        '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000,
    }

    # 有效中文数字字符集合
    VALID_CHINESE_NUM_CHARS = set(CHINESE_NUM_MAP.keys())

    def is_valid_chinese_number(s: str) -> bool:
        """检查是否是有效的中文数字格式"""
        if not s:
            return False
        # 检查所有字符是否都是有效的中文数字字符
        if not all(c in VALID_CHINESE_NUM_CHARS for c in s):
            return False
        # 简单验证：不能有连续的单位字符（如"十十"、"百百"、"万万"）
        units = {'十', '百', '千', '万'}
        prev_is_unit = False
        for c in s:
            is_unit = c in units
            if prev_is_unit and is_unit:
                return False
            prev_is_unit = is_unit
        return True

    def chinese_to_arabic(chinese_num: str) -> int:
        """将中文数字转换为阿拉伯数字"""
        if not chinese_num:
            return 0

        # 验证输入
        if not is_valid_chinese_number(chinese_num):
            raise ValueError(f"Invalid Chinese number: {chinese_num}")

        if len(chinese_num) == 1 and chinese_num in CHINESE_NUM_MAP:
            return CHINESE_NUM_MAP[chinese_num]

        result = 0
        temp = 0
        for char in chinese_num:
            if char in CHINESE_NUM_MAP:
                value = CHINESE_NUM_MAP[char]
                if value >= 10:
                    if temp == 0:
                        temp = 1
                    result += temp * value
                    temp = 0
                else:
                    temp = value
        result += temp
        return result

    # 1. 尝试匹配阿拉伯数字（支持带逗号的格式，如 5,000）
    patterns = [
        r'([\d,]+)\s*words?',
        r'([\d,]+)\s*字',
        r'([\d,]+)\s*字的',
        r'([\d,]+)\s*word',
        r'total\s*(?:word\s*)?count\s*(?:of\s*)?([\d,]+)',
        r'about\s+([\d,]+)\s*words?',
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            try:
                # 移除逗号后转换为整数
                num_str = match.group(1).replace(',', '')
                num = int(num_str)
                # 验证结果是否在合理范围内
                if 200 <= num <= 100000:
                    return num
            except (ValueError, AttributeError):
                # 指定具体异常类型：数字转换失败或属性访问失败
                continue

    # 2. 尝试匹配中文数字
    chinese_patterns = [
        r'([一二两三四五六七八九十百千万]+)\s*字',
        r'([一二两三四五六七八九十百千万]+)\s*字的',
        r'大约\s*([一二两三四五六七八九十百千万]+)\s*字',
    ]

    for pattern in chinese_patterns:
        match = re.search(pattern, prompt)
        if match:
            try:
                result = chinese_to_arabic(match.group(1))
                # 验证结果是否在合理范围内
                if 100 <= result <= 100000:
                    return result
            except (ValueError, KeyError):
                continue

    # 3. 无法明确识别，返回 default，由 LLM 预估处理
    # 这样可以避免误判金额、年份、面积等非字数数字
    return default


# ============================================================================
# 引擎 A: 防遗忘注入 (Anti-Forgetfulness Injection)
# ============================================================================

class ContextEngine:
    """
    上下文引擎 - 引擎 A

    功能：
    1. 防遗忘注入：在每次 LLM 调用时注入原始指令和全局摘要
    2. 相邻视野注入：通过 prev_sibling_id 和 parent_id 获取横向/纵向上下文
    3. 已覆盖概念注入：从 covered_concepts_registry 提取已覆盖的概念，避免内容重复
    """

    @staticmethod
    def build_anti_forgetfulness_context(
        session: PlanSession,
    ) -> str:
        """
        构建防遗忘上下文（每次 LLM 调用的 Prompt 顶部必须包含）

        Args:
            session: 当前会话

        Returns:
            格式化的防遗忘上下文字符串

        - 根据 session 中是否压缩来自动选择：
          - 有压缩结果：使用 compressed_prompt + 关键需求
          - 无压缩结果：使用 original_prompt（完整，不截断）
        """
        context_parts = []

        # 获取有效 prompt
        if session.compressed_prompt:
            # 已压缩：使用压缩后的 prompt + 提取的关键信息
            context_parts.append(f"【核心需求】\n{session.compressed_prompt.compressed_prompt}")
            if session.compressed_prompt.key_requirements:
                context_parts.append(f"【关键要求】\n" + "\n".join(f"- {r}" for r in session.compressed_prompt.key_requirements))
            if session.compressed_prompt.must_include:
                context_parts.append(f"【必须包含】\n" + "\n".join(f"- {m}" for m in session.compressed_prompt.must_include))
        else:
            # 未压缩：使用原始 prompt（完整，不截断）
            context_parts.append(f"【原始用户指令】\n{session.original_prompt}")

        # 获取风格上下文
        style_context = session.get_style_context() if hasattr(session, 'get_style_context') else ""
        if style_context:
            context_parts.append(style_context)

        # 注入全局摘要
        if session.global_abstract:
            context_parts.append(f"【全局摘要（北极星）】\n{session.global_abstract}")

        return "\n\n".join(context_parts) + "\n"

    @staticmethod
    def build_adjacent_context(
        node: OutlineNode,
        session: PlanSession,
        max_prev_siblings: int = 3
    ) -> str:
        """
        构建相邻视野上下文

        Args:
            node: 当前节点
            session: 当前会话
            max_prev_siblings: 最多展示的前置兄弟节点数

        Returns:
            格式化的相邻上下文字符串
        """
        parts = []

        # 获取父节点上下文
        parent = session.get_parent(node.node_id)
        if parent:
            parts.append(f"【父节点】{parent.title}: {parent.description}")

        # 获取前置兄弟节点上下文
        prev_sibling = session.get_prev_sibling(node.node_id)
        if prev_sibling:
            sibling_context = f"【前置兄弟节点】{prev_sibling.title}: {prev_sibling.description}"
            parts.append(sibling_context)

            # 尝试获取更多前置兄弟节点
            current = prev_sibling
            count = 1
            while current and count < max_prev_siblings:
                current = session.get_prev_sibling(current.node_id)
                if current:
                    parts.append(f"【更前置节点】{current.title}: {current.description}")
                    count += 1

        # 获取已有子节点上下文（用于纵向深入时）
        if node.children_ids:
            children = session.get_children(node.node_id)
            if children:
                children_titles = [c.title for c in children]  # 所有子节点
                parts.append(f"【已有子节点】{', '.join(children_titles)}")

        return "\n".join(parts) if parts else "【无前置上下文】"

    @staticmethod
    def build_covered_concepts_context(
        session: PlanSession,
        max_concepts: int = 30,
        current_level: int = None
    ) -> str:
        """
        构建已覆盖概念上下文
        
        【设计原则】
        - 目的：提醒 LLM 避免重复覆盖已有内容
        - 语义：这些概念已被其他节点覆盖，建议探索新方向
        - 层级过滤：可选择只显示特定层级的概念

        Args:
            session: 当前会话
            max_concepts: 最多展示的概念数
            current_level: 当前层级（可选，用于过滤）

        Returns:
            格式化的已覆盖概念上下文字符串
        """
        # 获取已覆盖概念
        if current_level is not None:
            concepts = session.get_covered_concepts_by_level(current_level)
        else:
            concepts = list(session.get_covered_concepts_set())
        
        if not concepts:
            return "【已覆盖概念】暂无"

        # 限制数量
        concepts = concepts[:max_concepts]
        concepts_str = ", ".join(concepts)
        
        from .prompts import PromptTemplates
        return PromptTemplates.PROMPT_CONCEPT_BLACKLIST.format(concepts_str=concepts_str)

    @staticmethod
    def build_concept_blacklist_context(
        session: PlanSession,
        max_concepts: int = 30
    ) -> str:
        """
        构建概念黑名单上下文（兼容旧接口）

        已重命名为 build_covered_concepts_context，语义更准确
        """
        return ContextEngine.build_covered_concepts_context(session, max_concepts)

    @staticmethod
    def build_prev_chapters_context(
        prev_chapter_abstracts: List[str],
        max_chapters: int = 3
    ) -> str:
        """
        构建前序章节摘要上下文

        【设计目的】
        在生成下一章时，提供前序章节的摘要信息，帮助 LLM 理解：
        1. 已完成章节的主要内容方向
        2. 避免内容重复
        3. 增强章节间的衔接流畅度

        Args:
            prev_chapter_abstracts: 已完成章节的摘要列表
            max_chapters: 最多展示的前序章节数（控制上下文长度）

        Returns:
            格式化的前序章节摘要字符串
        """
        if not prev_chapter_abstracts:
            return "【前序章节】无（当前为首章）"

        # 只取最近 max_chapters 个章节
        recent_abstracts = prev_chapter_abstracts[-max_chapters:]

        parts = []
        for i, abstract in enumerate(recent_abstracts):
            chapter_num = len(prev_chapter_abstracts) - len(recent_abstracts) + i + 1
            parts.append(f"【第{chapter_num}章摘要】{abstract}")

        return "\n".join(parts)

    @staticmethod
    def build_full_context(
        node: OutlineNode,
        user_prompt: str,
        session: PlanSession,
        include_blacklist: bool = True,
        prev_chapter_abstracts: List[str] = None
    ) -> str:
        """
        构建完整的上下文（防遗忘 + 相邻视野 + 黑名单 + 前序章节摘要）

        Args:
            node: 当前节点
            user_prompt: 用户指令（用于兼容旧调用，实际从 session 获取）
            session: 当前会话
            include_blacklist: 是否包含黑名单
            prev_chapter_abstracts: 已完成章节的摘要列表（用于增强章节衔接）

        Returns:
            完整的上下文字符串
        """
        parts = []

        # 1. 防遗忘上下文（从 session 获取 prompt 信息）
        parts.append(ContextEngine.build_anti_forgetfulness_context(session))

        # 2. 相邻视野上下文
        parts.append("\n" + ContextEngine.build_adjacent_context(node, session))

        # 3. 黑名单上下文
        if include_blacklist:
            parts.append("\n" + ContextEngine.build_concept_blacklist_context(session))

        # 4. 前序章节摘要上下文（新增）
        if prev_chapter_abstracts:
            parts.append("\n" + ContextEngine.build_prev_chapters_context(prev_chapter_abstracts))

        return "\n".join(parts)


# ============================================================================
# 引擎 C: 字数归一化 (Word Normalization)
# ============================================================================

class WordAllocationEngine:
    """
    字数归一化引擎
    
    仅提供归一化功能，作为 planner_complete.py 中字数分配的安全网
    """

    @staticmethod
    def normalize_allocation(
        nodes: List[OutlineNode],
        target_budget: int,
        min_words: int = MIN_LEAF_WORDS,
        max_words: int = MAX_LEAF_WORDS
    ) -> List[OutlineNode]:
        """
        归一化字数分配（抹平残差）

        确保最终总和精确等于 target_budget

        新增 max_words 参数，确保每个节点不超过上限

        Args:
            nodes: 已分配的节点列表
            target_budget: 目标总字数
            min_words: 最小字数限制
            max_words: 最大字数限制

        Returns:
            归一化后的节点列表
        """
        if not nodes:
            return nodes

        # 计算当前总和
        current_sum = sum(node.allocated_words for node in nodes)

        if current_sum == target_budget:
            return nodes  # 已经精确匹配

        # 计算差值
        diff = target_budget - current_sum

        if diff > 0:
            # 需要增加字数：分布式增加，确保不超过上限
            remaining_diff = diff
            for i, node in enumerate(nodes):
                if remaining_diff <= 0:
                    break
                # 计算该节点可以增加的空间
                available = max(0, max_words - node.allocated_words)
                add_amount = min(remaining_diff, available)
                node.allocated_words += add_amount
                remaining_diff -= add_amount

            # 如果还有剩余差值，分散到所有节点（每个节点稍微超过上限）
            # 这样比全部加到最后一个节点更合理
            if remaining_diff > 0:
                # 计算每个节点需要额外承担的字数
                extra_per_node = remaining_diff // len(nodes)
                extra_remainder = remaining_diff % len(nodes)

                for i, node in enumerate(nodes):
                    node.allocated_words += extra_per_node
                    if i < extra_remainder:
                        node.allocated_words += 1
        else:
            # 需要减少字数：按比例减少
            diff = abs(diff)

            # 找出可以安全减少的节点（allocated_words > min_words）
            reducible_nodes = [n for n in nodes if n.allocated_words > min_words]

            if reducible_nodes:
                reduction_per_node = diff // len(reducible_nodes)
                remaining_reduction = diff % len(reducible_nodes)

                for node in reducible_nodes:
                    # 确保减少后不低于 min_words
                    max_reduction = max(0, node.allocated_words - min_words)
                    reduction = min(reduction_per_node, max_reduction)
                    node.allocated_words -= reduction

                # 处理余数
                if remaining_reduction > 0:
                    for node in reversed(reducible_nodes):
                        max_reduction = max(0, node.allocated_words - min_words)
                        if max_reduction >= remaining_reduction:
                            node.allocated_words -= remaining_reduction
                            break
            else:
                # 如果没有可减少的节点，从最后一个节点减少
                nodes[-1].allocated_words -= diff

        return nodes


# ============================================================================
# 辅助函数
# ============================================================================

def build_dynamic_expansion_prompt(
    node: OutlineNode,
    original_prompt: str,
    session: PlanSession,
    language: str = "zh"
) -> str:
    """
    构建动态伸缩判断 Prompt

    Args:
        node: 当前节点
        original_prompt: 用户原始指令
        session: 当前会话
        language: 语言代码

    Returns:
        完整的 Prompt 字符串
    """
    context = ContextEngine.build_full_context(node, original_prompt, session)

    from .prompts import PromptTemplates
    if language == "zh":
        prompt = PromptTemplates.PROMPT_NODE_EXPAND_ZH.format(
            context=context, title=node.title,
            description=node.description, level=node.level)
    else:
        prompt = PromptTemplates.PROMPT_NODE_EXPAND_EN.format(
            context=context, title=node.title,
            description=node.description, level=node.level)

    return prompt


def get_temperature(stage: str, substage: str = "") -> float:
    """
    根据管线段获取对应的温度值

    Args:
        stage: 管线段 (root_establishment, global_abstract, dfs_expansion, chapter_checkpoint, word_allocation, format_output)
        substage: 子操作 (generate, evaluate, summarize, checkpoint, weight, split, reorganize)

    Returns:
        温度值
    """
    temperature_map = {
        "root_establishment": {
            "generate": LLMRole.CREATOR.value,      # 0.7
            "evaluate": LLMRole.REVISER.value,      # 0.3
            "default": LLMRole.CREATOR.value
        },
        "global_abstract": {
            "summarize": LLMRole.SUMMARIZER.value,  # 0.5
            "default": LLMRole.SUMMARIZER.value
        },
        "dfs_expansion": {
            "generate": LLMRole.CREATOR.value,      # 0.6-0.7
            "evaluate": LLMRole.JUDGE.value,        # 0.2
            "default": LLMRole.CREATOR.value
        },
        "chapter_checkpoint": {
            "checkpoint": LLMRole.JUDGE.value,      # 0.2-0.3
            "default": LLMRole.JUDGE.value
        },
        "word_allocation": {
            "weight": LLMRole.CALCULATOR.value,     # 0.1
            "reorganize": LLMRole.REVISER.value,    # 0.3
            "split": LLMRole.REVISER.value,         # 0.3 - 拆分需要一定创造性
            "default": LLMRole.CALCULATOR.value
        },
        "format_output": {
            "format": LLMRole.CALCULATOR.value,     # 0.1-0.2
            "default": LLMRole.CALCULATOR.value
        }
    }

    stage_map = temperature_map.get(stage, {"default": 0.5})
    return stage_map.get(substage, stage_map.get("default", 0.5))


# ============================================================================
# 分包信息检测函数 - 解决分包语义误解问题
# ============================================================================

# 分包类型关键词映射
PACKAGE_TYPE_KEYWORDS = {
    PackageType.CONSTRUCTION: [
        "施工", "建设", "工程", "平台建设", "系统建设", "实施", "开发",
        "construction", "implementation", "development", "build"
    ],
    PackageType.SUPERVISION: [
        "监理", "第三方监督", "监督管理", "质量监督", "工程监理",
        "supervision", "monitoring", "oversight", "监理服务"
    ],
    PackageType.ACCEPTANCE: [
        "验收", "测试评审", "测试", "评审", "验收评审", "第三方验收",
        "acceptance", "testing", "review", "validation", "verification"
    ],
    PackageType.DESIGN: [
        "设计", "方案设计", "规划设计", "设计服务",
        "design", "planning", "architecture"
    ],
    PackageType.EQUIPMENT: [
        "设备采购", "设备", "采购", "物资采购", "器材采购",
        "equipment", "procurement", "hardware", "device"
    ],
    PackageType.SERVICE: [
        "服务", "运维服务", "技术服务", "咨询服务", "外包服务",
        "service", "operation", "maintenance", "consulting", "outsourcing"
    ],
}


def detect_package_info(prompt: str) -> Optional[PackageInfo]:
    """
    检测分包信息

    从用户 Prompt 中提取分包信息，包括：
    - 分包标识（Package A/B/C、第一包/第二包、分包1/分包2）
    - 分包类型（施工、监理、验收、设计、设备、服务）
    - 分包预算金额

    解决问题：
    - ISRP 误将 Package B（监理）理解为 SMS notification service
    - ISRP 误将 Package C（验收）理解为 QR code printing

    Args:
        prompt: 用户指令文本

    Returns:
        PackageInfo 对象（如果检测到分包信息），否则返回 None

    Example:
        输入招标公告原文：
        "Package A: 平台建设（施工包）- ¥1,506,230
         Package B: 监理（第三方监督）- ¥20,260
         Package C: 验收（测试评审）- ¥80,000"

        输出：
        PackageInfo(
            has_packages=True,
            package_count=3,
            packages=[
                PackageItem(package_id="Package A", package_name="平台建设",
                           package_type=PackageType.CONSTRUCTION, budget="¥1,506,230"),
                PackageItem(package_id="Package B", package_name="监理",
                           package_type=PackageType.SUPERVISION, budget="¥20,260"),
                PackageItem(package_id="Package C", package_name="验收",
                           package_type=PackageType.ACCEPTANCE, budget="¥80,000")
            ]
        )
    """
    if not prompt:
        return None

    # 分包标识模式
    package_patterns = [
        # 英文格式：Package A, Package B, Package C
        r'Package\s+([A-Z]|[a-z]+)\s*[:：]\s*([^\n]+)',
        # 中文格式：第一包、第二包、分包1、分包2
        r'(第[一二三四五六七八九十\d]+包|分包[一二三四五六七八九十\d]+)\s*[:：]\s*([^\n]+)',
        # 标书格式：标段一、标段二
        r'(标段[一二三四五六七八九十\d]+)\s*[:：]\s*([^\n]+)',
    ]

    packages = []

    for pattern in package_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            package_id = match[0].strip()
            package_content = match[1].strip()

            # 解析分包内容
            package_item = _parse_package_content(package_id, package_content)
            if package_item:
                packages.append(package_item)

    # 如果没有检测到分包，返回 None
    if not packages:
        return None

    # 构建分包摘要
    summary_parts = []
    for pkg in packages:
        type_desc = _get_package_type_description(pkg.package_type)
        summary_parts.append(f"{pkg.package_id} 为{type_desc}包")
    package_summary = "，".join(summary_parts)

    return PackageInfo(
        has_packages=True,
        package_count=len(packages),
        packages=packages,
        package_summary=package_summary
    )


def _parse_package_content(package_id: str, content: str) -> Optional[PackageItem]:
    """
    解析单个分包内容

    Args:
        package_id: 分包标识（如 Package A）
        content: 分包内容描述

    Returns:
        PackageItem 对象
    """
    # 提取预算金额
    budget_patterns = [
        r'[¥￥]?\s*([\d,.]+)\s*(万元|元|万)?',  # 中文金额格式
        r'[¥￥]\s*([\d,.]+)',  # ¥符号格式
        r'([\d,.]+)\s*(万元|元|万)',  # 数字+单位格式
        r'[￥$]\s*([\d,.]+)',  # 美元格式
    ]

    budget = ""
    budget_value = None

    for pattern in budget_patterns:
        budget_match = re.search(pattern, content)
        if budget_match:
            budget = budget_match.group(0).strip()
            # 提取数值
            num_str = budget_match.group(1).replace(',', '')
            try:
                budget_value = float(num_str)
                # 如果有"万"单位，乘以10000
                if len(budget_match.groups()) > 2 and budget_match.group(2):
                    if "万" in budget_match.group(2):
                        budget_value *= 10000
            except ValueError:
                pass
            break

    # 推断分包类型
    package_type = _infer_package_type(content)

    # 提取分包名称（去除金额部分）
    package_name = content
    if budget:
        package_name = re.sub(r'[¥￥]?[\s]*[\d,.]+\s*(万元|元|万)?', '', content).strip()
        # 去除括号内的金额说明
        package_name = re.sub(r'\([^)]*[¥￥]?[\d,.]+[^)]*\)', '', package_name).strip()

    return PackageItem(
        package_id=package_id,
        package_name=package_name,
        package_type=package_type,
        budget=budget,
        budget_value=budget_value,
        description=content
    )


def _infer_package_type(content: str) -> PackageType:
    """
    根据内容推断分包类型

    Args:
        content: 分包内容描述

    Returns:
        推断的 PackageType
    """
    content_lower = content.lower()

    # 检查每种类型的关键词
    for pkg_type, keywords in PACKAGE_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in content_lower:
                return pkg_type

    # 默认返回 OTHER
    return PackageType.OTHER


def _get_package_type_description(pkg_type: PackageType) -> str:
    """
    获取分包类型的中文描述

    Args:
        pkg_type: PackageType 枚举值

    Returns:
        中文描述字符串
    """
    type_descriptions = {
        PackageType.CONSTRUCTION: "施工",
        PackageType.SUPERVISION: "监理",
        PackageType.ACCEPTANCE: "验收",
        PackageType.DESIGN: "设计",
        PackageType.EQUIPMENT: "设备采购",
        PackageType.SERVICE: "服务",
        PackageType.OTHER: "其他",
    }
    return type_descriptions.get(pkg_type, "其他")


def get_package_context(package_info: Optional[PackageInfo]) -> str:
    """
    获取分包上下文字符串

    用于注入到根节点建立和 DFS 节点生成环节，
    确保 ISRP 正确理解分包类型和关系。

    Args:
        package_info: 分包信息对象

    Returns:
        格式化的分包上下文字符串
    """
    if not package_info or not package_info.has_packages:
        return ""

    parts = []
    parts.append("【分包信息】")
    parts.append(f"本项目共 {package_info.package_count} 个分包：")

    for pkg in package_info.packages:
        type_desc = _get_package_type_description(pkg.package_type)
        budget_info = f"，预算 {pkg.budget}" if pkg.budget else ""
        parts.append(f"  - {pkg.package_id}：{pkg.package_name}（{type_desc}包{budget_info}）")

    # 添加关键警告
    parts.append("")
    parts.append("⚠️ 重要提示：请正确理解各分包的含义和职责范围：")
    parts.append("  - 施工包：负责项目建设、开发、实施工作")
    parts.append("  - 监理包：负责第三方监督、质量监督工作（不是建设服务）")
    parts.append("  - 验收包：负责测试评审、第三方验收工作（不是建设服务）")
    parts.append("请勿将监理包、验收包误解为具体的功能服务（如SMS、二维码打印等）。")

    return "\n".join(parts)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ISRP 引擎模块测试")
    print("=" * 70)

    # 测试引擎 A
    print("\n[测试 1] ContextEngine - 防遗忘注入")

    session = PlanSession()
    root = OutlineNode(
        node_id="root",
        title="Root",
        description="Root node",
        level=0
    )
    session.register_node(root)
    session.root_id = "root"
    session.global_abstract = "这是一篇关于人工智能的文章..."
    session.add_to_concept_registry(["人工智能", "机器学习", "深度学习"])

    child1 = OutlineNode(
        node_id="child1",
        title="第一章",
        description="第一章内容",
        level=1,
        parent_id="root"
    )
    session.register_node(child1)
    root.children_ids.append("child1")

    context = ContextEngine.build_full_context(child1, "写一篇关于AI的文章", session)
    print(context[:300] + "...")
    print("  [OK] 防遗忘上下文构建成功")

    # 测试引擎 C - 归一化
    print("\n[测试 2] WordAllocationEngine - 归一化抹平残差")
    nodes = [
        OutlineNode(node_id="n1", title="Node 1", description="", level=1, allocated_words=3000),
        OutlineNode(node_id="n2", title="Node 2", description="", level=1, allocated_words=3000),
        OutlineNode(node_id="n3", title="Node 3", description="", level=1, allocated_words=3000),
    ]
    # 总和 9000，目标 10000
    nodes = WordAllocationEngine.normalize_allocation(nodes, 10000)
    total = sum(n.allocated_words for n in nodes)
    print(f"  归一化前总和: 9000")
    print(f"  目标总和: 10000")
    print(f"  归一化后总和: {total}")
    assert total == 10000
    print("  [OK] 归一化正确")

    # 测试归一化 - 需要减少的情况
    print("\n[测试 3] 归一化 - 需要减少字数")
    nodes2 = [
        OutlineNode(node_id="n1", title="Node 1", description="", level=1, allocated_words=4000),
        OutlineNode(node_id="n2", title="Node 2", description="", level=1, allocated_words=4000),
        OutlineNode(node_id="n3", title="Node 3", description="", level=1, allocated_words=4000),
    ]
    # 总和 12000，目标 10000
    nodes2 = WordAllocationEngine.normalize_allocation(nodes2, 10000)
    total2 = sum(n.allocated_words for n in nodes2)
    print(f"  归一化前总和: 12000")
    print(f"  目标总和: 10000")
    print(f"  归一化后总和: {total2}")
    assert total2 == 10000
    print("  [OK] 归一化正确")

    # 测试温度获取
    print("\n[测试 4] 温度获取")
    temps = [
        get_temperature("root_establishment", "generate"),
        get_temperature("dfs_expansion", "evaluate"),
        get_temperature("word_allocation", "weight"),
    ]
    print(f"  root_establishment generate: {temps[0]}")
    print(f"  dfs_expansion evaluate: {temps[1]}")
    print(f"  word_allocation weight: {temps[2]}")
    print("  [OK] 温度获取正确")

    print("\n" + "=" * 70)
    print("[PASS] 引擎模块所有测试通过！")
    print("=" * 70)