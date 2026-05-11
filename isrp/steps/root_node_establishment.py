"""
Phase II — 根节点建立

建立大纲树的根节点，确定全文的写作方向与视角：
  - 意图发散：生成多个候选切入角度
  - 串行反思：基于 6 维评分进行生成-评估-优化迭代（最多 k 轮），
    直至根节点质量达到停止条件
  - 章节框架提取：检测用户是否提供了显式章节结构，
    触发快捷路径时跳过发散阶段，直接按用户框架预建节点
  - 创建根节点并初始化 PlanSession

"""
import sys
from typing import Tuple, Optional

from ..schemas import (
    OutlineNode, NodeStatus, generate_node_id,
    RootDivergenceResult, StyleExtractionResult, detect_language,
    ExplicitChapterFrame, FrameValidationResult, robust_json_parse
)
from ..utils.llm_client import LLMClient
from ..prompts import PromptTemplates, MAX_ITERATIONS
from ..engines import get_temperature

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout

# 导入模板库
from .template_library import get_template, detect_writing_type

# 类型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..evaluators.base import BaseEvaluator
    from ..evaluators.base import EvalType


class RootNodeEstablishment:
    """Root Node Establishment"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        evaluator: 'BaseEvaluator' = None,
        enable_logging: bool = True,
        rag_service = None
    ):
        """
        初始化 Root 确立器

        Args:
            llm_client: LLM 客户端
            prompts: Prompt 模板实例
            evaluator: 评估器实例（可选）
            enable_logging: 是否启用日志
            rag_service: RAG 服务实例
        """
        self.llm = llm_client
        self.prompts = prompts
        self.evaluator = evaluator
        self.enable_logging = enable_logging
        self.rag_service = rag_service

    def execute(
        self,
        user_prompt: str,
        total_words: int,
        style_result: StyleExtractionResult,
        session,
        sample_id: str = None
    ) -> Tuple[OutlineNode, str, str]:
        """
        执行 Root 确立

        Args:
            user_prompt: 用户指令
            total_words: 目标字数
            style_result: 风格提取结果
            session: 会话实例

        Returns:
            (root节点, 核心角度, 核心架构)
        """
        if self.enable_logging:
            print_and_flush("\n[Root Node Establishment] Intent Divergence & Root Establishment")
            print_and_flush("=" * 60)

        # 检测语言
        lang = detect_language(user_prompt)
        self.prompts._language = lang

        # 检查是否有明确章节框架
        explicit_frame = None
        if style_result and hasattr(style_result, 'explicit_chapter_frame'):
            explicit_frame = style_result.explicit_chapter_frame

        # === 第一层：Query Style Recognition 的 LLM 识别结果 ===
        if explicit_frame and explicit_frame.has_explicit_frame and explicit_frame.confidence >= 0.8:
            # 快捷路径：使用明确框架
            if self.enable_logging:
                print_and_flush(f"  [Layer 1] LLM-detected frame (confidence: {explicit_frame.confidence:.2f})")
            return self._create_root_with_validation(
                user_prompt, total_words, style_result, explicit_frame, session, lang, sample_id
            )

        # 第二层：模板匹配
        from ..engines import match_document_template
        writing_type = style_result.writing_type if style_result else ""
        template_frame = match_document_template(user_prompt, writing_type)

        if template_frame and template_frame.confidence >= 0.8:
            if self.enable_logging:
                print_and_flush(f"  [Layer 2] Template-matched frame: {template_frame.frame_source}")
            return self._create_root_with_validation(
                user_prompt, total_words, style_result, template_frame, session, lang, sample_id
            )

        # 移除第三层低置信度兜底逻辑
        # 当 confidence < 0.8 时，应该走意图发散流程让LLM自主创作
        # 强制提升置信度会导致缺少上下文的请求（如"根据以上的文章信息"）无法获得创意内容
        #
        # 原代码（已移除）：
        # if explicit_frame and 0.6 <= explicit_frame.confidence < 0.8:
        #     explicit_frame.confidence = 0.85
        #     return self._create_root_with_validation(...)

        # 原有流程：意图发散
        if self.enable_logging:
            print_and_flush("  [Fallback] No explicit frame detected, using divergence flow")
        return self._execute_divergence_flow(
            user_prompt, total_words, style_result, session, lang
        )

    def _create_root_with_validation(
        self,
        user_prompt: str,
        total_words: int,
        style_result: StyleExtractionResult,
        explicit_frame: ExplicitChapterFrame,
        session,
        lang: str,
        sample_id: str = None
    ) -> Tuple[OutlineNode, str, str]:
        """
        单次验证框架合理性，然后创建 Root

        节省 Token：跳过意图发散的 3-5 次 LLM 调用，仅保留 1 次验证

        为每个章节创建子节点，确保框架被落实执行

        为快捷路径样本创建 RAG 简化日志
        """
        if self.enable_logging:
            print_and_flush(f"  [Shortcut] Explicit frame detected (confidence: {explicit_frame.confidence:.2f})")
            print_and_flush(f"  [Shortcut] Chapters: {[c.title for c in explicit_frame.chapters]}")
            print_and_flush(f"  [Shortcut] Running single validation...")

        # 直接使用 sample_id 参数，不再访问 session 私有属性
        if self.rag_service and sample_id:
            try:
                chapters_list = [c.title for c in explicit_frame.chapters]
                self.rag_service.log_shortcut_path_sample(
                    sample_id=sample_id,
                    prompt_length=len(user_prompt),
                    confidence=explicit_frame.confidence,
                    chapters=chapters_list
                )
            except Exception as e:
                if self.enable_logging:
                    print_and_flush(f"  [RAG Log] Failed to log shortcut sample: {e}")

        # 单次 LLM 验证
        validation_result = self._validate_explicit_frame(
            user_prompt, explicit_frame, style_result, lang
        )

        # 根据验证结果调整框架
        if validation_result and not validation_result.is_valid:
            if self.enable_logging:
                print_and_flush(f"    [Validation] Issues found: {validation_result.suggestions}")
            # 使用调整后的框架
            adjusted_chapters = self._adjust_frame(explicit_frame, validation_result)
        else:
            adjusted_chapters = explicit_frame.chapters

        # 提取核心角度
        core_angle = self._extract_core_angle(user_prompt, style_result)

        # 构建架构描述
        chapter_titles = [c.title for c in adjusted_chapters if c.needs_expansion]
        core_structure = " → ".join(chapter_titles)

        # 创建 Root 节点
        root = OutlineNode(
            node_id=generate_node_id(),
            title=core_angle,
            description=core_structure,
            level=0,
            word_budget=total_words,
            status=NodeStatus.ACCEPTED,
            is_finalized=True
        )

        if self.enable_logging:
            print_and_flush(f"  [Shortcut] Root created: {root.title}")
            print_and_flush(f"  [Shortcut] Structure: {core_structure}")

        # 为每个章节创建子节点
        total_weight = sum(c.estimated_weight for c in adjusted_chapters)
        for i, chapter_item in enumerate(adjusted_chapters):
            # 计算章节字数预算（按权重比例分配）
            chapter_words = int(total_words * chapter_item.estimated_weight / total_weight) if total_weight > 0 else 500

            # 生成有意义的描述，而非空洞的"章节：xxx"
            # 传入 style_result 以注入主题关键词
            chapter_description = self._generate_chapter_description(
                chapter_item.title, root.title, i, len(adjusted_chapters), style_result
            )

            # 创建章节节点
            chapter_node = OutlineNode(
                node_id=generate_node_id(),
                title=chapter_item.title,
                description=chapter_description,
                level=1,  # 一级子节点
                word_budget=chapter_words,
                status=NodeStatus.ACCEPTED,
                is_finalized=chapter_item.is_fixed_format,  # 固定格式章节标记为已完成
                needs_vertical_expansion=not chapter_item.is_fixed_format,  # 非固定格式章节需要扩展
                parent_id=root.node_id
            )

            # 注册到 session
            session.register_node(chapter_node)
            root.children_ids.append(chapter_node.node_id)

            if self.enable_logging:
                status = "[Fixed]" if chapter_item.is_fixed_format else "[Expand]"
                print_and_flush(f"    Created chapter {i+1}: {chapter_item.title} ({chapter_words} words) {status}")

        if self.enable_logging:
            print_and_flush(f"  [Shortcut] Created {len(adjusted_chapters)} chapter nodes")
            print_and_flush(f"  [Shortcut] Saved ~2-3 LLM calls by using explicit frame")

        return root, core_angle, core_structure

    def _validate_explicit_frame(
        self,
        user_prompt: str,
        explicit_frame: ExplicitChapterFrame,
        style_result: StyleExtractionResult,
        lang: str
    ) -> Optional[FrameValidationResult]:
        """
        验证章节框架合理性
        """
        # 构建章节列表字符串
        chapters_str = "\n".join([
            f"- {c.title} (权重: {c.estimated_weight}, 需展开: {c.needs_expansion})"
            for c in explicit_frame.chapters
        ])

        writing_type = style_result.writing_type if style_result else "文章"

        prompt = self.prompts.get_template("PROMPT_1_FRAME_VALIDATE", lang).format(
            user_prompt=user_prompt,
            chapters=chapters_str,
            writing_type=writing_type
        )

        try:
            response = self.llm.call(
                prompt,
                temperature=get_temperature("root_establishment", "evaluate")
            )
            result = robust_json_parse(response, FrameValidationResult)
            return result
        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"    [Validation] Error: {e}, using original frame")
            return None

    def _adjust_frame(
        self,
        explicit_frame: ExplicitChapterFrame,
        validation_result: FrameValidationResult
    ):
        """
        根据验证结果调整框架

        智能插入新章节到正确位置
        """
        from ..schemas import ChapterFrameItem

        # 如果 LLM 返回了完整调整后的章节列表，直接使用
        if validation_result.adjusted_chapters:
            return [
                ChapterFrameItem(
                    title=title,
                    needs_expansion=title not in ['标题页', '摘要', '参考文献', '致谢', '附录'],
                    estimated_weight=5,
                    is_fixed_format=title in ['标题页', '摘要', '参考文献', '致谢', '附录']
                )
                for title in validation_result.adjusted_chapters
            ]

        adjusted_chapters = list(explicit_frame.chapters)

        # 智能插入缺失章节
        for missing in validation_result.missing_chapters:
            new_chapter = ChapterFrameItem(
                title=missing,
                needs_expansion=True,
                estimated_weight=5,
                is_fixed_format=False
            )

            # 根据章节名称确定插入位置
            insert_pos = self._get_chapter_insert_position(missing, adjusted_chapters)

            if insert_pos is not None:
                adjusted_chapters.insert(insert_pos, new_chapter)
            else:
                # 无法确定位置，追加到末尾
                adjusted_chapters.append(new_chapter)

        return adjusted_chapters

    def _get_chapter_insert_position(self, chapter_name: str, existing_chapters: list) -> Optional[int]:
        """
        根据章节名称确定插入位置

        Args:
            chapter_name: 要插入的章节名
            existing_chapters: 现有章节列表

        Returns:
            插入位置索引，None 表示追加到末尾
        """
        # 章节位置规则：在 X 之后、Y 之前
        position_rules = {
            '文献综述': ('引言', '方法'),  # 文献综述在引言之后、方法之前
            '理论基础': ('引言', '方法'),
            '研究假设': ('文献综述', '方法'),
            '研究设计': ('文献综述', '方法'),
        }

        rule = position_rules.get(chapter_name)
        if not rule:
            return None

        after_chapter, before_chapter = rule
        after_pos = -1
        before_pos = len(existing_chapters)

        # 找到"在X之后"的位置
        for i, ch in enumerate(existing_chapters):
            if after_chapter in ch.title:
                after_pos = i
            if before_chapter in ch.title:
                before_pos = i

        # 如果找到了前后章节，插入到中间
        if after_pos >= 0 and before_pos > after_pos:
            return after_pos + 1
        elif after_pos >= 0:
            # 只找到"在X之后"，插入到X后面
            return after_pos + 1
        elif before_pos < len(existing_chapters):
            # 只找到"在Y之前"，插入到Y前面
            return before_pos

        return None

    def _extract_core_angle(
        self,
        user_prompt: str,
        style_result: StyleExtractionResult
    ) -> str:
        """
        从用户 prompt 和风格结果中提取核心角度
        """
        writing_type = style_result.writing_type if style_result else "文章"

        # 提取主题（简化版，取 prompt 前 50 字符）
        topic = user_prompt[:50].replace("\n", " ").strip()
        if len(user_prompt) > 50:
            topic += "..."

        return f"{writing_type}：{topic}"

    def _execute_divergence_flow(
        self,
        user_prompt: str,
        total_words: int,
        style_result: StyleExtractionResult,
        session,
        lang: str
    ) -> Tuple[OutlineNode, str, str]:
        """
        原有流程：意图发散与串行反思
        """
        # 获取风格上下文
        style_context = session.get_style_context() if hasattr(session, 'get_style_context') else ""
        fact_type_guidance = session.get_fact_type_guidance() if hasattr(session, 'get_fact_type_guidance') else ""

        # 1.1 初始提议
        if self.enable_logging:
            print_and_flush("  [1.1] Generating initial proposal...")

        prompt = self.prompts.get_template("PROMPT_1_DIVERGENCE", lang).format(
            user_prompt=user_prompt,
            style_context=style_context,
            fact_type_guidance=fact_type_guidance
        )

        initial_result = self.llm.call_with_schema(
            prompt,
            RootDivergenceResult,
            temperature=get_temperature("root_establishment", "generate")
        )

        current_angle = initial_result.angle_name
        current_rationale = initial_result.rationale
        current_structure = initial_result.high_level_structure

        if self.enable_logging:
            print_and_flush(f"    Angle: {current_angle}")

        # 1.2 串行反思循环
        iteration = 0
        is_perfect = False

        if self.enable_logging:
            eval_info = f"[Evaluator: {type(self.evaluator).__name__}]" if self.evaluator else "[Evaluator: 硬编码]"
            print_and_flush(f"  [1.2] Reflection loop starting... {eval_info}")

        while iteration < MAX_ITERATIONS and not is_perfect:
            iteration += 1
            if self.enable_logging:
                print_and_flush(f"  [1.2.{iteration}] Reflection iteration {iteration}/{MAX_ITERATIONS}")

            # 使用评估器或硬编码评估
            if self.evaluator:
                is_perfect, current_angle, current_rationale, current_structure = self._evaluate_with_evaluator(
                    user_prompt, style_context, current_angle, current_rationale, current_structure, lang
                )
            else:
                is_perfect, current_angle, current_rationale, current_structure = self._evaluate_builtin(
                    user_prompt, style_context, current_angle, current_rationale, current_structure, lang
                )

        # 1.3 创建 Root 节点
        root = OutlineNode(
            node_id=generate_node_id(),
            title=current_angle,
            description=current_structure,
            level=0,
            word_budget=total_words,
            status=NodeStatus.ACCEPTED,
            is_finalized=True
        )

        if self.enable_logging:
            print_and_flush(f"  [1.3] Root created: {root.title}")

        return root, current_angle, current_structure

    def _evaluate_with_evaluator(
        self,
        user_prompt: str,
        style_context: str,
        current_angle: str,
        current_rationale: str,
        current_structure: str,
        lang: str
    ) -> Tuple[bool, str, str, str]:
        """使用注入的评估器进行评估"""
        from ..evaluators.base import EvalType

        refine_prompt = self.evaluator.get_refine_prompt(
            user_prompt=user_prompt,
            style_context=style_context,
            angle_name=current_angle,
            rationale=current_rationale,
            high_level_structure=current_structure,
            language=lang
        )

        try:
            response = self.llm.call_with_retry(
                refine_prompt,
                temperature=get_temperature("root_establishment", "evaluate")
            )

            # 使用评估器解析结果
            eval_result = self.evaluator.parse_result(response, EvalType.REFINE)
            avg_score = eval_result.average_score
            is_perfect = avg_score >= 9.5

            if self.enable_logging:
                print_and_flush(f"    Score: {avg_score:.2f}, Perfect: {is_perfect}")

            if is_perfect:
                return True, current_angle, current_rationale, current_structure

            # 获取深化后的内容
            data = eval_result.extra
            new_angle = data.get("refined_angle_name") or current_angle
            new_rationale = data.get("refined_rationale") or current_rationale
            new_structure = data.get("refined_structure") or current_structure

            # 处理结构为列表的情况
            if isinstance(new_structure, list):
                new_structure = "\n".join(new_structure)

            return False, new_angle, new_rationale, new_structure

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"    Refinement error: {e}")
            return True, current_angle, current_rationale, current_structure  # 保持当前状态

    def _evaluate_builtin(
        self,
        user_prompt: str,
        style_context: str,
        current_angle: str,
        current_rationale: str,
        current_structure: str,
        lang: str
    ) -> Tuple[bool, str, str, str]:
        """使用内置评分进行评估"""
        from ..schemas import RootRefinementResult

        refine_prompt = self.prompts.get_template("PROMPT_1_REFINE", lang).format(
            user_prompt=user_prompt,
            style_context=style_context,
            angle_name=current_angle,
            rationale=current_rationale,
            high_level_structure=current_structure
        )

        try:
            refine_result = self.llm.call_with_schema(
                refine_prompt,
                RootRefinementResult,
                temperature=get_temperature("root_establishment", "evaluate")
            )

            avg_score = refine_result.average_score
            is_perfect = avg_score >= 9.5

            if self.enable_logging:
                print_and_flush(f"    Score: {avg_score:.2f}, Perfect: {is_perfect}")

            if is_perfect:
                return True, current_angle, current_rationale, current_structure

            new_angle = refine_result.refined_angle_name or current_angle
            new_rationale = refine_result.refined_rationale or current_rationale
            new_structure = refine_result.refined_structure or current_structure

            return False, new_angle, new_rationale, new_structure

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"    Refinement error: {e}")
            return True, current_angle, current_rationale, current_structure

    def _generate_chapter_description(
        self,
        chapter_title: str,
        root_title: str,
        chapter_index: int,
        total_chapters: int,
        style_result = None
    ) -> str:
        """
        生成有意义的章节描述

        根据章节标题和位置生成有意义的描述，
        而不是使用空洞的"章节：xxx"格式。

        注入主题关键词，使描述更具针对性。

        Args:
            chapter_title: 章节标题
            root_title: 根节点标题
            chapter_index: 章节索引（从0开始）
            total_chapters: 总章节数
            style_result: 风格提取结果（可选）

        Returns:
            有意义的描述文本
        """
        # 获取主题信息
        core_subject = ""
        theme_keywords = []
        subject_domain = ""
        writing_type = ""  # 文体类型
        user_provided_facts = []  # 用户提供的具体事实
        if style_result:
            raw_subject = getattr(style_result, 'core_subject', '') or ""
            # 过滤无效的 core_subject 占位符
            # 当 core_subject 是"未明确指定"、"需基于...推断"等占位符时，不注入描述
            if raw_subject and not any(kw in raw_subject for kw in ['未明确', '需基于', '推断', '待定', '未指定']):
                core_subject = raw_subject
            theme_keywords = getattr(style_result, 'theme_keywords', []) or []
            subject_domain = getattr(style_result, 'subject_domain', '') or ""
            writing_type = getattr(style_result, 'writing_type', '') or ""
            user_provided_facts = getattr(style_result, 'user_provided_facts', []) or []

        # 根据文体类型选择描述风格
        # 仅对小说/故事类使用叙事风格
        is_fiction = writing_type == '小说' or '故事' in writing_type or '架空' in writing_type

        # 根据章节类型选择相关的事实信息
        # 不再简单地取前3条，而是根据章节关键词匹配相关事实
        relevant_facts = []
        if user_provided_facts:
            # 章节关键词到事实关键词的映射
            chapter_fact_mapping = {
                "简介": ["地点", "学校", "单位", "公司", "机构"],
                "背景": ["时间", "背景", "原因", "目的"],
                "过程": ["过程", "内容", "活动", "工作", "任务"],
                "内容": ["内容", "活动", "工作", "任务", "具体"],
                "收获": ["收获", "成果", "成绩", "效果"],
                "反思": ["反思", "问题", "不足", "改进"],
                "总结": ["总结", "成果", "效果", "时长"],
                "结语": ["总结", "成果", "展望"],
            }

            # 匹配章节相关的事实
            matched_fact_keywords = []
            for chapter_keyword, fact_keywords in chapter_fact_mapping.items():
                if chapter_keyword in chapter_title:
                    matched_fact_keywords.extend(fact_keywords)

            # 如果没有匹配到，使用更精确的默认关键词
            if not matched_fact_keywords:
                matched_fact_keywords = ["时间", "地点"]

            # 筛选相关事实（包含匹配关键词的事实）
            for fact in user_provided_facts:
                for kw in matched_fact_keywords:
                    if kw in fact:
                        if fact not in relevant_facts:
                            relevant_facts.append(fact)
                        break

            # 如果没有匹配的事实，不强制注入（避免所有章节内容重复）
            # 移除强制注入前3条的逻辑，这会导致所有章节描述相同的事实内容

        # 过滤无效事实信息
        invalid_patterns = [
            "字数", "字数要求", "字", "要求", "写作", "撰写", "文章",
            "目标", "任务", "篇幅", "大约", "左右"
        ]
        filtered_facts = []
        for fact in relevant_facts:
            # 跳过纯字数或无意义信息
            is_invalid = False
            fact_lower = fact.lower()
            for pattern in invalid_patterns:
                # 检查是否以关键词开头，或包含"字数：xxx字"模式
                if fact_lower.startswith(pattern) or pattern in fact_lower:
                    # 检查是否只包含字数信息（长度短且包含数字）
                    if "字" in fact and (len(fact) < 20 or any(c.isdigit() for c in fact)):
                        is_invalid = True
                        break
            if not is_invalid and len(fact) > 5:
                filtered_facts.append(fact)

        # 构建事实信息字符串 - 融入内容而非标注元数据
        # 进一步减少模板化语言
        facts_str = ""
        if filtered_facts:
            # 将事实信息自然融入描述，避免"重点涵盖xxx"的模板化表述
            # 例如："聚焦xxx问题，分析xxx案例"
            if len(filtered_facts) >= 2:
                facts_str = f"，结合{'与'.join(filtered_facts[:2])}进行分析"
            elif len(filtered_facts) == 1:
                facts_str = f"，围绕{filtered_facts[0]}展开"

        # 小说文体的特殊处理
        if is_fiction:
            # 使用叙事风格的描述模板
            narrative_templates = {
                "引子": "本章作为故事开篇，引出核心冲突与悬念，介绍关键人物与故事背景。",
                "开端": "本章开启故事主线，介绍主人公与核心矛盾，为后续发展埋下伏笔。",
                "发展": "本章推进故事发展，深化人物关系，展开核心矛盾。",
                "高潮": "本章进入故事高潮，矛盾冲突达到顶点，揭示关键真相。",
                "转折": "本章迎来转折，故事走向发生变化，人物命运出现转机。",
                "结局": "本章收束全文，解决核心矛盾，完成人物命运交代。",
                "尾声": "本章作为尾声，交代后续发展，留下回味空间。",
                "余烬": "本章作为故事收束，交代后续发展，完成人物命运交代。",
                "阴翳": "本章描写阴翳笼罩下的场景与氛围，展开神秘情节，揭示隐藏秘密。",
                "礼赞": "本章描绘礼赞仪式的场景与人物行动，展现信仰与冲突的交织。",
                "诅咒信": "本章展现诅咒信引发的危机与悬念，推动情节走向高潮。",
            }
            for keyword, template in narrative_templates.items():
                if keyword in chapter_title:
                    return template

            # 小说章节通用描述
            if "第" in chapter_title and "章" in chapter_title:
                return f"本章推进故事发展，展开{chapter_title}的核心情节，推动人物命运走向。"

            # 小说小节描述
            if any(kw in chapter_title for kw in ['场景', '氛围', '人物', '情节', '行动', '冲突', '悬念']):
                return f"本节描写{chapter_title}的具体场景与人物行动，推动故事情节发展。"

        # 优先使用 template_library 获取模板
        # 确定主题
        subject = core_subject if core_subject else root_title

        # 检测文体类型
        detected_type = writing_type
        if not detected_type and user_provided_facts:
            all_text = " ".join(user_provided_facts)
            detected_type = detect_writing_type(all_text)

        # 使用 template_library 获取模板
        template_desc = get_template(
            chapter_title=chapter_title,
            subject=subject,
            writing_type=detected_type,
            user_facts=user_provided_facts
        )

        # 如果模板已完整填充（不是默认模板），直接返回
        default_template = f"阐述{subject}的相关内容，展开详细分析与论述。"
        if template_desc != default_template and "{" not in template_desc:
            # 如果有匹配的事实，融入描述
            if facts_str:
                return template_desc.rstrip("。") + facts_str + "。"
            return template_desc

        # 使用 {subject} 占位符的描述模板
        # 减少"本部分..."模式化开头，使描述更自然
        description_templates = {
            # 开头类 - 自然化语言
            "引言": "开篇系统介绍{subject}的研究背景、目的与全文结构，交代研究缘起与核心关切。",
            "前言": "概述{subject}的写作背景、目的与全文框架，引出核心议题。",
            "绪论": "系统阐述{subject}的研究背景、意义、目的与方法，构建全文的理论框架与研究路径。",
            "摘要": "概括{subject}的核心内容，包括研究问题、主要方法、关键发现与核心结论，语言精炼、信息完整。",

            # 背景类 - 自然化语言
            "背景": "阐述{subject}的发展背景，介绍历史起源、演进脉络与当前状况，结合具体数据或案例说明。",
            "研究背景": "系统阐述{subject}的研究背景，说明问题产生的现实动因与理论依据，引用权威数据或典型案例支撑。",
            "选题背景": "说明{subject}选题的背景，交代选题来源、现实意义与研究必要性。",

            # 综述类 - 自然化语言
            "文献综述": "系统梳理{subject}领域的国内外研究成果，分析现有研究的贡献与不足，确立研究的理论起点。",
            "研究现状": "综述{subject}的国内外研究现状，梳理主要学术观点与研究进展，指出研究空白与本研究切入点。",

            # 方法类 - 自然化语言
            "方法": "详细阐述{subject}的研究方法与技术路线，说明研究设计、数据来源与分析工具。",
            "研究方法": "系统介绍{subject}的研究设计与分析方法，说明样本选取、数据收集与分析技术。",
            "技术路线": "明确{subject}研究的技术路径与实施步骤，说明研究过程的关键节点与操作流程。",

            # 内容类
            "主要内容": "详细阐述{subject}的核心内容，展开深入分析与论述，明确关键要点与论述逻辑。",
            "研究内容": "系统展开{subject}的研究内容，分析关键问题与核心观点，构建研究的主体框架。",

            # 结果类
            "结果": "本部分呈现{subject}研究的主要发现与数据分析结果，客观展示研究产出，使用图表辅助说明。",
            "研究结果": "系统呈现{subject}的研究结果，包括数据发现、统计分析与关键结论。",
            "研究发现": "总结{subject}研究的核心发现，提炼具有理论价值与实践意义的结论，结合具体证据支撑。",

            # 讨论类
            "讨论": "对{subject}的研究结果进行深入讨论，阐释其理论内涵与实践启示。",
            "分析与讨论": "对{subject}的核心问题进行深入分析与讨论，揭示内在逻辑与理论对话。",

            # 结论类
            "结论": "总结{subject}研究的核心观点与贡献，指出研究局限与未来方向。",
            "总结": "对{subject}研究进行系统总结，归纳核心观点与主要结论。",
            "结语": "作为结语，总结{subject}研究的要点，展望未来发展趋势或研究方向。",
            "研究结论": "系统总结{subject}的研究结论，阐明理论贡献与实践价值。",

            # 参考文献类
            "参考文献": "列出研究过程中引用的所有文献资料，按学术规范格式编排。",

            # 建议类
            "建议": "针对{subject}提出针对性建议，为实践应用与政策制定提供参考。",
            "对策建议": "针对{subject}提出具体对策与建议，明确实施路径与关键措施。",
            "措施": "针对{subject}制定具体改进措施，确保问题得到有效解决。",

            # 目标类
            "目标": "明确{subject}研究的目标与预期成果，为后续工作指明方向。",
            "研究目标": "本部分阐明{subject}研究的核心目标与具体任务，明确研究的价值定位。",

            # 意义类
            "意义": "阐述{subject}研究的理论意义与实践价值，论证研究的重要性与必要性。",
            "目的与意义": "本部分阐明{subject}研究的目的与意义，说明研究的价值与必要性。",

            # 活动类
            "活动计划": "本部分详细规划活动安排，明确活动目标、时间安排与实施步骤。",
            "活动内容": "本部分具体描述活动内容，说明活动形式、主要环节与执行细节。",
            "主要成果": "总结主要成果，说明量化成果、质性成效与典型案例。",

            # 创新类
            "创新点": "突出创新之处，说明理论、方法或应用层面的独特贡献。",
            "可行性分析": "分析可行性，评估技术可行性、经济可行性与风险评估。",

            # 报告类
            "主要工作及成效": "系统总结工作与成效，说明工作任务完成情况与主要成绩。",
            "存在问题及原因分析": "深入分析问题与原因，揭示问题表现、影响程度与根本原因。",
            "整改措施": "提出整改措施，明确整改目标、具体措施与责任主体。",
            "下一步工作": "规划下一步工作，明确工作重点、目标任务与实施路径。",

            # 分析类 - 使用领域特定分析维度
            # "分析"模板在匹配时会调用 _get_domain_specific_analysis_dims 获取特定维度
            "分析": "深入分析{subject}{domain_dims}，揭示内在逻辑与实践启示。",
            "数据分析": "进行{subject}数据分析，说明数据来源、统计方法与核心数据结论。",
            "问题分析": "剖析{subject}问题，揭示问题具体表现、影响范围与成因。",
            "原因分析": "分析{subject}原因，梳理直接原因、深层原因与根本原因。",

            # 研究类
            "案例研究": "进行{subject}案例研究，说明案例选取依据与分析框架应用。",
            "实证研究": "本部分开展{subject}实证研究，说明研究假设、变量设计与数据检验方法。",
            "理论研究": "进行{subject}理论研究，界定核心概念并构建理论框架。",

            # 理论类
            "理论框架": "构建{subject}理论框架，界定核心概念与变量关系。",
            "概念界定": "本部分界定{subject}核心概念，明确概念内涵外延与相关概念的区别。",
            "理论基础": "阐述{subject}理论基础，说明所依据的基础理论及其指导意义。",

            # 设计类
            "系统设计": "阐述{subject}系统设计，说明系统架构、功能模块与关键技术选型。",
            "方案设计": "提出{subject}方案设计，说明设计思路、方案内容与技术路线图。",
            "模型设计": "构建{subject}分析模型，说明模型假设、变量定义与参数设置。",

            # 实践类
            "实践探索": "总结{subject}实践经验，说明实践背景、探索过程与经验教训。",
            "应用实践": "展示{subject}应用实践，说明应用场景、实施过程与效果评估。",
            "实践应用": "阐述{subject}实践应用，说明应用领域、操作方法与推广价值。",

            # 案例类
            "案例分析": "深入剖析案例，揭示案例背景、分析过程与关键发现。",

            # 比较类 - 添加subject占位符和对比维度
            "比较分析": "对{subject}进行比较分析，明确比较维度、异同点与规律总结。",
            "对比分析": "对{subject}进行对比分析，从管理模式、核心策略、运作机制等方面展开，揭示关键差异与成功经验。",
            "异同分析": "分析{subject}的异同点，明确相同之处、差异之处及其成因。",

            # 发展类
            "发展趋势": "分析{subject}发展趋势，介绍历史演变、当前特征与未来方向。",
            "发展历程": "梳理{subject}发展历程，介绍起源阶段、发展阶段与成熟阶段。",
            "前景": "本部分展望{subject}发展前景，说明发展机遇、面临挑战与未来趋势。",
            "展望": "进行{subject}前景展望，说明技术演进趋势与应用拓展方向。",
            "机遇": "分析{subject}机遇与挑战，说明外部机遇、内部优势与潜在风险。",
            "挑战与对策": "剖析{subject}挑战与对策，说明挑战表现与应对策略。",
            "面临的挑战": "剖析{subject}面临挑战，说明挑战类型与解决思路。",
            "关键技术与架构": "阐述{subject}关键技术，说明核心技术原理与技术架构。",
            "应用场景与产业": "阐述{subject}应用场景，说明典型应用领域与产业融合模式。",

            # 对策类
            "对策": "针对{subject}提出针对性的解决策略，明确实施路径与关键措施。",
            "对策建议": "结合{subject}的问题分析，提出具有可操作性的对策建议。",

            # 概述类
            "概述": "对{subject}进行总体介绍，明确研究范围与主要内容。",
            "总体概述": "对{subject}进行宏观介绍，构建整体认知框架。",

            # 党性分析类（党政公文风格）
            "党性分析": "对照党章党规，从政治信仰、党员意识等方面进行{subject}自我剖析。",
            "党性认识": "对照党章党规，从政治信仰、党员意识、担当作为等方面进行{subject}自我评价。",
            "自我评价": "对照党员标准，客观评定{subject}在党性修养方面的总体表现与不足。",
            "存在问题": "坚持问题导向，深入查摆{subject}在理论学习、政治素质、担当作为等方面存在的具体问题。",
            "原因分析": "从理想信念、宗旨意识、党性修养等方面剖析{subject}问题产生的深层原因。",
            "整改措施": "针对{subject}查摆出的问题，制定切实可行的整改清单与具体举措。",
            "努力方向": "明确{subject}今后的努力方向，提出强化理论武装、锤炼党性修养等具体目标。",
            "整改": "针对{subject}存在问题制定整改措施，明确时间表与责任人。",

            # 实习/社会实践类
            "实习单位简介": "介绍{subject}实习单位基本情况，说明单位性质、规模、主营业务与行业地位。",
            "实践单位简介": "介绍{subject}实践单位基本情况，说明单位背景、规模特点与主要职能。",
            "实习内容": "本部分详细记录{subject}实习工作内容，说明岗位职责、参与项目与具体任务。",
            "实习过程": "按阶段记录{subject}实习过程，说明各阶段时间、主要任务与完成情况。",
            "实践内容": "本部分详细描述{subject}实践内容与过程，说明实践目标、具体活动与阶段性成果。",
            "收获与体会": "总结{subject}实习收获，说明专业技能提升、职业认知转变与个人成长。",
            "收获与反思": "总结{subject}收获与反思，说明能力提升、认知变化与改进方向。",
            "问题与反思": "反思{subject}问题与不足，说明遇到的问题、原因分析与经验教训。",
            "实习总结": "系统总结{subject}实习经历，说明主要收获、能力变化与职业规划。",
            "实践总结": "总结{subject}实践经历，说明实践成果、个人成长与社会认知。",

            # 自媒体/新媒体类（活泼生动风格）
            "开篇": "设计{subject}吸引人的开篇，设置悬念、营造画面感、引导读者兴趣。",
            "悬念": "本部分设置{subject}悬念，提出问题、揭示矛盾、激发好奇心。",
            "热议": "展示{subject}热议观点，呈现典型观点与多元立场对比。",
            "舆论": "分析{subject}舆论反响，揭示舆论焦点与不同声音。",
            "心声": "本部分描写{subject}心理活动，表达真实情感与内心矛盾。",
            "心理": "深入描写{subject}心理，展现心理变化过程与情感转折点。",
            "共鸣": "设计{subject}共鸣内容，挖掘普遍情感与共同经历。",
            "互动": "设计{subject}互动内容，设置讨论话题与读者参与方式。",
            "科普": "进行{subject}科普说明，将专业知识通俗化、实用化。",
            "干货": "提供{subject}实用干货，说明具体方法、操作步骤与实用技巧。",
            "结尾": "设计引发讨论的{subject}结尾，重申核心观点、升华话题。",
            "升华": "本部分升华{subject}全文观点，提炼核心价值、启发读者思考。",
            "引言": "作为{subject}开篇导引，引入热点背景、提出核心问题。",
            "实践路径": "梳理{subject}实践路径，介绍典型做法与成功案例。",
            "多元": "多维度分析{subject}，呈现不同视角与多元观点。",
            "生存": "剖析{subject}生存困境，揭示困境表现与应对策略。",
            "困境": "分析{subject}困境表现，说明困境类型与突破思路。",
            "两难": "探讨{subject}两难处境，揭示矛盾焦点与平衡之道。",
            "缺位": "揭示{subject}制度缺位，说明缺位表现与完善建议。",
            "期盼": "梳理{subject}各方期盼，介绍利益相关方诉求与政策建议。",
            "建议": "提出{subject}具体建议，说明问题针对性与措施可行性。",
            "结语": "作为{subject}结语，回顾核心观点、升华价值。",
            "现实": "揭示{subject}现实张力，说明理想与现实差距及解决思路。",
            "拷问": "进行{subject}深度拷问，追问核心问题、引发读者思考。",

            # 调研报告类（政府公文风格）
            "调研背景": "说明{subject}调研背景，交代调研动因、现实需求与政策依据。",
            "调研方法": "说明{subject}调研方法，介绍调研时间、对象、方式与样本情况。",
            "调研情况": "概述{subject}调研情况，说明调研范围、对象特征与数据来源。",
            "现状分析": "分析{subject}现状，介绍基本情况、主要特点与存在问题。",
            "问题分析": "深入分析{subject}问题，揭示问题类型、表现形式与深层原因。",
            "对策建议": "提出{subject}对策建议，说明总体思路、具体措施与实施路径。",
            "改进建议": "提出{subject}改进建议，明确问题导向与具体方案。",
            "调研结论": "总结{subject}调研结论，归纳主要发现与核心结论。",

            # 小说创作类
            "第一章": "本章作为{subject}故事开端，介绍主要人物与背景设定，制造悬念引出冲突。",
            "第二章": "本章推进{subject}故事发展，展开人物关系与矛盾冲突。",
            "第三章": "本章进入{subject}高潮部分，矛盾冲突达到顶峰。",
            "第四章": "本章走向{subject}转折，冲突开始化解或出现新变化。",
            "第五章": "本章作为{subject}结局，总结故事脉络，完成人物命运交代。",

            # 议论文类
            "分论点": "展开{subject}论证，从具体角度阐述核心观点。",
            "论证": "运用论据支撑{subject}观点，结合实例或数据进行分析。",
            "论据": "提供{subject}论据支撑，引用名言、数据或案例进行论证。",
            "驳论": "对{subject}反面观点进行驳斥，增强论证的说服力。",
            "辩证": "进行{subject}辩证分析，论述观点的内在逻辑与相互关系。",
        }

        # 1. 尝试精确匹配模板
        for keyword, template in description_templates.items():
            if keyword in chapter_title:
                # 使用 {subject} 占位符动态注入主题
                if "{subject}" in template:
                    # 优先使用章节标题中的关键词作为subject
                    # 避免多个章节使用相同的core_subject导致描述重复
                    chapter_subject = self._extract_chapter_subject(chapter_title, core_subject)

                    # 支持 {domain_dims} 占位符
                    if "{domain_dims}" in template:
                        domain_dims = self._get_domain_specific_analysis_dims(chapter_title, chapter_subject)
                        base_desc = template.format(subject=chapter_subject, domain_dims=domain_dims)
                    else:
                        base_desc = template.format(subject=chapter_subject)

                    # 注入用户事实信息
                    if facts_str:
                        return base_desc + facts_str
                    return base_desc
                else:
                    # 无占位符的模板直接返回，也注入事实信息
                    if facts_str:
                        return template + facts_str
                    return template

        # 2. 尝试从标题提取关键词生成描述
        # 避免使用"探讨xxx相关内容"等空洞模板化表述
        title_keywords = self._extract_title_keywords(chapter_title)
        if title_keywords:
            # 小说文体使用叙事风格描述
            if is_fiction:
                return f"本节{title_keywords}，通过场景描写与人物行动推进故事情节发展。"
            # 直接返回具体指导，不再使用空洞的关键词映射
            if core_subject and core_subject not in chapter_title:
                base_desc = f"本部分围绕{core_subject}，{title_keywords}，需结合具体案例或数据进行阐述。"
                if facts_str:
                    return base_desc + facts_str
                return base_desc
            base_desc = f"本部分{title_keywords}，确保内容详实、论证充分。"
            if facts_str:
                return base_desc + facts_str
            return base_desc

        # 3. 从章节标题提取特异性关键词（避免重复）
        # 不再使用全局 theme_keywords[:3]，而是从标题中提取独特关键词
        specific_keywords = self._extract_specific_keywords_from_title(
            chapter_title, chapter_index, total_chapters, theme_keywords
        )
        if specific_keywords:
            # 小说文体使用叙事风格描述
            if is_fiction:
                return f"本节描写{chapter_title}的具体场景，{specific_keywords}，推动故事走向。"
            # 生成具体指导性描述，避免模板化表述
            if core_subject and core_subject not in chapter_title:
                base_desc = f"本部分聚焦{core_subject}主题下的{chapter_title}，具体分析{specific_keywords}，需提供实例支撑。"
                if facts_str:
                    return base_desc + facts_str
                return base_desc
            base_desc = f"针对{chapter_title}进行详细展开，重点分析{specific_keywords}，确保内容具体。"
            if facts_str:
                return base_desc + facts_str
            return base_desc

        # 4. 仅使用核心主题生成描述
        # 小说文体使用叙事风格描述
        if is_fiction:
            if core_subject:
                return f"本节围绕{core_subject}展开故事，描写{chapter_title}的场景与人物行动。"
            return f"本节描写{chapter_title}的场景与人物行动，推进故事情节发展。"

        if core_subject and core_subject not in chapter_title:
            base_desc = f"详细阐述{chapter_title}在{core_subject}研究中的具体内容，需明确观点并提供论据支撑。"
            if facts_str:
                return base_desc + facts_str
            return base_desc

        # 5. 根据位置生成描述（兜底逻辑）
        # 小说文体使用叙事风格
        if is_fiction:
            position = chapter_index + 1
            if position == 1:
                return f"本节作为故事开篇，描绘{chapter_title}的场景与氛围，引出后续情节。"
            elif position == total_chapters:
                return f"本节作为故事收束，完成{chapter_title}的命运交代，留下回味空间。"
            else:
                return f"本节推进故事发展，描写{chapter_title}的具体情节与人物行动。"

        # 位置兜底逻辑，避免空洞表述
        position = chapter_index + 1
        if position == 1:
            base_desc = f"作为开篇导引，介绍{chapter_title}的背景与核心内容，引出后续论述主线。"
            if facts_str:
                return base_desc + facts_str
            return base_desc
        elif position == 2:
            base_desc = f"本部分承接开篇，详细展开{chapter_title}的具体内容，明确分析维度与关键要点。"
            if facts_str:
                return base_desc + facts_str
            return base_desc
        elif position == total_chapters - 1:
            base_desc = f"对{chapter_title}进行系统分析与归纳，总结核心发现，为结论做铺垫。"
            if facts_str:
                return base_desc + facts_str
            return base_desc
        elif position == total_chapters:
            base_desc = f"总结全文核心观点，归纳{chapter_title}的主要结论与启示。"
            if facts_str:
                return base_desc + facts_str
            return base_desc
        else:
            # 中间章节：根据章节标题生成具体描述
            base_desc = f"深入分析{chapter_title}的具体内容，结合实例进行详细阐述，揭示核心观点。"
            if facts_str:
                return base_desc + facts_str
            return base_desc

    def _extract_specific_keywords_from_title(
        self,
        chapter_title: str,
        chapter_index: int,
        total_chapters: int,
        theme_keywords: list
    ) -> str:
        """
        从章节标题提取特异性关键词

        根据章节标题和位置提取独特关键词，避免所有章节使用相同的 theme_keywords。
        这解决了之前"多个章节描述雷同"的核心问题。

        Args:
            chapter_title: 章节标题
            chapter_index: 章节索引
            total_chapters: 总章节数
            theme_keywords: 全局主题关键词列表

        Returns:
            提取的特异性关键词描述
        """
        import re

        # 1. 从标题中提取名词性关键词（中文分词简化版）
        # 匹配常见名词模式
        noun_patterns = [
            r'([\u4e00-\u9fa5]{2,4}(?:研究|分析|设计|建设|发展|改革|创新|管理|服务|体系|机制|模式|路径|策略|对策|措施|问题|困境|挑战|现状|趋势|案例|实证|理论|方法|技术|应用|实践|评估|评价|优化|改进))',
            r'([\u4e00-\u9fa5]{2,4}(?:化|型|性|度|率|力|观|念|识|思|想|路|道|方|向|标|准|则|规|律|系|统|体|制|度))',
            r'([\u4e00-\u9fa5]{3,6}(?:时期|阶段|历程|演进|变迁|背景|现状|概况|简介|简介))',
        ]

        for pattern in noun_patterns:
            match = re.search(pattern, chapter_title)
            if match:
                return match.group(1)

        # 2. 根据章节位置从 theme_keywords 选择不同关键词
        # 不同章节使用不同的关键词组合，避免重复
        if theme_keywords and len(theme_keywords) >= 1:
            # 使用章节索引作为偏移量，循环选择关键词
            offset = chapter_index % len(theme_keywords)
            selected_keyword = theme_keywords[offset]
            # 如果有多个关键词，组合使用
            if len(theme_keywords) >= 2:
                next_offset = (chapter_index + 1) % len(theme_keywords)
                if next_offset != offset:
                    return f"{selected_keyword}与{theme_keywords[next_offset]}"
            return selected_keyword

        # 3. 从标题中提取有意义的词汇（去除常见前缀）
        # 去除"第X章"、"一、"等前缀
        clean_title = chapter_title
        clean_title = re.sub(r'第[一二三四五六七八九十\d]+章[:：]?\s*', '', clean_title)
        clean_title = re.sub(r'[一二三四五六七八九十\d]+[、:：]\s*', '', clean_title)
        clean_title = re.sub(r'[\(（].*?[\)）]', '', clean_title)  # 去除括号内容

        # 提取核心词汇（2-6个汉字）
        if len(clean_title) >= 2:
            # 如果标题较短，直接使用
            if len(clean_title) <= 6:
                return clean_title
            # 否则提取前半部分
            return clean_title[:6]

        return ""

    def _get_analysis_method_for_domain(self, subject_domain: str, writing_type: str = "") -> str:
        """
        根据领域和文体类型返回合适的分析方法

        不同领域适用不同的分析框架：
        - 商业/管理领域：SWOT/PEST/波特五力等
        - 医疗/健康领域：循证医学方法、临床分析框架
        - 教育/学术领域：文献综述法、案例分析法
        - 影视/艺术领域：文本分析、符号学分析、视听语言分析
        - 工程/技术领域：技术评估、可行性分析

        Args:
            subject_domain: 主题领域
            writing_type: 文体类型

        Returns:
            推荐的分析方法名称，如果没有特定推荐则返回空字符串
        """
        domain_lower = subject_domain.lower() if subject_domain else ""
        writing_lower = writing_type.lower() if writing_type else ""

        # 商业/管理领域 - 适用SWOT/PEST/波特五力
        business_keywords = ['商业', '管理', '企业', '市场', '营销', '经济', '金融', '战略', '经营']
        if any(kw in domain_lower for kw in business_keywords):
            return "SWOT分析、PEST分析或波特五力模型"

        # 医疗/健康领域 - 不适用SWOT等
        medical_keywords = ['医疗', '医学', '健康', '护理', '临床', '医院', '疾控', '卫生']
        if any(kw in domain_lower for kw in medical_keywords):
            return "循证分析、病因分析或临床评估框架"

        # 教育/教学领域
        education_keywords = ['教育', '教学', '学校', '课程', '学生', '教师', '培训']
        if any(kw in domain_lower for kw in education_keywords):
            return "案例分析法、比较研究法或问卷调查分析"

        # 影视/艺术领域
        art_keywords = ['影视', '电影', '艺术', '动画', '鉴赏', '视听', '导演', '编剧']
        if any(kw in domain_lower for kw in art_keywords) or any(kw in writing_lower for kw in art_keywords):
            return "文本分析、视听语言分析或符号学分析"

        # 历史/文化领域
        history_keywords = ['历史', '文化', '文物', '考古', '民俗', '民族']
        if any(kw in domain_lower for kw in history_keywords):
            return "历史文献分析、比较分析法或田野调查法"

        # 工程/技术领域
        tech_keywords = ['工程', '技术', '系统', '软件', '信息', '数据', '算法']
        if any(kw in domain_lower for kw in tech_keywords):
            return "技术评估、可行性分析或系统分析方法"

        # 法律/司法领域
        legal_keywords = ['法律', '司法', '法院', '案件', '法规', '权利']
        if any(kw in domain_lower for kw in legal_keywords):
            return "法律解释、案例分析或规范分析法"

        # 哲学/社会科学领域
        philosophy_keywords = ['哲学', '社会', '政治', '思想', '理论', '伦理']
        if any(kw in domain_lower for kw in philosophy_keywords):
            return "概念分析、逻辑论证或文献研究法"

        # 默认返回空，使用通用分析方法
        return ""

    def _extract_chapter_subject(self, chapter_title: str, core_subject: str) -> str:
        """
        从章节标题提取特定的章节主题

        解决问题：当多个章节匹配同一模板时，避免使用相同的core_subject导致描述重复。
        例如：
        - 第2章"安踏供应链管理模式分析" -> "安踏供应链管理模式"
        - 第3章"耐克供应链管理模式分析" -> "耐克供应链管理模式"

        Args:
            chapter_title: 章节标题
            core_subject: 全文核心主题（作为fallback）

        Returns:
            章节特定的主题描述
        """
        # 移除常见的章节后缀词
        suffixes = ['分析', '研究', '概述', '简介', '总结', '探讨', '论述', '阐述']
        title_clean = chapter_title
        for suffix in suffixes:
            if chapter_title.endswith(suffix) and len(chapter_title) > len(suffix) + 2:
                title_clean = chapter_title[:-len(suffix)]
                break

        # 如果清理后的标题比原主题更具体，使用清理后的标题
        if len(title_clean) > 2 and title_clean != core_subject:
            # 检查标题是否包含具体实体（如"安踏"、"耐克"）
            entity_keywords = ['安踏', '耐克', '华为', '小米', '苹果', '阿里巴巴', '腾讯', '京东',
                              '李宁', '特步', '鸿星尔克', '比亚迪', '宁德时代', '大疆', 'OPPO', 'vivo']
            for entity in entity_keywords:
                if entity in chapter_title:
                    return title_clean

        # 如果章节标题足够具体，直接使用
        if len(chapter_title) >= 4 and len(chapter_title) <= 20:
            # 检查是否包含关键词组合
            key_combos = ['供应链', '管理', '模式', '策略', '机制', '路径', '体系', '架构']
            if any(kw in chapter_title for kw in key_combos):
                return chapter_title

        # fallback到core_subject
        return core_subject if core_subject else "本"

    def _get_domain_specific_analysis_dims(self, chapter_title: str, subject: str) -> str:
        """
        根据章节标题和主题获取领域特定的分析维度

        解决问题：通用"分析"模板使用"关键要素、运作机制"等空洞维度，
        无法为供应链管理、商业模式等特定领域提供针对性指导。

        Args:
            chapter_title: 章节标题
            subject: 主题描述

        Returns:
            领域特定的分析维度描述，如果无法识别则返回默认描述
        """
        # 领域关键词到分析维度的映射
        domain_analysis_dims = {
            # 供应链/物流领域
            '供应链': '供应商管理、库存控制、物流配送、成本优化与风险应对',
            '物流': '运输模式、仓储布局、配送效率、成本控制与服务质量',
            '仓储': '库存管理、出入库流程、空间利用与自动化程度',

            # 商业模式/管理领域
            '商业模式': '价值主张、客户细分、收入来源与成本结构',
            '管理模式': '组织架构、决策机制、激励机制与文化建设',
            '营销策略': '目标市场、渠道选择、品牌定位与促销手段',
            '品牌': '品牌定位、形象塑造、传播渠道与市场认知',
            '竞争': '竞争优势、市场定位、差异化策略与竞争态势',

            # 技术/IT领域
            '技术架构': '系统架构、技术选型、性能优化与扩展性设计',
            '系统设计': '功能模块、数据流、接口设计与安全性',
            '软件开发': '需求分析、设计模式、测试策略与部署流程',

            # 教育/培训领域
            '教学': '教学内容、教学方法、教学效果与学生反馈',
            '课程': '课程目标、课程内容、教学方法与评价体系',
            '实训': '实训目标、实训内容、操作流程与技能提升',

            # 财务/经济领域
            '财务': '收入分析、成本结构、利润率与财务风险',
            '投资': '投资策略、风险评估、收益预期与退出机制',
            '成本': '成本构成、成本控制、成本优化与效益分析',

            # 政治/党建领域
            '党性': '政治信念、纪律意识、作风表现与责任担当',
            '党建': '组织建设、思想建设、制度建设与活动开展',
            '思想': '理论学习、思想认识、观念转变与实践应用',

            # 历史/文化领域
            '历史': '历史背景、发展脉络、关键事件与历史影响',
            '文化': '文化内涵、表现形式、传承方式与现代价值',
            '思想': '核心概念、理论体系、历史演变与当代意义',
        }

        # 检查标题中是否包含领域关键词
        for domain_keyword, dims in domain_analysis_dims.items():
            if domain_keyword in chapter_title or domain_keyword in subject:
                return f"，从{dims}等维度展开分析"

        # 默认维度（避免过于空洞）
        return "，从核心要素、内在机制与关键特征等维度展开分析"

    def _extract_title_keywords(self, title: str) -> str:
        """
        从标题中提取关键词，用于生成描述

        当章节标题不匹配模板字典时，尝试从标题中识别关键词，
        生成更具针对性的描述。

        Args:
            title: 章节标题

        Returns:
            提取的关键词描述，如果无法提取则返回空字符串
        """
        # 常见关键词映射：标题关键词 -> 具体写作指导
        # 避免返回空洞短语，返回具体的写作方向指导
        keyword_map = {
            # 分析类
            "分析": "明确分析维度，运用相关理论框架进行系统剖析",
            "解析": "拆解核心要素，揭示各要素间的内在逻辑关系",
            "探讨": "围绕核心问题展开深入思考，提出明确观点",

            # 研究类
            "研究": "界定研究对象范围，阐述研究路径与方法选择",
            "调查": "说明调查方法、样本范围与数据来源",
            "考察": "明确考察维度，建立科学的评估标准",

            # 设计类
            "设计": "阐述设计理念，提出可落地的实施方案",
            "构建": "说明构建原则，绘制清晰的技术路线图",
            "建立": "明确建立方法，列出具体的实施步骤",

            # 案例类
            "案例": "选取典型案例，提炼可推广的经验与教训",
            "实例": "通过具体实例展示操作过程与实际效果",
            "示例": "以示例为载体，分析关键做法与成效",

            # 理论类
            "理论": "梳理相关理论，明确核心概念与理论框架",
            "原理": "阐明核心原理，解释其运作机制",
            "机制": "揭示内在机制，说明各要素的作用方式",

            # 实证类
            "实证": "说明数据来源，展示验证过程与结果",
            "验证": "阐述验证方法，呈现检验结论",
            "检验": "明确检验标准，给出判别依据",

            # 比较类
            "比较": "确定比较维度，分析异同并得出结论",
            "对比": "选取对比要素，揭示关键差异",
            "差异": "描述差异表现，探究成因与影响",

            # 发展类
            "发展": "梳理发展历程，分析各阶段特征与趋势",
            "演进": "呈现演变过程，提炼阶段性特征",
            "趋势": "基于现状分析，预测未来发展方向",
            "前景": "展望发展前景，分析机遇与挑战",
            "展望": "总结当前态势，预测未来发展路径",

            # 问题类
            "问题": "识别关键问题，分析问题表现与成因",
            "困境": "描述困境表现，探讨突破路径",
            "挑战": "列举面临的挑战，提出应对策略",

            # 对策类
            "对策": "针对问题提出解决路径与实施建议",
            "策略": "制定具体策略，明确执行要点",
            "路径": "设计实施路径，列出关键步骤",

            # 应用类
            "应用": "结合具体场景阐述应用方式与效果",
            "实践": "总结实践经验，提炼操作要点",
            "实施": "明确实施步骤，提出保障措施",

            # 评估类
            "评估": "建立评估指标体系，阐述评估方法",
            "评价": "设定评价标准，分析评价结果",
            "测评": "说明测评工具，展示量化方法",

            # 优化类
            "优化": "明确优化方向，提出改进措施",
            "改进": "分析改进需求，制定具体方案",
            "完善": "指出完善要点，设计提升路径",

            #党性分析类
            "党性": "对照党章党规，进行自我剖析与评价",
            "整改": "针对问题制定整改措施，明确责任落实",
            "查摆": "深入查摆问题，进行自我剖析",

            #历史类
            "历史": "梳理历史沿革，分析发展脉络",
            "历程": "呈现演进历程，提炼阶段特征",
            "革命": "回顾革命历程，总结历史贡献",
            "时期": "描述时代特征，交代历史背景",
            "阶段": "划分发展阶段，明确各阶段核心任务",

            #实习实践类
            "实习": "描述实习过程，总结能力提升与收获",
            "实践": "记录实践过程，提炼经验与感悟",
            "见习": "叙述见习经历，总结学习收获",
            "支教": "描述支教经历，分享教育感悟",
            "收获": "系统总结收获，提炼感悟与体会",

            #调研类
            "调研": "说明调研方法，呈现调研发现与结论",
            "营商": "分析营商环境现状，探讨优化对策",
            "市场": "分析市场现状，预测发展态势",

            #概念类
            "概念": "界定核心概念，分析其内涵与外延",
            "内涵": "深入解读内涵，揭示本质特征",
            "界定": "明确概念边界，划分范畴",

            # 影响类 - 改进具体性
            "影响": "分析影响的具体表现与作用机制，结合实例说明",
            "作用": "阐述作用方式，分析作用效果与案例",
            "意义": "阐明理论意义，说明实践价值",

            #建设类
            "建设": "提出建设思路，设计实施方案",
            "体系": "构建体系框架，设计结构组成",
            "制度": "分析制度现状，提出完善建议",

            #创新类
            "创新": "分析创新需求，提出突破方向与举措",
            "突破": "识别突破点，设计突破路径",

            #传承类
            "传承": "阐述传承方式，提出保护措施",
            "保护": "分析保护现状，提出保护机制",
            "非遗": "介绍非遗项目，提出传承与保护建议",

            #辩证类
            "辩证": "分析辩证关系，阐明统一逻辑",
            "联系": "揭示内在联系，分析互动关系",
            "区别": "明确区别特征，进行差异分析",
            "统一": "分析统一逻辑，设计整合路径",

            #自救类
            "自救": "介绍自救方法，分析心理应对策略",
            "安全": "阐述安全意识，提出防范措施",
            "教育": "分析教育意义，总结启示",

            # 特征类
            "特征": "归纳核心特征，分析表现形式",
            "特点": "提炼主要特点，进行对比分析",
            "现状": "描述当前现状，分析存在问题",

            # 场景类
            "场景": "描绘具体场景，分析场景要素与特征",
            "领域": "界定领域范围，分析领域特点",
            "行业": "分析行业现状，探讨发展趋势",
        }

        # 遍历关键词映射，查找匹配
        for keyword, description in keyword_map.items():
            if keyword in title:
                return description

        return ""