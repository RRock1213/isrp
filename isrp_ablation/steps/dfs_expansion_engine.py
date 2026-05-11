"""
Phase II — DFS 节点扩展引擎

核心生成模块，采用深度优先搜索策略自顶向下构建大纲树：

  纵向扩展 (Vertical Expansion):
    基于当前节点的记忆状态（相邻节点 + 父节点上下文摘要），
    单次生成所有子节点，实现全局分割以平衡内容分布。

  横向扩展 (Horizontal Expansion):
    当现有同级节点的话题覆盖度不足时，触发横向探针补充新节点。

  概念覆盖池 (Concept Coverage Pool):
    记录已覆盖的核心概念集合，生成前检查重复，生成后更新池。

  迭代优化:
    每个节点经历 initial evaluation -> refine -> re-evaluation 循环，
    保留历史最佳版本（防止优化退化）。

  内嵌章节检查点:
    完成一级章节子树后即时触发 ChapterLevelValidation，
    在缺陷向兄弟节点级联传播前实现即时遏制。

"""
from typing import List, Dict, Any, Optional, Tuple
import sys
import re

from ..schemas import (
    OutlineNode, NodeStatus, generate_node_id,
    NodeGenerationResult, robust_json_parse
)
from ..utils.llm_client import LLMClient
from ..utils.node_ops import NodeOperations
from ..utils.tree_ops import TreeOperations
from ..prompts import PromptTemplates
from ..engines import (
    ContextEngine, MIN_LEAF_WORDS, MAX_LEAF_WORDS, MAX_DEPTH, MAX_WIDTH,
    get_temperature
)

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout

# 类型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..evaluators.base import BaseEvaluator, EvalType
    from .chapter_level_validation import ChapterLevelValidation


class DFSExpansionEngine:
    """DFS Expansion Engine"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        evaluator: 'BaseEvaluator' = None,
        node_ops: NodeOperations = None,
        tree_ops: TreeOperations = None,
        enable_logging: bool = True,
        checkpoint: 'ChapterLevelValidation' = None,
        rag_service = None,
        max_width: int = MAX_WIDTH
    ):
        self.llm = llm_client
        self.prompts = prompts
        self.evaluator = evaluator
        self.node_ops = node_ops
        self.tree_ops = tree_ops
        self.enable_logging = enable_logging
        self._checkpoint = checkpoint
        self.rag_service = rag_service
        self.max_width = max_width

        # 已完成章节摘要列表
        self._completed_chapter_abstracts: List[str] = []

        # RAG 相关状态
        self._sample_id: str = ""  # 当前样本 ID
        self._rag_enabled: bool = False  # RAG 是否启用

    def set_sample_id(self, sample_id: str):
        """设置当前样本 ID（用于 RAG 检索）"""
        self._sample_id = sample_id
        self._rag_enabled = self.rag_service is not None and self.rag_service.is_enabled()

    def execute(
        self,
        user_prompt: str,
        root: OutlineNode,
        session,
        sample_id: str = None
    ) -> OutlineNode:
        """
        执行 DFS 展开

        Args:
            user_prompt: 用户指令
            root: 根节点
            session: 会话实例
            sample_id: 样本 ID（用于 RAG 检索）

        Returns:
            更新后的根节点
        """
        if self.enable_logging:
            print_and_flush("\n[DFS Expansion] DFS Expansion with Embedded Checkpoint")
            print_and_flush("=" * 60)

        # 初始化 sample_id 和 RAG 状态
        if sample_id:
            self.set_sample_id(sample_id)

        # 清空章节摘要
        self._completed_chapter_abstracts = []

        # 开始 DFS 展开
        self._dfs_expand(session, root, user_prompt, 0)

        # DFS完成后检查结构完整性
        self._validate_structure_completeness(session, root, user_prompt)

        return root

    def _dfs_expand(
        self,
        session,
        node: OutlineNode,
        user_prompt: str,
        depth: int
    ) -> None:
        """
        DFS 递归展开

        Args:
            session: 会话实例
            node: 当前节点
            user_prompt: 用户指令
            depth: 当前深度
        """
        # 深度检查：如果已达到最大深度，不再继续展开
        if node.level >= MAX_DEPTH:
            node.is_finalized = True
            if node.core_concepts:
                session.add_to_concept_registry(node.core_concepts, source_node=node)
            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}[Leaf] {node.title} (max depth reached)")
            return

        # 检查节点是否已有子节点（由根节点建立快捷路径创建）
        # 如果已有子节点，跳过生成步骤，直接处理已有子节点
        if node.children_ids and len(node.children_ids) > 0:
            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}[Expand] {node.title} (using existing {len(node.children_ids)} children)")

            # 为快捷路径创建的节点也进行 RAG 检索增强
            if self._rag_enabled and self._sample_id and node.level == 0:
                try:
                    rag_context = self.rag_service.get_rag_context_for_node(
                        sample_id=self._sample_id,
                        node_title=node.title,
                        node_description=node.description or "",
                        top_k=3,
                        max_length=1500,
                        node_level=node.level
                    )
                    if rag_context and self.enable_logging:
                        print_and_flush(f"{indent}[RAG] Retrieved context for shortcut path: {node.title[:30]}...")
                except Exception as e:
                    if self.enable_logging:
                        print_and_flush(f"{indent}[RAG] Error in shortcut path: {str(e)[:50]}")

            # 直接处理已有子节点
            for child_id in node.children_ids:
                child = session.get_node(child_id)
                if child is None:
                    continue

                # 判断是否为一级节点（章节）
                child_is_chapter = (child.level == 1) or (node.level == 0)

                # 为快捷路径创建的章节节点进行 RAG 检索
                if self._rag_enabled and self._sample_id and child_is_chapter:
                    try:
                        child_rag_context = self.rag_service.get_rag_context_for_node(
                            sample_id=self._sample_id,
                            node_title=child.title,
                            node_description=child.description or "",
                            top_k=3,
                            max_length=1000,
                            node_level=child.level
                        )
                        if child_rag_context and self.enable_logging:
                            indent2 = "  " * (depth + 1)
                            print_and_flush(f"{indent2}[RAG] Retrieved context for chapter: {child.title[:30]}...")
                        # 可选：将 RAG 上下文存储到节点中，供后续使用
                        if child_rag_context:
                            child.rag_context = child_rag_context
                    except Exception as e:
                        if self.enable_logging:
                            indent2 = "  " * (depth + 1)
                            print_and_flush(f"{indent2}[RAG] Error for chapter: {str(e)[:50]}")

                # 如果是 Level 1 节点（章节），在处理前进行优化
                if child_is_chapter and self._completed_chapter_abstracts:
                    self._refine_chapter_node_before_processing(session, child, user_prompt)

                # 如果子节点标记为已完成（固定格式章节），跳过扩展
                if child.is_finalized:
                    if self.enable_logging:
                        indent = "  " * depth
                        print_and_flush(f"{indent}  [Skip] {child.title} (fixed format)")
                    continue

                # 判断是否需要纵向展开
                needs_vertical = child.needs_vertical_expansion if hasattr(child, 'needs_vertical_expansion') else True

                if needs_vertical and child.level < MAX_DEPTH:
                    # 需要纵向展开：递归处理子节点
                    self._dfs_expand(session, child, user_prompt, depth + 1)
                else:
                    # 不需要纵向展开：标记为叶子节点
                    child.is_finalized = True
                    if child.core_concepts:
                        session.add_to_concept_registry(child.core_concepts, source_node=child)

            # 章节Checkpoint
            if node.level == 1 and self._checkpoint is not None:
                checkpoint_result = self._checkpoint.execute(
                    node, session, user_prompt, self._completed_chapter_abstracts
                )
                chapter_abstract = checkpoint_result.get("chapter_abstract", node.description or node.title)
                node.chapter_abstract = chapter_abstract
                self._completed_chapter_abstracts.append(chapter_abstract)

            return

        # 【核心步骤1】生成子节点并进行串行深化
        # 注意：原版没有检查 needs_vertical_expansion 和 children_ids
        # 每个未达最大深度的节点都会尝试生成子节点
        if self.enable_logging:
            indent = "  " * depth
            print_and_flush(f"{indent}[Expand] {node.title}")

        children = self._generate_and_refine_children(session, node, user_prompt, depth)

        # 如果没有生成子节点，标记为叶子节点
        if not children:
            node.is_finalized = True
            if node.core_concepts:
                session.add_to_concept_registry(node.core_concepts, source_node=node)
            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}[Leaf] {node.title} (no children generated)")
            return

        # 【核心步骤2】注册所有初始子节点
        for child in children:
            session.register_node(child)
            node.children_ids.append(child.node_id)
            child.parent_id = node.node_id

            # 注册核心概念
            if child.core_concepts:
                session.add_to_concept_registry(child.core_concepts, source_node=child)

        # 使用索引遍历，确保横向扩展新增的节点也会被处理
        processed_index = 0
        while processed_index < len(node.children_ids):
            child_id = node.children_ids[processed_index]
            child = session.get_node(child_id)

            if child is None:
                processed_index += 1
                continue

            # 检查是否是本次循环新增的节点（未被处理过）
            is_newly_added = child.status == NodeStatus.PENDING and child not in children

            if is_newly_added:
                if self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}  [DFS] Processing horizontally added node: {child.title}")

            # 判断是否为一级节点（章节）
            child_is_chapter = (child.level == 1) or (node.level == 0)

            # 如果是 Level 1 节点（章节），在处理前进行优化
            if child_is_chapter and self._completed_chapter_abstracts:
                self._refine_chapter_node_before_processing(session, child, user_prompt)

            # 判断是否需要纵向展开
            needs_vertical = child.needs_vertical_expansion if hasattr(child, 'needs_vertical_expansion') else False

            if needs_vertical and child.level < MAX_DEPTH:
                # 需要纵向展开：递归处理子节点
                self._dfs_expand(session, child, user_prompt, depth + 1)
            else:
                # 不需要纵向展开：标记为叶子节点
                child.is_finalized = True
                if child.core_concepts:
                    session.add_to_concept_registry(child.core_concepts, source_node=child)

            # 横向探针：检查是否需要补充同级节点
            self._horizontal_probe(session, node, child, user_prompt, depth)

            # 移动到下一个节点
            processed_index += 1

        # 【核心步骤6 - 新增】章节Checkpoint
        # 当一级节点（章节）的所有子节点处理完毕后，执行Checkpoint
        if node.level == 1 and self._checkpoint is not None:
            checkpoint_result = self._checkpoint.execute(
                node, session, user_prompt, self._completed_chapter_abstracts
            )
            # 保存章节摘要到节点属性
            chapter_abstract = checkpoint_result.get("chapter_abstract", node.description or node.title)
            node.chapter_abstract = chapter_abstract
            # 记录章节摘要到列表，用于后续章节优化
            self._completed_chapter_abstracts.append(chapter_abstract)

    def _generate_and_refine_children(
        self,
        session,
        parent: OutlineNode,
        user_prompt: str,
        depth: int
    ) -> List[OutlineNode]:
        """
        生成子节点并进行串行深化

        【执行流程】
        1. 构建上下文（防遗忘注入）
        1.5 RAG 检索增强
        2. LLM调用生成候选节点列表
        3. 为每个候选创建OutlineNode对象
        4. 主题重复检测
        5. 串行深化评估（最多2次迭代）
        6. 返回最终的children列表

        【关键点】
        - LLM返回的children数量通常为2-4个
        - 所有候选节点都会被创建，不会被丢弃
        - 串行深化只修改节点属性，不删除节点
        """
        # 确保父节点有描述
        if parent.description is None or not parent.description.strip():
            parent.description = f"关于'{parent.title}'的内容阐述"

        # 构建上下文
        context = ContextEngine.build_full_context(
            parent, user_prompt, session
        )

        if not context or not context.strip():
            # 使用 effective_prompt（已由 planner 处理）
            effective_prompt = session.get_effective_prompt()
            context = f"【用户指令】\n{effective_prompt}\n\n【当前节点】\n{parent.title}"

        # RAG 检索增强
        rag_context = ""
        if self._rag_enabled and self._sample_id:
            try:
                rag_context = self.rag_service.get_rag_context_for_node(
                    sample_id=self._sample_id,
                    node_title=parent.title,
                    node_description=parent.description or "",
                    top_k=3,
                    max_length=1500,
                    node_level=parent.level
                )
                if rag_context and self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}[RAG] Retrieved context for: {parent.title[:30]}...")
            except Exception as e:
                if self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}[RAG] Error: {str(e)[:50]}")

        # 计算当前节点的层级编号
        node_index = TreeOperations.get_node_index(session, parent) if self.tree_ops else ""

        # 获取风格上下文和关键实体上下文
        style_context = session.get_style_context() if hasattr(session, 'get_style_context') else ""
        key_entities_context = ""
        if hasattr(session, 'style_result') and session.style_result:
            entities = session.style_result.key_entities
            if entities:
                key_entities_context = f"关键实体: {', '.join(entities[:10])}"

        # 将 RAG 上下文合并到 context
        if rag_context:
            context = context + "\n\n" + rag_context

        min_width = min(2, self.max_width)
        max_width = self.max_width

        lang = self.prompts._language
        prompt = self.prompts.get_template("PROMPT_3_GENERATE", lang).format(
            context=context,
            style_context=style_context,
            total_words=session.global_target_words,
            level=parent.level + 1,
            node_index=node_index,
            node_title=parent.title,
            node_description=parent.description or "待展开",
            key_entities_context=key_entities_context,
            min_width=min_width,
            max_width=max_width
        )

        try:
            response = self.llm.call(
                prompt,
                temperature=get_temperature("dfs_expansion", "generate")
            )
            result = robust_json_parse(response, dict)
        except Exception as e:
            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}[Error] Generation failed: {e}")
            return []

        # 解析子节点
        children_data = result.get("children", [])
        children = []

        for i, child_data in enumerate(children_data):
            # 计算节点编号
            parent_index = TreeOperations.get_node_index(session, parent) if self.tree_ops else ""
            child_num = len(parent.children_ids) + i + 1
            child_index = f"{parent_index}.{child_num}" if parent_index else str(child_num)

            # 清理标题
            raw_title = child_data.get("title", f"子节点 {child_num}")
            title_without_index = re.sub(r'^[\d.]+\s*', '', raw_title).strip()
            final_title = f"{child_index} {title_without_index}"

            child = OutlineNode(
                node_id=generate_node_id(),
                title=final_title,
                description=child_data.get("description", ""),
                level=parent.level + 1,
                status=NodeStatus.PENDING,
                word_budget=0,
                parent_id=parent.node_id,
                core_concepts=child_data.get("core_concepts", []),
                needs_vertical_expansion=child_data.get("needs_vertical_expansion", True)
            )

            children.append(child)

        # 【步骤2.5】主题重复检测
        self._check_theme_repetition(children, depth)

        # 【步骤3】串行深化评估
        children = self._refine_children_iteratively(session, parent, children, user_prompt, depth)

        return children

    def _generate_children(
        self,
        session,
        parent: OutlineNode,
        user_prompt: str,
        depth: int
    ) -> List[OutlineNode]:
        """生成子节点"""
        # 构建上下文
        context = ContextEngine.build_full_context(
            parent, user_prompt, session
        )

        # 获取风格上下文
        style_context = session.get_style_context() if hasattr(session, 'get_style_context') else ""
        key_entities_context = ""
        if hasattr(session, 'style_result') and session.style_result:
            entities = session.style_result.key_entities
            if entities:
                key_entities_context = f"关键实体: {', '.join(entities[:10])}"

        # 获取节点索引
        node_index = TreeOperations.get_node_index(session, parent) if self.tree_ops else ""

        # 构建 Prompt
        min_width = min(2, self.max_width)
        max_width = self.max_width

        lang = self.prompts._language
        prompt = self.prompts.get_template("PROMPT_3_GENERATE", lang).format(
            context=context,
            style_context=style_context,
            total_words=session.global_target_words,
            level=parent.level + 1,
            node_index=node_index,
            node_title=parent.title,
            node_description=parent.description or "待展开",
            key_entities_context=key_entities_context,
            min_width=min_width,
            max_width=max_width
        )

        try:
            response = self.llm.call(
                prompt,
                temperature=get_temperature("dfs_expansion", "generate")
            )
            result = robust_json_parse(response, dict)
        except Exception as e:
            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}[Error] Generation failed: {e}")
            return []

        # 解析子节点
        children_data = result.get("children", [])
        children = []

        for i, child_data in enumerate(children_data):
            # 计算节点编号
            parent_index = TreeOperations.get_node_index(session, parent) if self.tree_ops else ""
            child_num = len(parent.children_ids) + i + 1
            child_index = f"{parent_index}.{child_num}" if parent_index else str(child_num)

            # 清理标题
            raw_title = child_data.get("title", f"子节点 {child_num}")
            title_without_index = re.sub(r'^[\d.]+\s*', '', raw_title).strip()
            final_title = f"{child_index} {title_without_index}"

            child = OutlineNode(
                node_id=generate_node_id(),
                title=final_title,
                description=child_data.get("description", ""),
                level=parent.level + 1,
                status=NodeStatus.PENDING,
                word_budget=0,
                parent_id=parent.node_id,
                core_concepts=child_data.get("core_concepts", []),
                needs_vertical_expansion=child_data.get("needs_vertical_expansion", True)
            )

            children.append(child)

        return children

    def _check_theme_repetition(
        self,
        children: List[OutlineNode],
        depth: int
    ) -> None:
        """
        检查子节点之间的主题重复度

        【检测逻辑】
        1. 提取每个节点的标题关键词和核心概念
        2. 计算节点之间的概念重叠度
        3. 如果重叠度过高（>50%），打印警告
        """
        if len(children) < 2:
            return

        # 收集所有节点的关键词
        node_keywords = []
        for child in children:
            title_words = set(child.title.split())
            desc_words = set(child.description.split()) if child.description else set()
            concepts = set(child.core_concepts) if child.core_concepts else set()
            all_keywords = title_words | desc_words | concepts
            filtered_keywords = {w for w in all_keywords if len(w) > 1}
            node_keywords.append(filtered_keywords)

        # 检查节点之间的重叠度
        if self.enable_logging:
            indent = "  " * depth
            high_overlap_pairs = []

            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    if not node_keywords[i] or not node_keywords[j]:
                        continue

                    intersection = node_keywords[i] & node_keywords[j]
                    union = node_keywords[i] | node_keywords[j]
                    overlap_ratio = len(intersection) / len(union) if union else 0

                    if overlap_ratio > 0.5:
                        high_overlap_pairs.append((
                            children[i].title,
                            children[j].title,
                            overlap_ratio,
                            list(intersection)[:5]
                        ))

            if high_overlap_pairs:
                print_and_flush(f"{indent}[Warning] High overlap detected between sibling nodes:")
                for title1, title2, ratio, overlap_words in high_overlap_pairs:
                    print_and_flush(f"{indent}  - '{title1}' vs '{title2}': {ratio:.1%} overlap")
                    print_and_flush(f"{indent}    Overlapping concepts: {', '.join(overlap_words)}")

    def _refine_children_iteratively(
        self,
        session,
        parent: OutlineNode,
        children: List[OutlineNode],
        user_prompt: str,
        depth: int
    ) -> List[OutlineNode]:
        """
        串行深化：基于评分细则迭代优化子节点

        【优化后的执行流程】
        1. 首先对所有子节点进行初始评估，获得真实基准分数
        2. 最多迭代2次优化
        3. 每个节点独立跟踪达标状态：
           - 已达标（分数 >= 9.0）的节点跳过后续评估
           - 未达标的节点继续评估和修改
        4. 所有节点都达标时提前退出

        最佳结果保留机制：
        1. 对初始状态进行真实LLM评估，获得准确基准分数
        2. 为每个节点维护最佳结果记录
        3. 迭代结束后统一应用最佳结果
        4. 确保最终结果不会比初始状态差（通过分数比较）
        """
        max_refine_iterations = 2

        # 为每个节点维护最佳结果
        best_results: Dict[str, Dict] = {}

        # 跟踪每个节点的达标状态
        perfected = {child.node_id: False for child in children}

        # 第一步：对所有子节点进行初始评估，获得真实基准分数
        if self.enable_logging:
            indent = "  " * depth
            print_and_flush(f"{indent}  [Initial Evaluation] Evaluating {len(children)} children to establish baseline...")

        for child in children:
            is_perfect, initial_result = self._evaluate_child_node(
                session, parent, child, user_prompt, depth,
                previous_best=None
            )

            best_results[child.node_id] = initial_result

            if is_perfect:
                perfected[child.node_id] = True
                if self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}    [{child.title}] Initial score: {initial_result['score']:.2f} - Already perfect!")

        # 检查是否所有节点初始评估就达标
        if all(perfected.values()):
            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}  [Refine] All children are already perfect after initial evaluation!")
            # 统一应用最佳结果
            for child in children:
                best = best_results[child.node_id]
                child.title = best["title"]
                child.description = best["description"]
                child.needs_vertical_expansion = best["needs_vertical_expansion"]
            return children

        # 第二步：迭代优化未达标的节点
        for iteration in range(max_refine_iterations):
            if self.enable_logging:
                indent = "  " * depth
                pending_count = sum(1 for child in children if not perfected.get(child.node_id, False))
                print_and_flush(f"{indent}  [Refine {iteration+1}/{max_refine_iterations}] Optimizing {pending_count}/{len(children)} children below threshold...")

            all_perfect = True
            for child in children:
                # 跳过已达标的节点
                if perfected.get(child.node_id, False):
                    continue

                # 传入上一轮最佳结果，获取新的最佳结果
                is_perfect, best_result = self._evaluate_child_node(
                    session, parent, child, user_prompt, depth,
                    previous_best=best_results.get(child.node_id)
                )

                # 更新最佳结果记录
                best_results[child.node_id] = best_result

                if is_perfect:
                    perfected[child.node_id] = True
                else:
                    all_perfect = False

            if all_perfect:
                if self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}  [Refine] All children are perfect, stopping refinement")
                break

        # 统一应用最佳结果
        for child in children:
            best = best_results[child.node_id]
            child.title = best["title"]
            child.description = best["description"]
            child.needs_vertical_expansion = best["needs_vertical_expansion"]

            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}    [{child.title}] Final score: {best['score']:.2f}")

        return children

    def _evaluate_child_node(
        self,
        session,
        parent: OutlineNode,
        child: OutlineNode,
        user_prompt: str,
        depth: int,
        previous_best: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        评估子节点是否完善

        Args:
            session: 会话实例
            parent: 父节点
            child: 待评估的子节点
            user_prompt: 用户指令
            depth: 当前深度
            previous_best: 上一轮最佳结果，用于比较避免迭代退化
        """
        if self.enable_logging:
            indent = "  " * depth
            print_and_flush(f"{indent}  [Evaluate] {child.title}")

        # 强制使用评估器 - 不允许无评估器的简化实现
        if self.evaluator is None:
            raise ValueError(
                "评估器未初始化！必须在创建 DFSExpansionEngine 时传入 evaluator 参数。"
                "请使用 EvaluatorFactory.create('isrp') 创建 ISRP 评估器。"
            )

        return self._evaluate_with_evaluator(session, parent, child, user_prompt, depth, previous_best)

    def _evaluate_with_evaluator(
        self,
        session,
        parent: OutlineNode,
        child: OutlineNode,
        user_prompt: str,
        depth: int,
        previous_best: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        使用评估器评估

        Args:
            session: 会话实例
            parent: 父节点
            child: 待评估的子节点
            user_prompt: 用户指令
            depth: 当前深度
            previous_best: 上一轮最佳结果，用于比较避免迭代退化
        """
        from ..evaluators.base import EvalType

        # 获取上下文（使用 session 中的 effective_prompt）
        context = ContextEngine.build_adjacent_context(child, session)

        eval_prompt = self.evaluator.get_dfs_eval_prompt(
            node_title=child.title,
            node_description=child.description,
            context=context,
            language=self.prompts._language
        )

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.llm.call(
                    eval_prompt,
                    temperature=get_temperature("dfs_expansion", "evaluate")
                )

                if not response or not response.strip():
                    # LLM 返回空，保留当前状态作为最佳结果
                    current_result = {
                        "score": 8.0,
                        "title": child.title,
                        "description": child.description,
                        "needs_vertical_expansion": child.needs_vertical_expansion
                    }
                    return True, current_result

                eval_result = self.evaluator.parse_result(response, EvalType.DFS_EVAL)
                avg_score = eval_result.average_score
                data = eval_result.extra

                is_perfect = avg_score >= 9.0

                # 与上一轮最佳结果比较
                if previous_best and avg_score < previous_best.get("score", 0):
                    # 当前结果不如上一轮，保留上一轮最佳结果
                    if self.enable_logging:
                        indent = "  " * depth
                        print_and_flush(f"{indent}    Score: {avg_score:.2f} < previous {previous_best['score']:.2f}, keeping previous best")
                    return is_perfect, previous_best

                if self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}    Score: {avg_score:.2f}, Perfect: {is_perfect}")

                result = {
                    "score": avg_score,
                    "title": data.get("refined_title") or child.title,
                    "description": data.get("refined_description") or child.description,
                    "needs_vertical_expansion": data.get("needs_vertical_expansion", child.needs_vertical_expansion)
                }

                return is_perfect, result

            except Exception as e:
                last_error = e
                if self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}    Evaluation error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue

        # 所有重试都失败，保持当前状态
        current_result = {
            "score": 8.0,
            "title": child.title,
            "description": child.description,
            "needs_vertical_expansion": child.needs_vertical_expansion
        }
        return True, current_result

    def _horizontal_probe(
        self,
        session,
        parent: OutlineNode,
        last_child: OutlineNode,
        user_prompt: str,
        depth: int
    ) -> None:
        """横向探针：判断是否需要补充同级节点"""
        # 宽度上限检查：已达上限则不再补充
        if len(parent.children_ids) >= self.max_width:
            if self.enable_logging:
                indent = "  " * depth
                print_and_flush(f"{indent}  [Horizontal Probe] Max width {self.max_width} reached, stop probing")
            return

        if self.enable_logging:
            indent = "  " * depth
            print_and_flush(f"{indent}  [Horizontal Probe] Checking if more siblings needed...")

        # 收集兄弟节点信息
        existing_siblings_info = []
        for cid in parent.children_ids:
            sibling = session.get_node(cid)
            if sibling:
                sibling_info = f"标题: {sibling.title}"
                if sibling.description:
                    sibling_info += f"\n描述: {sibling.description[:200]}..."
                existing_siblings_info.append(sibling_info)

        # 收集已覆盖概念
        covered_concepts = session.get_covered_concepts_set() if hasattr(session, 'get_covered_concepts_set') else set()

        # 使用 effective_prompt（已由 planner 处理压缩或原始）
        effective_prompt = session.get_effective_prompt()

        from ..prompts import PromptTemplates
        probe_prompt = PromptTemplates.PROMPT_HORIZONTAL_PROBE_ZH.format(
            effective_prompt=effective_prompt,
            global_abstract=session.global_abstract or '未设置全局摘要',
            parent_title=parent.title,
            parent_desc=parent.description if parent.description else '无',
            sibling_count=len(parent.children_ids),
            sibling_list=chr(10).join([f'{i+1}. {info}' for i, info in enumerate(existing_siblings_info)]),
            covered_concepts=', '.join(sorted(covered_concepts)[:30]) if covered_concepts else '无',
            last_child_title=last_child.title)

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.llm.call(
                    probe_prompt,
                    temperature=get_temperature("dfs_expansion", "evaluate")
                )

                if not response or not response.strip():
                    if attempt < max_retries - 1:
                        continue  # 重试
                    return

                result = robust_json_parse(response, dict)

                if not result:
                    if attempt < max_retries - 1:
                        continue  # 重试
                    return

                needs_more = result.get("needs_more_siblings", False)
                coverage_score = result.get("coverage_score", 10)

                if coverage_score >= 8:
                    if self.enable_logging:
                        indent = "  " * depth
                        print_and_flush(f"{indent}  [Horizontal Probe] Coverage score {coverage_score}/10, no need for more siblings")
                    return

                if needs_more and result.get("suggested_sibling"):
                    sibling_data = result["suggested_sibling"]

                    # 检查是否重复
                    raw_title = sibling_data.get("title", "补充节点")
                    title_without_index = re.sub(r'^[\d.]+\s*', '', raw_title).strip()

                    # 简单的重复检测
                    existing_titles_lower = [
                        session.get_node(cid).title.lower().replace(" ", "")
                        for cid in parent.children_ids
                        if session.get_node(cid)
                    ]
                    new_title_lower = title_without_index.lower().replace(" ", "")

                    for existing_lower in existing_titles_lower:
                        if new_title_lower in existing_lower or existing_lower in new_title_lower:
                            if self.enable_logging:
                                indent = "  " * depth
                                print_and_flush(f"{indent}  [Horizontal Probe] Skipped: similar to existing node")
                            return

                    # 创建新节点
                    parent_index = TreeOperations.get_node_index(session, parent) if self.tree_ops else ""
                    new_sibling_num = len(parent.children_ids) + 1
                    new_sibling_index = f"{parent_index}.{new_sibling_num}" if parent_index else str(new_sibling_num)
                    final_title = f"{new_sibling_index} {title_without_index}"

                    new_sibling = OutlineNode(
                        node_id=generate_node_id(),
                        title=final_title,
                        description=sibling_data.get("description", ""),
                        level=last_child.level,
                        status=NodeStatus.PENDING,
                        word_budget=0,
                        parent_id=parent.node_id,
                        prev_sibling_id=last_child.node_id,
                        needs_vertical_expansion=True
                    )

                    session.register_node(new_sibling)
                    parent.children_ids.append(new_sibling.node_id)

                    if new_sibling.core_concepts:
                        session.add_to_concept_registry(new_sibling.core_concepts, source_node=new_sibling)

                    if self.enable_logging:
                        indent = "  " * depth
                        diff = sibling_data.get("differentiation", "无")
                        print_and_flush(f"{indent}  [Horizontal Probe] Added new sibling: {new_sibling.title}")
                        print_and_flush(f"{indent}    Differentiation: {diff}")

                return  # 成功完成，退出重试循环

            except Exception as e:
                last_error = e
                if self.enable_logging:
                    indent = "  " * depth
                    print_and_flush(f"{indent}  [Horizontal Probe] Error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue

    def _refine_chapter_node_before_processing(
        self,
        session,
        chapter_node: OutlineNode,
        user_prompt: str
    ) -> None:
        """
        在开始处理章节前，基于前序章节摘要优化当前章节节点

        【执行时机】在 DFS 开始处理新的 Level 1 节点之前

        【功能】
        1. 获取前序章节摘要
        2. 调用 LLM 评估当前章节是否需要调整以增强衔接
        3. 如需调整，更新 title/description/core_concepts

        Args:
            session: 会话实例
            chapter_node: 当前待处理的章节节点（Level 1）
            user_prompt: 用户原始指令
        """
        if not self._completed_chapter_abstracts:
            # 首章无需优化
            return

        if self.enable_logging:
            print(f"\n  [Refine Chapter] Checking: {chapter_node.title}")

        # 构建前序章节摘要
        prev_abstracts = "\n".join([
            f"第{i+1}章摘要: {abstract}"
            for i, abstract in enumerate(self._completed_chapter_abstracts[-3:])
        ])

        # 使用 effective_prompt（已由 planner 处理压缩或原始）
        effective_prompt = session.get_effective_prompt()

        # 获取风格上下文
        style_context = session.get_style_context() if hasattr(session, 'get_style_context') else ""

        # 构建优化 Prompt
        from ..prompts import PromptTemplates
        prompt = PromptTemplates.PROMPT_CONTENT_SUPPLEMENT_ZH.format(
            effective_prompt=effective_prompt,
            existing_summary=prev_abstracts,
            style_context=style_context)

        try:
            response = self.llm.call(
                prompt,
                temperature=get_temperature("dfs_expansion", "generate")
            )

            if not response or not response.strip():
                if self.enable_logging:
                    print_and_flush("    [Content Generation] Empty response, skipping")
                return

            result = robust_json_parse(response, dict)
            if not result:
                if self.enable_logging:
                    print_and_flush("    [Content Generation] Failed to parse response")
                return

            new_chapters_data = result.get("new_chapters", [])
            if not new_chapters_data:
                if self.enable_logging:
                    print_and_flush("    [Content Generation] No new chapters suggested")
                return

            # 获取根节点
            root = session.get_node(session.root_id)

            # 创建新章节节点
            for ch_data in new_chapters_data:
                insert_after = ch_data.get("insert_after", len(root.children_ids))

                # 计算章节编号
                new_chapter_num = insert_after + 1
                new_chapter_index = str(new_chapter_num)

                title = ch_data.get("title", "补充章节")
                # 清理标题中可能已有的编号
                clean_title = re.sub(r'^第?[一二两三四五六七八九十\d]+章\s*', '', title).strip()
                final_title = f"第{new_chapter_index}章 {clean_title}"

                new_chapter = OutlineNode(
                    node_id=generate_node_id(),
                    title=final_title,
                    description=ch_data.get("description", ""),
                    level=1,
                    status=NodeStatus.PENDING,
                    word_budget=0,
                    parent_id=root.node_id,
                    core_concepts=ch_data.get("core_concepts", []),
                    needs_vertical_expansion=True
                )

                session.register_node(new_chapter)

                # 插入到正确位置
                if insert_after >= len(root.children_ids):
                    root.children_ids.append(new_chapter.node_id)
                else:
                    root.children_ids.insert(insert_after, new_chapter.node_id)

                if self.enable_logging:
                    print_and_flush(f"    [Content Generation] Added chapter: {final_title}")

                # 注册核心概念
                if new_chapter.core_concepts:
                    session.add_to_concept_registry(new_chapter.core_concepts, source_node=new_chapter)

            # 重新编号所有章节
            self._renumber_chapters(session, root)

            # 对新章节执行DFS扩展
            for ch_data in new_chapters_data:
                insert_after = ch_data.get("insert_after", len(root.children_ids))
                if insert_after < len(root.children_ids):
                    new_chapter_id = root.children_ids[insert_after]
                    new_chapter = session.get_node(new_chapter_id)
                    if new_chapter and new_chapter.needs_vertical_expansion:
                        self._dfs_expand(session, new_chapter, user_prompt, 1)

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"    [Content Generation] Error: {str(e)[:100]}")

    def _renumber_chapters(self, session, root: OutlineNode) -> None:
        """
        重新编号所有章节

        当插入新章节后，需要重新编号以保持连续性。
        """
        for i, chapter_id in enumerate(root.children_ids):
            chapter = session.get_node(chapter_id)
            if not chapter:
                continue

            new_num = i + 1
            old_title = chapter.title

            # 提取标题中非编号部分
            clean_title = re.sub(r'^第?[一二两三四五六七八九十\d]+章\s*', '', old_title).strip()
            chapter.title = f"第{new_num}章 {clean_title}"

            if self.enable_logging and chapter.title != old_title:
                print_and_flush(f"    [Renumber] '{old_title[:25]}...' -> '{chapter.title[:25]}...'")