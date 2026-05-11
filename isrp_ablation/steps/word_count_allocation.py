"""
Phase III — BFS 字数分配

采用广度优先策略，将总字数预算 W_t 逐层按权重分配到每个叶子节点：
  1. 结构约束检测：子节点数量与字数预算的可行范围匹配，
     超出范围时触发合并式优化
  2. 节点权重评估：LLM 评估同级子节点的相对重要性权重 omega_v，
     按比例分配父节点预算
  3. 甜点区间调整：确保每个叶子节点的字数落在 [w_min, w_max] 内，
     低于下限则合并，高于上限则拆分
  4. 末端修正：消除比例分配的累积误差，确保总字数精确匹配 W_t

此模块与大纲生成完全解耦，大纲树稳定后才执行。

"""
import sys
from collections import deque
from typing import List

from ..schemas import OutlineNode, NodeWeight, robust_json_parse
from ..utils.llm_client import LLMClient
from ..utils.node_ops import NodeOperations
from ..utils.tree_ops import TreeOperations
from ..prompts import PromptTemplates
from ..engines import (
    WordAllocationEngine, MIN_LEAF_WORDS, MAX_LEAF_WORDS,
    get_temperature
)

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout

class WordCountAllocation:
    """Word Count Allocation (BFS)"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        node_ops: NodeOperations,
        enable_logging: bool = True
    ):
        self.llm = llm_client
        self.prompts = prompts
        self.node_ops = node_ops
        self.enable_logging = enable_logging

    def execute(self, session) -> None:
        """
        执行全局字数分配

        Args:
            session: 会话实例
        """
        if self.enable_logging:
            print_and_flush("\n[Word Count Allocation] Global Allocation with LLM Weight Evaluation")
            print_and_flush("=" * 60)

        root = session.get_node(session.root_id)
        total_budget = session.global_target_words

        # Phase 0: 预检测与节点数量优化
        self._pre_merge_if_needed(session, total_budget)

        # Phase 1: BFS 权重分配
        queue = deque([(root, total_budget)])

        while queue:
            node, budget = queue.popleft()
            children = session.get_children(node.node_id)

            if not children:
                # 叶子节点：直接分配字数
                node.allocated_words = budget
                continue

            # LLM 评估权重
            children = self._evaluate_weights_with_llm(session, node, children)

            # 按权重比例分配
            children = self._allocate_by_weight(children, budget)

            for child in children:
                session.node_registry[child.node_id] = child
                queue.append((child, child.allocated_words))

        # Phase 2: 甜点检测与重组
        self._detect_and_restructure_sweet_spot(session)

        # Phase 3: 归一化
        leaves = session.get_all_leaf_nodes()
        current_total = sum(n.allocated_words for n in leaves)

        if current_total != total_budget:
            if self.enable_logging:
                print_and_flush(f"  [Warning] Word count mismatch: {current_total} vs {total_budget}, normalizing...")
            leaves = WordAllocationEngine.normalize_allocation(leaves, total_budget, MIN_LEAF_WORDS, MAX_LEAF_WORDS)
            for leaf in leaves:
                session.node_registry[leaf.node_id] = leaf

        # Phase 4: 单子节层级回退优化
        self._optimize_single_section_chapters(session)

        # 输出大纲可视化
        if self.enable_logging:
            TreeOperations.print_outline_tree(session)
            total = sum(n.allocated_words for n in leaves)
            print_and_flush(f"\n  Allocated {len(leaves)} leaves, total: {total} words")

    def _pre_merge_if_needed(self, session, total_budget: int) -> None:
        """预检测与节点数量优化（ 改进：放宽阈值，保护章节结构）"""
        SWEET_SPOT_MID = 200  # 每200字一个节点，支持更细粒度划分

        leaves = session.get_all_leaf_nodes()
        current_count = len(leaves)

        if current_count == 0:
            return

        ideal_count = total_budget // SWEET_SPOT_MID
        max_allowed = int(ideal_count * 2.0)  # 允许 2x 理想数量

        if current_count <= max_allowed:
            if self.enable_logging:
                print_and_flush(f"  [Phase 0] Leaf count OK: {current_count} <= {max_allowed}")
            return

        if self.enable_logging:
            print_and_flush(f"\n  [Phase 0] Pre-merge Required: {current_count} -> {max_allowed}")

        merge_count = current_count - max_allowed
        self._execute_pre_merge(session, merge_count)

    def _execute_pre_merge(self, session, merge_count: int) -> None:
        """执行预合并（ 改进：修复 index out of range 错误）"""
        leaves = session.get_all_leaf_nodes()

        # 按父节点分组
        sibling_groups = {}
        for leaf in leaves:
            if leaf.parent_id not in sibling_groups:
                sibling_groups[leaf.parent_id] = []
            sibling_groups[leaf.parent_id].append(leaf)
        MIN_CHILDREN_PER_CHAPTER = 2

        merged = 0
        for parent_id, siblings in sibling_groups.items():
            if merged >= merge_count:
                break
            if len(siblings) <= MIN_CHILDREN_PER_CHAPTER:
                if self.enable_logging and len(siblings) > 1:
                    print_and_flush(f"    [Protect] {parent_id}: keeping {len(siblings)} children (minimum)")
                continue

            # 按标题排序（保持顺序）
            siblings.sort(key=lambda x: x.title)
            siblings_by_size = sorted(siblings, key=lambda x: x.allocated_words)

            while len(siblings_by_size) > MIN_CHILDREN_PER_CHAPTER and merged < merge_count:
                if len(siblings_by_size) < 2:
                    break

                source = siblings_by_size[0]  # 最小的节点
                target = siblings_by_size[1]  # 第二小的节点

                # 检查合并后是否超过字数上限
                merged_words = target.allocated_words + source.allocated_words
                if merged_words > MAX_LEAF_WORDS:
                    # 合并后会超过上限，移除最小节点尝试下一个组合
                    siblings_by_size.pop(0)
                    if len(siblings_by_size) < 2:
                        break
                    continue

                NodeOperations.merge_two_nodes(
                    session, target, source,
                    self.llm, self.prompts, self.enable_logging
                )

                # 更新列表
                siblings_by_size.pop(0)  # 移除已合并的 source
                if len(siblings_by_size) > 0:
                    siblings_by_size[0] = target  # 更新 target 的字数
                merged += 1

    def _evaluate_weights_with_llm(self, session, parent: OutlineNode, children: List[OutlineNode]) -> List[OutlineNode]:
        """使用 LLM 评估权重"""
        lang = self.prompts._language
        style_context = session.get_style_context() if hasattr(session, 'get_style_context') else ""

        # 构建兄弟节点列表
        siblings_list = "\n".join([
            f"  - ID: {child.node_id}, 标题: {child.title}, 描述: {child.description[:100] if child.description else '无'}..."
            for child in children
        ])

        prompt = self.prompts.get_template("PROMPT_5_WEIGHT", lang).format(
            style_context=style_context,
            parent_title=parent.title,
            parent_description=parent.description or "无描述",
            siblings_list=siblings_list
        )

        try:
            response = self.llm.call(prompt, temperature=get_temperature("word_allocation", "weight"))
            result = robust_json_parse(response, dict)

            # 更新权重
            allocations = result.get("allocations", [])
            weight_map = {a["node_id"]: a["relative_weight"] for a in allocations if "node_id" in a and "relative_weight" in a}

            for child in children:
                if child.node_id in weight_map:
                    child.relative_weight = weight_map[child.node_id]
                else:
                    child.relative_weight = 5  # 默认权重

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"    [Warning] Weight evaluation failed: {e}, using equal weights")
            for child in children:
                child.relative_weight = 5

        return children

    def _allocate_by_weight(self, nodes: List[OutlineNode], total_budget: int) -> List[OutlineNode]:
        """按权重比例分配字数"""
        if not nodes:
            return nodes

        total_weight = sum(node.relative_weight for node in nodes)

        if total_weight <= 0:
            avg = total_budget // len(nodes)
            for node in nodes:
                node.allocated_words = avg
            return nodes

        allocated = 0
        for i, node in enumerate(nodes):
            if i == len(nodes) - 1:
                # 最后一个节点：吸收剩余预算
                remaining = total_budget - allocated
                node.allocated_words = remaining
            else:
                ratio = node.relative_weight / total_weight
                budget = int(ratio * total_budget)
                node.allocated_words = budget
                allocated += budget

        return nodes

    def _detect_and_restructure_sweet_spot(self, session) -> None:
        """甜点检测与重组"""
        if self.enable_logging:
            print_and_flush("\n  [Phase 2] Sweet Spot Detection & Restructuring")
        max_iterations = 5  # 防止无限循环
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            leaf_nodes = session.get_all_leaf_nodes()
            nodes_to_merge = [n for n in leaf_nodes if n.allocated_words < MIN_LEAF_WORDS]
            nodes_to_split = [n for n in leaf_nodes if n.allocated_words > MAX_LEAF_WORDS]

            if iteration == 1 and self.enable_logging:
                print_and_flush(f"    Total leaf nodes: {len(leaf_nodes)}")
                print_and_flush(f"    Nodes below {MIN_LEAF_WORDS} words: {len(nodes_to_merge)}")
                print_and_flush(f"    Nodes above {MAX_LEAF_WORDS} words: {len(nodes_to_split)}")

            if not nodes_to_split and not nodes_to_merge:
                # 所有节点都在范围内，退出循环
                break

            # 1. 处理字数过多的节点（拆分）
            for node in nodes_to_split:
                if self.enable_logging:
                    print_and_flush(f"    [Split] Splitting: {node.title} ({node.allocated_words} words)")

                # 调用拆分逻辑
                split_data = self._generate_split_data(session, node)
                if split_data and split_data.get("new_nodes"):
                    NodeOperations.split_node(session, node, split_data["new_nodes"])
                    if self.enable_logging:
                        print_and_flush(f"      -> Split into {len(split_data['new_nodes'])} nodes")
                else:
                    # 拆分失败，强制限制字数
                    node.allocated_words = MAX_LEAF_WORDS
                    if self.enable_logging:
                        print_and_flush(f"      -> Failed to split, capping at {MAX_LEAF_WORDS} words")

            # 2. 处理字数过少的节点（合并）- 需要重新获取叶子节点
            leaf_nodes = session.get_all_leaf_nodes()
            nodes_to_merge = [n for n in leaf_nodes if n.allocated_words < MIN_LEAF_WORDS]

            for node in nodes_to_merge:
                NodeOperations.merge_with_sibling(
                    session, node, self.llm, self.prompts, self.enable_logging
                )

    def _optimize_single_section_chapters(self, session) -> None:
        """单子节层级回退优化"""
        if self.enable_logging:
            print_and_flush("\n  [Phase 4] Single-Section Chapter Optimization")

        root = session.get_node(session.root_id)
        if not root:
            return

        chapters = session.get_children(root.node_id)
        optimized_count = 0

        for chapter in chapters:
            children = session.get_children(chapter.node_id)

            if len(children) == 1:
                single_child = children[0]
                merged_words = chapter.allocated_words + single_child.allocated_words
                if merged_words > MAX_LEAF_WORDS:
                    # 合并后会超过上限，跳过此次优化
                    if self.enable_logging:
                        print_and_flush(f"    [Skip] {chapter.title}: merged words ({merged_words}) > {MAX_LEAF_WORDS}")
                    continue

                # 合并描述
                if single_child.description:
                    if chapter.description:
                        chapter.description = f"{chapter.description}\n{single_child.description}"
                    else:
                        chapter.description = single_child.description

                # 合并字数
                chapter.allocated_words = single_child.allocated_words

                # 更新 children_ids
                chapter.children_ids = [cid for cid in chapter.children_ids if cid != single_child.node_id]

                # 从注册表中移除
                if single_child.node_id in session.node_registry:
                    del session.node_registry[single_child.node_id]

                optimized_count += 1

        if self.enable_logging:
            print_and_flush(f"    Total optimized: {optimized_count} chapters")

    def _generate_split_data(self, session, node: OutlineNode) -> dict:
        """
        使用 LLM 生成节点拆分数据

        Args:
            session: 会话实例
            node: 待拆分的节点

        Returns:
            拆分数据字典，包含 new_nodes 列表；失败返回 None
        """
        from typing import Optional

        lang = self.prompts._language
        # 目标：每个新节点不超过 MAX_LEAF_WORDS
        num_splits = max(2, (node.allocated_words + MAX_LEAF_WORDS - 1) // MAX_LEAF_WORDS)
        target_words_per_node = node.allocated_words // num_splits

        # 使用现有的 PROMPT_5_SPLIT 模板
        prompt = self.prompts.get_template("PROMPT_5_SPLIT", lang).format(
            node_title=node.title,
            node_description=node.description or "",
            current_words=node.allocated_words,
            target_words_per_node=target_words_per_node
        )

        # 动态替换拆分数量
        import re
        if lang == "zh":
            prompt = re.sub(r'拆分为 \d+-\d+ 个', f'拆分为 {num_splits} 个', prompt)
        else:
            prompt = re.sub(r'split into \d+-\d+ ', f'split into {num_splits} ', prompt)

        try:
            response = self.llm.call(
                prompt,
                temperature=get_temperature("word_allocation", "split")
            )

            result = robust_json_parse(response, dict)

            if result and "new_nodes" in result:
                # 验证拆分数量
                actual_splits = len(result["new_nodes"])
                if actual_splits < num_splits:
                    if self.enable_logging:
                        print_and_flush(f"      [Warning] LLM returned {actual_splits} nodes, expected {num_splits}")
                return result

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"      Split generation error: {e}")

        return None