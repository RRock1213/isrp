"""
Phase III — 描述优化

在 DFS 扩展完成后，批量精修所有节点的章节描述。
优化目标：
  - specificity >= 8：明确指出要写的具体内容
  - guidance >= 8：提供清晰可执行的写作方向
  - 去除模板化语言（"需涵盖"、"应包括"等），融入用户具体信息
分批处理（每批 4 节点），确保描述质量一致。

"""
from typing import List, Dict, Any
from ..schemas import OutlineNode, PlanSession, robust_json_parse
from ..utils.llm_client import LLMClient
from ..prompts import PromptTemplates
from ..engines import get_temperature

from . import print_and_flush, flush_stdout


class DescriptionOptimize:
    """Description Optimize"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        enable_logging: bool = True
    ):
        self.llm = llm_client
        self.prompts = prompts
        self.enable_logging = enable_logging

    def execute(
        self,
        root: OutlineNode,
        session: PlanSession
    ) -> OutlineNode:
        """
        执行描述优化

        Args:
            root: 根节点
            session: 会话实例

        Returns:
            优化后的根节点
        """
        if self.enable_logging:
            print_and_flush("\n[Description Optimize] Description Optimization")
            print_and_flush("=" * 60)

        # 收集所有节点
        all_nodes = self._collect_nodes(root, session)

        if not all_nodes:
            if self.enable_logging:
                print_and_flush("  No nodes to optimize")
            return root

        # 按父节点分组
        batches = self._group_by_parent(all_nodes)

        if self.enable_logging:
            print_and_flush(f"  Total nodes: {len(all_nodes)}, Batches: {len(batches)}")

        # 分批处理
        optimized_count = 0
        for batch in batches:
            try:
                optimized = self._optimize_batch(batch, session)
                optimized_count += optimized
            except Exception as e:
                if self.enable_logging:
                    print_and_flush(f"  [Warning] Batch optimization failed: {e}")
                continue

        if self.enable_logging:
            print_and_flush(f"  Optimized {optimized_count}/{len(all_nodes)} descriptions")

        return root

    def _collect_nodes(
        self,
        root: OutlineNode,
        session: PlanSession
    ) -> List[OutlineNode]:
        """收集所有节点（广度优先）"""
        nodes = []
        queue = [root]

        while queue:
            node = queue.pop(0)
            if node.level > 0:  # 跳过根节点
                nodes.append(node)

            # 添加子节点到队列
            for child_id in node.children_ids:
                child = session.get_node(child_id)
                if child:
                    queue.append(child)

        return nodes

    def _group_by_parent(
        self,
        nodes: List[OutlineNode]
    ) -> List[List[OutlineNode]]:
        """按父节点分组，每批最多 4 个节点"""
        groups: Dict[str, List[OutlineNode]] = {}

        for node in nodes:
            parent_key = node.parent_id or "root"
            if parent_key not in groups:
                groups[parent_key] = []
            groups[parent_key].append(node)

        # 将大组拆分为小批次
        batches = []
        for group_nodes in groups.values():
            for i in range(0, len(group_nodes), 4):
                batches.append(group_nodes[i:i+4])

        return batches

    def _optimize_batch(
        self,
        nodes: List[OutlineNode],
        session: PlanSession
    ) -> int:
        """
        批量优化节点描述

        Returns:
            优化成功的节点数
        """
        if not nodes:
            return 0

        # 构建优化 Prompt
        prompt = self._build_optimization_prompt(nodes, session)

        try:
            response = self.llm.call(
                prompt,
                temperature=get_temperature("word_allocation", "reorganize")
            )

            # 解析结果
            optimizations = self._parse_optimization_result(response)

            # 应用优化
            optimized = 0
            for node in nodes:
                # 尝试匹配标题（可能包含编号）
                clean_title = node.title
                for title_key, new_desc in optimizations.items():
                    if title_key in clean_title or clean_title in title_key:
                        if new_desc and len(new_desc) > 20:
                            node.description = new_desc
                            optimized += 1
                        break

            return optimized

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"  [Error] Optimization failed: {e}")
            return 0

    def _build_optimization_prompt(
        self,
        nodes: List[OutlineNode],
        session: PlanSession
    ) -> str:
        """构建优化 Prompt"""
        user_prompt = session.get_effective_prompt()

        # 格式化节点信息
        nodes_info = []
        for node in nodes:
            nodes_info.append(f"""标题: {node.title}
当前描述: {node.description or "无"}
层级: {node.level}""")

        nodes_str = "\n\n".join(nodes_info)

        lang = self.prompts._language

        from ..prompts import PromptTemplates
        if lang == "en":
            return PromptTemplates.PROMPT_DESC_OPTIMIZE_EN.format(
                user_prompt=user_prompt, nodes_str=nodes_str)

        return PromptTemplates.PROMPT_DESC_OPTIMIZE_ZH.format(
            user_prompt=user_prompt, nodes_str=nodes_str)

    def _parse_optimization_result(
        self,
        response: str
    ) -> Dict[str, str]:
        """解析优化结果"""
        try:
            result = robust_json_parse(response, dict)
            return result if result else {}
        except Exception:
            return {}