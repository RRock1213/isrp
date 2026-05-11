"""
ISRP — 节点操作工具

提供大纲节点的常用操作：
  - merge_two_nodes:      合并两个同层级节点
  - split_node:           将一个节点拆分为多个子节点
  - condense_children:    浓缩子节点（合并冗余）
  - smart_merge_with_llm: 使用 LLM 语义理解进行智能合并

"""
import re
from typing import List, Dict, Optional, Tuple

from ..schemas import (
    OutlineNode, NodeStatus, generate_node_id, robust_json_parse
)
from .llm_client import LLMClient

class NodeOperations:
    """节点操作工具集"""

    @staticmethod
    def merge_two_nodes(
        session,
        target: OutlineNode,
        source: OutlineNode,
        llm_client: LLMClient = None,
        prompts = None,
        enable_logging: bool = True
    ) -> None:
        """
        将 source 节点合并到 target 节点

        Args:
            session: 会话实例
            target: 合并目标节点
            source: 被合并的源节点
            llm_client: LLM 客户端（用于智能合并）
            prompts: Prompt 模板实例
            enable_logging: 是否启用日志
        """
        use_llm = llm_client is not None and prompts is not None

        if use_llm:
            # 使用 LLM 进行智能合并
            merged_title, merged_desc, merged_concepts = NodeOperations._smart_merge_with_llm(
                llm_client, prompts, target, source, enable_logging
            )

            if merged_title:
                target.title = merged_title
                target.description = merged_desc
                if merged_concepts:
                    target.core_concepts = merged_concepts
            else:
                # LLM 合并失败，使用备用策略
                target.title = NodeOperations._create_fallback_merge_title(target.title, source.title)
                target.description = f"{target.description}\n\n{source.description}" if target.description else source.description
        else:
            # 硬合并（不推荐，仅作为备用）
            target.title = NodeOperations._create_fallback_merge_title(target.title, source.title)
            target.description = f"{target.description}\n\n{source.description}" if target.description else source.description

        # 合并字数
        target.allocated_words += source.allocated_words

        # 合并核心概念
        if source.core_concepts:
            if target.core_concepts:
                target.core_concepts = list(set(target.core_concepts + source.core_concepts))
            else:
                target.core_concepts = source.core_concepts

        # 合并子节点
        source_children = session.get_children(source.node_id)
        for child in source_children:
            child.parent_id = target.node_id
            target.children_ids.append(child.node_id)

        # 从父节点中移除 source
        if source.parent_id:
            parent = session.get_node(source.parent_id)
            if parent:
                parent.children_ids = [cid for cid in parent.children_ids if cid != source.node_id]

        # 从注册表中移除 source
        if source.node_id in session.node_registry:
            del session.node_registry[source.node_id]

    @staticmethod
    def split_node(
        session,
        node: OutlineNode,
        new_nodes_data: List[Dict]
    ) -> None:
        """
        拆分节点为多个子节点

        【 重构】拆分后保留原节点，新节点作为原节点的子节点
        - 符合 LLM 对"拆分=展开子节点"的理解
        - 清理标题中的编号前缀（如"2.1"、"2.2"）

        【字数守恒】确保拆分后总字数等于原节点字数
        【】确保每个新节点字数不超过 MAX_LEAF_WORDS

        Args:
            session: 会话实例
            node: 待拆分的节点
            new_nodes_data: 新节点数据列表
        """
        from ..engines import MAX_LEAF_WORDS

        original_words = node.allocated_words

        # 清理标题中的编号前缀的辅助函数
        def clean_title_prefix(title: str) -> str:
            """移除标题中的编号前缀（如"2.1"、"2.2"、"一、"等）"""
            patterns = [
                r'^[\d.]+\s+',           # "2.1 " "3.1.2 "
                r'^[一二三四五六七八九十]+[、.．]\s*',  # "一、" "二."
                r'^第[一二三四五六七八九十\d]+[章部节]\s*',  # "第一章" "第1章"
            ]
            cleaned = title
            for pattern in patterns:
                cleaned = re.sub(pattern, '', cleaned)
            return cleaned.strip()

        new_nodes = []
        allocated_total = 0

        for i, data in enumerate(new_nodes_data):
            # 最后一个节点吸收残差，确保字数守恒
            if i == len(new_nodes_data) - 1:
                word_count = original_words - allocated_total
            else:
                # 使用建议的字数，但不超过剩余可用字数
                suggested = data.get("allocated_words", original_words // len(new_nodes_data))
                remaining = original_words - allocated_total
                word_count = min(suggested, remaining - 1)  # 至少留1字给最后一个节点
            if word_count > MAX_LEAF_WORDS:
                word_count = MAX_LEAF_WORDS
            raw_title = data.get("title", f"{node.title} ({i+1})")
            clean_title = clean_title_prefix(raw_title)
            new_node = OutlineNode(
                node_id=generate_node_id(),
                title=clean_title,
                description=data.get("description", ""),
                level=node.level + 1,  # 修复：子节点层级 +1
                status=NodeStatus.ACCEPTED,
                parent_id=node.node_id,  # 修复：父节点是原节点
                is_finalized=True,
                allocated_words=word_count,
                core_concepts=data.get("core_concepts", node.core_concepts)
            )
            session.register_node(new_node)
            new_nodes.append(new_node)
            allocated_total += word_count
        remaining_words = original_words - allocated_total
        if remaining_words > 0:
            extra_node = OutlineNode(
                node_id=generate_node_id(),
                title=f"{node.title} (补充)",
                description="",
                level=node.level + 1,  # 修复：子节点层级
                status=NodeStatus.ACCEPTED,
                parent_id=node.node_id,  # 修复：父节点是原节点
                is_finalized=True,
                allocated_words=remaining_words,
                core_concepts=node.core_concepts
            )
            session.register_node(extra_node)
            new_nodes.append(extra_node)
        # 原节点保留其标题和描述，新节点成为其子节点
        node.children_ids = [n.node_id for n in new_nodes]
        # 原节点字数保持不变（作为父节点）

        # 注意：不再删除原节点，保持层级结构完整

    @staticmethod
    def condense_children(
        session,
        node: OutlineNode,
        llm_client: LLMClient,
        prompts,
        enable_logging: bool = True
    ) -> None:
        """
        浓缩子节点（多层压缩）

        通过 LLM 将多层子节点压缩为更紧凑的结构

        Args:
            session: 会话实例
            node: 父节点
            llm_client: LLM 客户端
            prompts: Prompt 模板实例
            enable_logging: 是否启用日志
        """
        children = session.get_children(node.node_id)
        if not children:
            return

        if enable_logging:
            print(f"      [Condense] Condensing {len(children)} children of {node.title}")

        # 构建浓缩 Prompt
        lang = prompts._language
        children_desc = "\n".join([
            f"  - {child.title}: {child.description[:200] if child.description else '无描述'}..."
            for child in children
        ])

        prompt = prompts.get_template("PROMPT_5_CONDENSE", lang).format(
            parent_title=node.title,
            parent_description=node.description or "无描述",
            children_list=children_desc,
            target_words=node.allocated_words
        )

        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = llm_client.call(prompt, temperature=0.3)
                result = robust_json_parse(response, dict)

                # 处理浓缩结果
                action = result.get("action", "keep")
                if action == "merge_children":
                    # 合并指定的子节点
                    merged_ids = result.get("merged_node_ids", [])
                    new_title = result.get("new_title", "")
                    new_description = result.get("new_description", "")

                    if merged_ids and new_title:
                        NodeOperations._merge_children_nodes(
                            session, node, merged_ids, new_title, new_description
                        )
                        if enable_logging:
                            print(f"        Merged {len(merged_ids)} children into: {new_title}")

                return  # 成功完成，退出重试循环

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    if enable_logging:
                        print(f"        [Retry {attempt + 1}/{max_retries}] Condense error: {e}")
                    continue
                else:
                    if enable_logging:
                        print(f"        [Warning] Condense error after {max_retries + 1} attempts: {e}")

    @staticmethod
    def _merge_children_nodes(
        session,
        parent: OutlineNode,
        merged_ids: List[str],
        new_title: str,
        new_description: str
    ) -> None:
        """
        合并指定的子节点

        【改进】保持合并后节点的相对位置，避免编号错乱

        Args:
            session: 会话实例
            parent: 父节点
            merged_ids: 待合并的子节点 ID 列表
            new_title: 新节点标题
            new_description: 新节点描述
        """
        # 创建新节点
        new_node = OutlineNode(
            node_id=generate_node_id(),
            title=new_title,
            description=new_description,
            level=parent.level + 1,
            status=NodeStatus.ACCEPTED,
            parent_id=parent.node_id,
            is_finalized=True
        )

        # 计算合并后的字数
        total_words = 0
        all_concepts = []
        for child_id in merged_ids:
            child = session.get_node(child_id)
            if child:
                total_words += child.allocated_words
                if child.core_concepts:
                    all_concepts.extend(child.core_concepts)

        new_node.allocated_words = total_words
        new_node.core_concepts = list(set(all_concepts))

        # 注册新节点
        session.register_node(new_node)

        # 更新父节点的 children_ids（保持位置）
        # 在第一个被合并节点的位置插入新节点
        new_children = []
        new_node_inserted = False
        for child_id in parent.children_ids:
            if child_id in merged_ids:
                # 在第一个被合并节点的位置插入新节点
                if not new_node_inserted:
                    new_children.append(new_node.node_id)
                    new_node_inserted = True
                # 跳过被合并的节点
            else:
                new_children.append(child_id)

        # 如果新节点还未插入（边界情况），追加到末尾
        if not new_node_inserted:
            new_children.append(new_node.node_id)

        parent.children_ids = new_children

        # 从注册表中移除被合并的节点
        for child_id in merged_ids:
            if child_id in session.node_registry:
                del session.node_registry[child_id]

    @staticmethod
    def _smart_merge_with_llm(
        llm_client: LLMClient,
        prompts,
        target: OutlineNode,
        source: OutlineNode,
        enable_logging: bool = True
    ) -> Tuple[str, str, List[str]]:
        """
        使用 LLM 智能合并两个节点

        Args:
            llm_client: LLM 客户端
            prompts: Prompt 模板实例
            target: 目标节点
            source: 源节点
            enable_logging: 是否启用日志

        Returns:
            (new_title, new_description, new_concepts) 或 (None, None, None) 如果失败
        """
        lang = prompts._language
        prompt = prompts.get_template("PROMPT_5_SMART_MERGE", lang).format(
            title1=target.title,
            description1=target.description or "无描述",
            concepts1=", ".join(target.core_concepts) if target.core_concepts else "无",
            title2=source.title,
            description2=source.description or "无描述",
            concepts2=", ".join(source.core_concepts) if source.core_concepts else "无"
        )

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = llm_client.call(prompt, temperature=0.3)
                if not response or not response.strip():
                    continue

                result = robust_json_parse(response, dict)

                new_title = result.get("new_title", "")
                new_desc = result.get("new_description", "")
                new_concepts = result.get("new_concepts", [])

                if new_title and new_desc:
                    return new_title, new_desc, new_concepts

            except Exception as e:
                if enable_logging and attempt < max_retries:
                    print(f"        [Retry {attempt + 1}/{max_retries}] Smart merge error: {e}")
                continue

        # 所有重试都失败
        if enable_logging:
            print(f"        [Warning] Smart merge failed, using fallback")
        return None, None, None

    @staticmethod
    def _create_fallback_merge_title(title1: str, title2: str) -> str:
        """
        创建备用的合并标题（不使用 & 符号）

        策略：提取两个标题的核心词，生成新的整合标题

        Args:
            title1: 第一个标题
            title2: 第二个标题

        Returns:
            合并后的标题
        """
        # 移除编号前缀
        clean1 = re.sub(r'^[\d.]+\s*', '', title1).strip()
        clean2 = re.sub(r'^[\d.]+\s*', '', title2).strip()

        # 简单策略：取较短的标题作为基础，或者直接拼接（使用顿号）
        if len(clean1) < 20 and len(clean2) < 20:
            return f"{clean1}与{clean2}"
        elif len(clean1) < len(clean2):
            return f"{clean1}：整合与延伸"
        else:
            return f"{clean2}：整合与延伸"

    @staticmethod
    def merge_with_sibling(
        session,
        node: OutlineNode,
        llm_client: LLMClient = None,
        prompts = None,
        enable_logging: bool = True
    ) -> None:
        """
        与兄弟节点合并

        Args:
            session: 会话实例
            node: 待合并的节点
            llm_client: LLM 客户端
            prompts: Prompt 模板实例
            enable_logging: 是否启用日志
        """
        if not node.parent_id:
            return

        parent = session.get_node(node.parent_id)
        if not parent:
            return

        siblings = session.get_children(node.parent_id)
        if len(siblings) <= 1:
            return

        # 找到相邻的兄弟节点
        node_idx = next((i for i, s in enumerate(siblings) if s.node_id == node.node_id), -1)
        if node_idx == -1:
            return

        # 选择合并目标（优先选择字数也较少的相邻节点）
        merge_target = None
        if node_idx > 0:
            merge_target = siblings[node_idx - 1]
        elif node_idx < len(siblings) - 1:
            merge_target = siblings[node_idx + 1]

        if merge_target:
            from ..engines import MAX_LEAF_WORDS
            merged_words = merge_target.allocated_words + node.allocated_words
            if merged_words > MAX_LEAF_WORDS:
                # 合并后会超过上限，跳过此次合并
                if enable_logging:
                    print(f"        [Skip] Merging '{node.title}' would exceed limit ({merged_words} > {MAX_LEAF_WORDS})")
                return

            # 执行合并
            NodeOperations.merge_two_nodes(
                session, merge_target, node,
                llm_client, prompts, enable_logging
            )
            if enable_logging:
                print(f"        Merged '{node.title}' into '{merge_target.title}'")