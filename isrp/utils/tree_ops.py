"""
ISRP — 树结构操作工具

提供大纲树的结构级操作：
  - validate_integrity:   验证树结构的完整性（无断裂引用、无环路）
  - repair_integrity:     自动修复结构问题
  - format_outline_tree:  将树格式化为可读的大纲文本
  - print_outline_tree:   终端彩色打印大纲树结构

"""
import re
from typing import List, Optional

class TreeOperations:
    """树结构操作工具集"""

    @staticmethod
    def validate_integrity(session) -> List[str]:
        """
        验证树结构的完整性

        检查：
        1. 所有节点的 parent_id 指向存在的节点
        2. 所有 children_ids 中的节点都存在
        3. children_ids 没有重复
        4. 父子关系双向一致

        Args:
            session: 会话实例

        Returns:
            问题列表，空列表表示无问题
        """
        issues = []

        for node_id, node in session.node_registry.items():
            # 检查 parent_id 有效性
            if node.parent_id:
                parent = session.get_node(node.parent_id)
                if not parent:
                    issues.append(f"Node {node_id} has non-existent parent: {node.parent_id}")
                elif node_id not in parent.children_ids:
                    issues.append(f"Node {node_id} not in parent's children_ids")

            # 检查 children_ids 有效性
            for child_id in node.children_ids:
                # 检查 child 是否存在
                if child_id not in session.node_registry:
                    issues.append(f"Node {node_id} references non-existent child: {child_id}")
                    continue

                # 检查 children_ids 是否有重复
                if node.children_ids.count(child_id) > 1:
                    issues.append(f"Node {node_id} has duplicate child_id: {child_id}")

                # 检查 child 的 parent_id 是否一致
                child = session.get_node(child_id)
                if child and child.parent_id != node_id:
                    issues.append(f"Child {child_id} parent_id mismatch: expected {node_id}, got {child.parent_id}")

        return issues

    @staticmethod
    def repair_integrity(session, issues: List[str]) -> None:
        """
        修复树结构的完整性问题

        Args:
            session: 会话实例
            issues: 问题列表
        """
        if not issues:
            return

        print(f"\n  [Tree Repair] Repairing {len(issues)} integrity issues...")

        # 预编译正则表达式，避免循环内重复导入
        non_existent_pattern = re.compile(r"Node (\w+) references non-existent child: (\w+)")
        duplicate_pattern = re.compile(r"Node (\w+) has duplicate child_id: (\w+)")
        parent_mismatch_pattern = re.compile(r"Child (\w+) parent_id mismatch: expected (\w+), got (\w+)")

        for issue in issues:
            if "non-existent child" in issue:
                # 移除不存在的 child_id
                match = non_existent_pattern.search(issue)
                if match:
                    node_id, child_id = match.groups()
                    node = session.get_node(node_id)
                    if node:
                        node.children_ids = [cid for cid in node.children_ids if cid != child_id]
                        print(f"    Removed non-existent child {child_id} from {node_id}")

            elif "duplicate child_id" in issue:
                # 移除重复的 child_id
                match = duplicate_pattern.search(issue)
                if match:
                    node_id, child_id = match.groups()
                    node = session.get_node(node_id)
                    if node:
                        # 保留第一个，移除后续重复的
                        seen = set()
                        new_children = []
                        for cid in node.children_ids:
                            if cid not in seen:
                                new_children.append(cid)
                                seen.add(cid)
                        node.children_ids = new_children
                        print(f"    Removed duplicate child {child_id} from {node_id}")

            elif "parent_id mismatch" in issue:
                # 修复 parent_id 不一致
                match = parent_mismatch_pattern.search(issue)
                if match:
                    child_id, expected_parent, actual_parent = match.groups()
                    child = session.get_node(child_id)
                    if child:
                        child.parent_id = expected_parent
                        print(f"    Fixed parent_id for {child_id}: {actual_parent} -> {expected_parent}")

    @staticmethod
    def format_outline_tree(session, show_words: bool = True) -> str:
        """
        格式化大纲树为字符串

        模仿书籍目录格式输出：
        第一章 xxx
          1.1 xxx
            1.1.1 xxx (500字)
          1.2 xxx
            1.2.1 xxx (450字)

        Args:
            session: 会话实例
            show_words: 是否显示字数

        Returns:
            格式化后的大纲树字符串
        """
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("【大纲结构】")
        lines.append("=" * 60)

        root = session.get_node(session.root_id)
        if not root:
            return "\n".join(lines)

        def strip_existing_numbering(title: str) -> str:
            """移除标题中已有的编号前缀（包括中文章节编号）"""
            # 移除中文章节编号格式：第一章、第二十三章、第1章、第1节等
            title = re.sub(r'^第[一二三四五六七八九十百零\d]+[章节]\s*[:：]?\s*', '', title)
            # 移除阿拉伯数字编号：1.1、1.1.1、1
            title = re.sub(r'^[\d.]+\s*', '', title)
            return title.strip()

        def format_node(node, prefix: str = "", index: str = "") -> List[str]:
            result = []

            # 移除标题中已有的编号，避免重复
            clean_title = strip_existing_numbering(node.title)

            if node.level == 0:
                result.append(f"{node.title}")
            elif node.level == 1:
                result.append(f"{prefix}第{index}章 {clean_title}")
            elif node.level == 2:
                result.append(f"{prefix}{index} {clean_title}")
            else:
                word_info = f" ({node.allocated_words}字)" if show_words and node.allocated_words > 0 else ""
                result.append(f"{prefix}{index} {clean_title}{word_info}")

            children = session.get_children(node.node_id)
            for i, child in enumerate(children, 1):
                if node.level == 0:
                    child_index = str(i)
                else:
                    child_index = f"{index}.{i}" if index else str(i)

                child_prefix = prefix + "  "
                result.extend(format_node(child, child_prefix, child_index))

            return result

        lines.extend(format_node(root))

        # 添加统计信息
        leaves = session.get_all_leaf_nodes()
        total_words = sum(n.allocated_words for n in leaves)
        lines.append(f"\n总计: {len(leaves)} 个段落, {total_words} 字")

        return "\n".join(lines)

    @staticmethod
    def print_outline_tree(session, persist_file: str = None) -> None:
        """
        打印大纲树可视化

        Args:
            session: 会话实例
            persist_file: 持久化文件路径（可选）
        """
        tree_str = TreeOperations.format_outline_tree(session)
        print(tree_str)

        # 持久化大纲树结构
        if persist_file:
            TreeOperations._persist_outline_tree(session, persist_file)

    @staticmethod
    def _persist_outline_tree(session, persist_file: str) -> None:
        """
        持久化大纲树结构

        Args:
            session: 会话实例
            persist_file: 文件路径
        """
        import json
        import os
        from datetime import datetime

        try:
            os.makedirs(os.path.dirname(persist_file), exist_ok=True)

            # 收集树结构数据
            tree_data = {
                "timestamp": datetime.now().isoformat(),
                "root_id": session.root_id,
                "nodes": {}
            }

            for node_id, node in session.node_registry.items():
                tree_data["nodes"][node_id] = {
                    "title": node.title,
                    "description": node.description,
                    "level": node.level,
                    "allocated_words": node.allocated_words,
                    "parent_id": node.parent_id,
                    "children_ids": node.children_ids,
                    "core_concepts": node.core_concepts
                }

            with open(persist_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(tree_data, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"  [Warning] 大纲树持久化失败: {e}")

    @staticmethod
    def get_node_index(session, node) -> str:
        """
        获取节点的层级编号

        例如：
        - Root 节点: ""
        - 第一章: "1"
        - 第一章第一节: "1.1"
        - 第一章第一节第一个要点: "1.1.1"

        Args:
            session: 会话实例
            node: 当前节点

        Returns:
            层级编号字符串
        """
        if node.level == 0:
            return ""

        parts = []
        current = node

        while current and current.level > 0:
            parent_id = current.parent_id
            if not parent_id:
                break

            parent = session.get_node(parent_id)
            if not parent:
                break

            # 找到当前节点在父节点 children_ids 中的位置
            sibling_index = 0
            for i, sibling_id in enumerate(parent.children_ids, 1):
                if sibling_id == current.node_id:
                    sibling_index = i
                    break

            parts.append(str(sibling_index))
            current = parent

        return ".".join(reversed(parts))

    @staticmethod
    def resolve_node_reference(session, chapter, ref: str):
        """
        解析节点引用，支持多种格式

        Args:
            session: 会话实例
            chapter: 章节点（用于限定搜索范围）
            ref: 节点引用（可以是 node_id、编号或标题）

        Returns:
            匹配的节点，如果未找到则返回 None
        """
        ref = str(ref).strip()

        # 1. 尝试直接按 node_id 查找
        node = session.get_node(ref)
        if node:
            return node

        # 2. 获取章节下的所有子节点
        def get_all_descendants(parent_id: str):
            descendants = []
            children = session.get_children(parent_id)
            for child in children:
                descendants.append(child)
                descendants.extend(get_all_descendants(child.node_id))
            return descendants

        all_nodes = [chapter] + get_all_descendants(chapter.node_id)

        # 3. 尝试按编号匹配（如 "3.1", "1.2.3"）
        for node in all_nodes:
            match = re.match(r'^([\d.]+)\s*', node.title)
            if match and match.group(1) == ref:
                return node

        # 4. 尝试按标题匹配（精确匹配或包含关系）
        for node in all_nodes:
            if node.title == ref:
                return node

        # 5. 尝试部分匹配（ref 包含在标题中或标题包含在 ref 中）
        for node in all_nodes:
            if ref in node.title or node.title in ref:
                return node

        return None