"""
Phase II — 章节级检查点验证

在一级章节的所有子节点生成完毕后即时触发，评估章节整体质量并应用补丁修复。
检查内容包括：
  - 结构完整性：子节点排列是否合理、是否存在孤立的单节点层级
  - 内容质量：子节点内容的评分、是否存在冗余或遗漏
  - 补丁操作：重新排序、添加过渡、修改属性、移除冗余、重组结构、调换位置
与全局验证形成两级机制：章节级（即时纠错）+ 全局级（终验保障）。

"""
import sys
from typing import Dict, List, Any, Optional
import re

from ..schemas import OutlineNode, NodeStatus, generate_node_id, robust_json_parse
from ..utils.llm_client import LLMClient
from ..utils.tree_ops import TreeOperations
from ..prompts import PromptTemplates
from ..engines import get_temperature

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout

# 类型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..evaluators.base import BaseEvaluator, EvalType

class ChapterLevelValidation:
    """Chapter-Level Validation"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        enable_logging: bool = True,
        evaluator: 'BaseEvaluator' = None
    ):
        self.llm = llm_client
        self.prompts = prompts
        self.enable_logging = enable_logging
        self.evaluator = evaluator

    def execute(
        self,
        chapter: OutlineNode,
        session,
        user_prompt: str,
        prev_chapter_abstracts: List[str] = None
    ) -> Dict[str, Any]:
        """
        执行 Checkpoint

        Args:
            chapter: 章节点
            session: 会话实例
            user_prompt: 用户指令
            prev_chapter_abstracts: 前序章节摘要列表

        Returns:
            Checkpoint 结果字典
        """
        if self.enable_logging:
            print_and_flush(f"\n  [Chapter Validation] Evaluating chapter: {chapter.title}")

        # 结构健康检查：检测结论位置异常等问题
        structure_patch = self._check_structure_health(session, chapter)

        # 收集章节内容
        chapter_content = self._collect_chapter_content(session, chapter)

        # 获取风格上下文
        style_context = session.get_style_context() if hasattr(session, 'get_style_context') else ""
        global_abstract = session.global_abstract or ""
        prev_abstracts = "\n".join(prev_chapter_abstracts) if prev_chapter_abstracts else "无"

        # 构建 Prompt - 使用 effective_prompt（已由 planner 处理压缩或原始）
        lang = self.prompts._language
        effective_prompt = session.get_effective_prompt()

        # 使用 evaluator 或默认 prompt
        if self.evaluator:
            if self.enable_logging:
                print_and_flush(f"    [Using {type(self.evaluator).__name__}]")
            prompt = self.evaluator.get_checkpoint_prompt(
                user_prompt=effective_prompt,
                style_context=style_context,
                global_abstract=global_abstract,
                prev_chapter_abstract=prev_abstracts,
                chapter_content=chapter_content,
                language=lang
            )
        else:
            prompt = self.prompts.get_template("PROMPT_4_CHECKPOINT", lang).format(
                user_prompt=effective_prompt,
                style_context=style_context,
                global_abstract=global_abstract,
                prev_chapter_abstract=prev_abstracts,
                chapter_content=chapter_content
            )

        # 带重试机制的评估
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.call(
                    prompt,
                    temperature=get_temperature("chapter_checkpoint", "checkpoint")
                )

                if not response or not response.strip():
                    if attempt < max_retries:
                        if self.enable_logging:
                            print_and_flush(f"    [Retry {attempt + 1}/{max_retries}] Empty checkpoint response")
                        continue
                    return self._default_checkpoint_result(chapter)

                # 使用 evaluator 解析结果
                if self.evaluator:
                    from ..evaluators.base import EvalType
                    eval_result = self.evaluator.parse_result(response, EvalType.CHECKPOINT)
                    result = eval_result.extra
                else:
                    result = robust_json_parse(response, dict)

                break

            except Exception as e:
                if attempt < max_retries:
                    if self.enable_logging:
                        print_and_flush(f"    [Retry {attempt + 1}/{max_retries}] Checkpoint error: {str(e)[:80]}...")
                    continue
                else:
                    if self.enable_logging:
                        print_and_flush(f"    Checkpoint error after {max_retries + 1} attempts: {e}")
                    return self._default_checkpoint_result(chapter)
        else:
            # 所有重试都没有返回
            return self._default_checkpoint_result(chapter)

        # 应用补丁
        patches = result.get("patches", [])

        # 如果有结构问题，合并到patches中
        if structure_patch:
            existing_patches = patches or []
            existing_patches.insert(0, structure_patch)  # 优先处理结构问题
            patches = existing_patches
            if self.enable_logging:
                print_and_flush(f"    [Structure Alert] {structure_patch.get('reason', 'Structure issue detected')}")

        if patches:
            self._apply_patches(session, chapter, patches)

        # 记录章节摘要
        chapter_abstract = result.get("chapter_abstract", chapter.description or chapter.title)

        if self.enable_logging:
            # 支持多种评估器返回格式
            if "average_score" in result:
                avg_score = float(result["average_score"])
            elif "evaluation" in result and result["evaluation"]:
                avg_score = self._calculate_avg_score(result["evaluation"])
            elif "scores" in result and result["scores"]:
                scores = result["scores"]
                avg_score = sum(scores.values()) / len(scores) if scores else 8.0
            else:
                avg_score = 8.0
            print_and_flush(f"    Score: {avg_score:.2f}")
            if patches:
                print_and_flush(f"    Applied {len(patches)} patches")

        return result

    def _collect_chapter_content(self, session, chapter: OutlineNode) -> str:
        """收集章节内容"""
        parts = [f"[{chapter.title}] (ID: {chapter.node_id})", chapter.description or ""]

        def collect_children(node: OutlineNode, depth: int = 0):
            children = session.get_children(node.node_id)
            for child in children:
                indent = "  " * depth
                parts.append(f"{indent}[{child.title}] (ID: {child.node_id})")
                parts.append(f"{indent}{child.description or ''}")
                collect_children(child, depth + 1)

        collect_children(chapter)
        return "\n".join(parts)

    def _apply_patches(self, session, chapter: OutlineNode, patches: List[Dict]) -> None:
        """
        应用补丁

        支持三种字段名：
        - action: 标准字段 (reorder/add_bridge/modify/remove/restructure/swap)
        - operation: LLM 可能返回的替代字段 (expand/insert_after/modify/remove)
        - type: LLM 可能返回的另一种字段名（content_insertion/content_expansion/symbolic_reinforcement）

        新增类型支持：
        - content_insertion: 在指定位置插入内容段落 -> 映射到 add_bridge
        - content_expansion: 扩展节点内容 -> 映射到 modify
        - symbolic_reinforcement: 添加过渡/象征性内容 -> 映射到 add_bridge
        """
        # operation/type 到 action 的映射
        OPERATION_MAP = {
            "expand": "modify",        # expand 视为 modify 的变体
            "insert_after": "add_bridge",  # insert_after 视为 add_bridge 的变体
            "insert_before": "add_bridge",
            "merge": "restructure",
            "split": "restructure",
            "delete": "remove",
        }

        # type 字段到 action 的映射
        TYPE_TO_ACTION_MAP = {
            "content_insertion": "add_bridge",      # 插入内容段落
            "content_expansion": "modify",          # 扩展节点内容
            "symbolic_reinforcement": "add_bridge", # 添加过渡内容
            "structural_adjustment": "restructure", # 结构调整
        }

        for patch in patches:
            # 优先使用 action，其次 operation，最后 type
            action = patch.get("action", "") or patch.get("operation", "") or patch.get("type", "")

            # 跳过空 action
            if not action:
                if self.enable_logging:
                    print_and_flush(f"    [Warning] Patch missing action/operation/type field, skipping: {str(patch)}...")
                continue

            # 记录原始类型用于日志
            original_action = action

            # 如果是 operation 特有值，映射到标准 action
            if action in OPERATION_MAP:
                action = OPERATION_MAP[action]
                if self.enable_logging:
                    print_and_flush(f"    [Patch] Mapped operation '{original_action}' -> action '{action}'")

            # 如果是 type 特有值，映射到标准 action
            elif action in TYPE_TO_ACTION_MAP:
                action = TYPE_TO_ACTION_MAP[action]
                if self.enable_logging:
                    print_and_flush(f"    [Patch] Mapped type '{original_action}' -> action '{action}'")

            # 检测未知类型并提供更清晰的警告
            elif action not in ["reorder", "add_bridge", "modify", "remove", "restructure", "swap"]:
                if self.enable_logging:
                    # 提示 LLM 应使用标准类型
                    valid_types = "reorder, add_bridge, modify, remove, restructure, swap"
                    print_and_flush(f"    [Warning] Unknown patch type '{action}', valid types: {valid_types}")
                    print_and_flush(f"    [Warning] Skipping patch: {str(patch)}...")
                continue

            try:
                if action == "reorder":
                    self._patch_reorder(session, chapter, patch)
                elif action == "add_bridge":
                    self._patch_add_bridge(session, chapter, patch)
                elif action == "modify":
                    self._patch_modify(session, chapter, patch)
                elif action == "remove":
                    self._patch_remove(session, chapter, patch)
                elif action == "restructure":
                    self._patch_restructure(session, chapter, patch)
                elif action == "swap":
                    self._patch_swap(session, chapter, patch)
                else:
                    if self.enable_logging:
                        print_and_flush(f"    [Warning] Unknown patch action: {action}, skipping patch: {str(patch)}")
            except Exception as e:
                if self.enable_logging:
                    print_and_flush(f"    [Error] Patch {action} failed: {e}")

    def _resolve_node_reference(self, session, chapter: OutlineNode, ref: str) -> Optional[OutlineNode]:
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
        def get_all_descendants(parent_id: str) -> List[OutlineNode]:
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

        # 4. 尝试按标题匹配（精确匹配）
        for node in all_nodes:
            if node.title == ref:
                return node

        # 5. 尝试部分匹配（ref 包含在标题中或标题包含在 ref 中）
        for node in all_nodes:
            if ref in node.title or node.title in ref:
                return node

        return None

    def _check_structure_health(self, session, chapter: OutlineNode) -> Optional[Dict[str, Any]]:
        """
        检查章节结构健康度

        主要检查：
        1. 结论/总结类章节是否出现在中间位置

        Args:
            session: 会话实例
            chapter: 当前章节节点

        Returns:
            如果发现结构问题，返回补丁字典；否则返回 None
        """
        # 获取所有一级节点（章节）
        root = session.get_node(session.root_id)
        if not root:
            return None

        chapter_nodes = [session.get_node(cid) for cid in root.children_ids]
        chapter_nodes = [c for c in chapter_nodes if c]  # 过滤None

        # 结论类关键词
        conclusion_keywords = ['结论', '总结', '结语', 'conclusion', 'summary', 'concluding']

        # 检查当前章节是否是结论类
        chapter_title_lower = chapter.title.lower()
        is_conclusion = any(kw in chapter_title_lower for kw in conclusion_keywords)

        if is_conclusion:
            # 找到当前章节在所有章节中的位置
            current_idx = None
            for i, c in enumerate(chapter_nodes):
                if c.node_id == chapter.node_id:
                    current_idx = i
                    break

            if current_idx is not None and current_idx < len(chapter_nodes) - 1:
                # 结论出现在中间位置，需要结构重组
                return {
                    "action": "restructure",
                    "reason": f"结论章节'{chapter.title}'出现在位置 {current_idx + 1}/{len(chapter_nodes)}，应移至末尾",
                    "current_position": current_idx,
                    "target_position": len(chapter_nodes) - 1,
                    "node_id": chapter.node_id
                }

        return None

    def _patch_reorder(self, session, chapter: OutlineNode, patch: Dict) -> None:
        """
        补丁：调整子节点顺序

        Args:
            session: 会话实例
            chapter: 章节点
            patch: {"action": "reorder", "new_order": ["id1", "id2", ...]}

        【支持多种ID格式】
        1. 完整 node_id: "node_xxx"
        2. 节点编号: "1.1", "2.3" 等
        3. 节点标题（部分匹配）
        """
        new_order = patch.get("new_order", [])
        if not new_order:
            return

        # 获取当前子节点列表
        current_children = [session.get_node(cid) for cid in chapter.children_ids]
        current_children = [c for c in current_children if c]  # 过滤None

        # 构建映射：node_id -> index, title -> index, 编号 -> index
        node_id_to_idx = {c.node_id: i for i, c in enumerate(current_children)}
        title_to_idx = {c.title: i for i, c in enumerate(current_children)}

        # 构建编号映射（从标题提取编号，如 "3.1 标题" -> "3.1"）
        index_to_idx = {}
        for i, c in enumerate(current_children):
            match = re.match(r'^([\d.]+)\s*', c.title)
            if match:
                index_to_idx[match.group(1)] = i

        # 转换 new_order 为实际索引列表
        resolved_indices = []
        for item in new_order:
            item_str = str(item).strip()

            # 尝试按 node_id 匹配
            if item_str in node_id_to_idx:
                resolved_indices.append(node_id_to_idx[item_str])
            # 尝试按编号匹配
            elif item_str in index_to_idx:
                resolved_indices.append(index_to_idx[item_str])
            # 尝试按标题匹配（支持部分匹配）
            else:
                for title, idx in title_to_idx.items():
                    if item_str in title or title in item_str:
                        resolved_indices.append(idx)
                        break

        # 验证解析结果
        if len(resolved_indices) != len(new_order):
            if self.enable_logging:
                print_and_flush(f"    [Patch reorder] Cannot resolve all items, skipping. Resolved: {len(resolved_indices)}/{len(new_order)}")
            return

        if len(set(resolved_indices)) != len(resolved_indices):
            if self.enable_logging:
                print_and_flush(f"    [Patch reorder] Duplicate indices detected, skipping")
            return

        # 执行重排序
        new_children_ids = [chapter.children_ids[i] for i in resolved_indices]

        # 如果是部分重排序，保留未涉及的节点
        if len(new_children_ids) < len(chapter.children_ids):
            remaining = [cid for cid in chapter.children_ids if cid not in new_children_ids]
            new_children_ids.extend(remaining)

        chapter.children_ids = new_children_ids

        # 更新prev_sibling_id
        prev_id = None
        for child_id in new_children_ids:
            child = session.get_node(child_id)
            if child:
                child.prev_sibling_id = prev_id
                prev_id = child_id

        if self.enable_logging:
            print_and_flush(f"    [Patch reorder] Reordered {len(resolved_indices)} children")

    def _patch_add_bridge(self, session, chapter: OutlineNode, patch: Dict) -> None:
        """
        补丁：添加桥梁节点

        【支持多种格式】
        格式1 (标准格式):
            {"position": "after:node_id", "node": {"title": "...", "description": "..."}}

        格式2 (LLM 常返回格式):
            {"target_id": "node_xxx", "title": "...", "content": "..."}

        格式3 (content_insertion 类型):
            {"type": "content_insertion", "target": "chapter_opening", "content": "...", "reason": "..."}

        格式4 (symbolic_reinforcement 类型):
            {"type": "symbolic_reinforcement", "target": "between node_xxx and node_yyy", "patch": "..."}
        """
        # 解析节点数据（支持多种格式）
        node_data = patch.get("node", {})
        if not node_data:
            # 格式2：字段直接在 patch 中
            node_data = {
                "title": patch.get("title", ""),
                "description": patch.get("content", "") or patch.get("description", "") or patch.get("patch", ""),
                "core_concepts": patch.get("core_concepts", [])
            }

        # 格式3/4：content_insertion 或 symbolic_reinforcement
        if not node_data.get("title"):
            content = patch.get("content", "") or patch.get("patch", "")
            if content:
                # 生成默认标题
                target = patch.get("target", "")
                if "opening" in target.lower() or "opening" in str(target).lower():
                    node_data["title"] = "章节开篇过渡"
                elif "between" in target.lower():
                    node_data["title"] = "过渡段落"
                else:
                    node_data["title"] = "补充内容"
                node_data["description"] = content

        if not node_data.get("title"):
            return

        new_node = OutlineNode(
            node_id=generate_node_id(),
            title=node_data.get("title", "桥梁节点"),
            description=node_data.get("description", "") or node_data.get("content", ""),
            level=chapter.level + 1,
            status=NodeStatus.ACCEPTED,
            parent_id=chapter.node_id,
            core_concepts=node_data.get("core_concepts", []),
            is_finalized=True
        )

        session.register_node(new_node)

        # 注册核心概念
        if new_node.core_concepts:
            session.add_to_concept_registry(new_node.core_concepts, source_node=new_node)

        # 解析位置
        position = patch.get("position", "")
        target_id = patch.get("target_id", "")
        target = patch.get("target", "")  # 新格式
        operation = patch.get("operation", "")

        if position.startswith("after:"):
            # 标准格式: position="after:node_id"
            ref_id = position[6:]
            try:
                idx = chapter.children_ids.index(ref_id)
                chapter.children_ids.insert(idx + 1, new_node.node_id)
            except ValueError:
                chapter.children_ids.append(new_node.node_id)
        elif position.startswith("before:"):
            # 标准格式: position="before:node_id"
            ref_id = position[7:]
            try:
                idx = chapter.children_ids.index(ref_id)
                chapter.children_ids.insert(idx, new_node.node_id)
            except ValueError:
                chapter.children_ids.insert(0, new_node.node_id)
        elif target_id and "after" in operation:
            # LLM 格式: target_id="node_xxx", operation="insert_after"
            try:
                idx = chapter.children_ids.index(target_id)
                chapter.children_ids.insert(idx + 1, new_node.node_id)
            except ValueError:
                chapter.children_ids.append(new_node.node_id)
        elif target_id and "before" in operation:
            # LLM 格式: target_id="node_xxx", operation="insert_before"
            try:
                idx = chapter.children_ids.index(target_id)
                chapter.children_ids.insert(idx, new_node.node_id)
            except ValueError:
                chapter.children_ids.insert(0, new_node.node_id)
        elif target:
            # 新格式：target 可能是 "chapter_opening", "between node_xxx and node_yyy"
            target_str = str(target).lower()
            if "opening" in target_str:
                # 插入到章节开头
                chapter.children_ids.insert(0, new_node.node_id)
            elif "between" in target_str:
                # 尝试解析 "between node_xxx and node_yyy"
                match = re.search(r'between\s+(node_\w+)\s+and\s+(node_\w+)', target_str)
                if match:
                    node_a, node_b = match.group(1), match.group(2)
                    try:
                        idx = chapter.children_ids.index(node_a)
                        chapter.children_ids.insert(idx + 1, new_node.node_id)
                    except ValueError:
                        chapter.children_ids.append(new_node.node_id)
                else:
                    chapter.children_ids.append(new_node.node_id)
            else:
                chapter.children_ids.append(new_node.node_id)
        else:
            chapter.children_ids.append(new_node.node_id)

        if self.enable_logging:
            print_and_flush(f"    [Patch add_bridge] Added: {new_node.title}")

    def _patch_modify(self, session, chapter: OutlineNode, patch: Dict) -> None:
        """
        补丁：修改节点属性

        【支持多种ID格式】
        1. 完整 node_id: "node_xxx"
        2. 节点编号: "1.1", "2.3" 等
        3. 节点标题（部分匹配）

        【支持多种字段名】
        - title: 节点标题
        - description / content: 节点描述
        - core_concepts: 核心概念

        【支持 content_expansion 类型】
        格式: {"type": "content_expansion", "target_id": "node_xxx", "patch": "扩展内容..."}
        或: {"type": "content_expansion", "target": "node_xxx", "original": "原始内容", "patch": "扩展后内容"}
        """
        # 获取节点 ID（支持多种字段名）
        node_id_or_ref = patch.get("node_id", "") or patch.get("target_id", "") or patch.get("target", "")
        if not node_id_or_ref:
            return

        # 尝试解析节点引用
        node = self._resolve_node_reference(session, chapter, node_id_or_ref)
        if not node:
            if self.enable_logging:
                print_and_flush(f"    [Patch modify] Node not found: {node_id_or_ref}")
            return

        # 更新标题
        if patch.get("title"):
            node.title = patch["title"]

        # 更新描述（支持多种字段名）
        if patch.get("description"):
            node.description = patch["description"]
        elif patch.get("content"):
            node.description = patch["content"]
        elif patch.get("patch"):
            # content_expansion 类型使用 patch 字段存储扩展内容
            # 可能是追加到现有描述，还是替换
            patch_content = patch["patch"]
            if patch.get("original") and node.description:
                # 如果提供了 original，说明是替换特定内容
                # 这里简单处理为追加
                node.description = f"{node.description}\n\n{patch_content}"
            else:
                # 否则替换整个描述
                node.description = patch_content

        # 更新核心概念
        if patch.get("core_concepts"):
            node.core_concepts = patch["core_concepts"]

        if self.enable_logging:
            print_and_flush(f"    [Patch modify] Modified node: {node.title}")

    def _patch_remove(self, session, chapter: OutlineNode, patch: Dict) -> None:
        """
        补丁：删除冗余节点

        【支持多种ID格式】
        1. 完整 node_id: "node_xxx"
        2. 节点编号: "1.1", "2.3" 等
        3. 节点标题（部分匹配）
        """
        node_id_or_ref = patch.get("node_id", "")
        if not node_id_or_ref:
            return

        # 尝试解析节点引用
        node = self._resolve_node_reference(session, chapter, node_id_or_ref)
        if not node:
            if self.enable_logging:
                print_and_flush(f"    [Patch remove] Node not found: {node_id_or_ref}")
            return

        node_id = node.node_id

        # 从children_ids中移除
        if node_id in chapter.children_ids:
            chapter.children_ids.remove(node_id)

        # 从registry中移除
        if node_id in session.node_registry:
            del session.node_registry[node_id]

        if self.enable_logging:
            print_and_flush(f"    [Patch remove] Removed node: {node.title}")

    def _patch_restructure(self, session, chapter: OutlineNode, patch: Dict) -> None:
        """
        补丁：结构重组（调整章节顺序）

        当检测到"结论"类章节出现在中间位置时，将其移至末尾
        """
        node_id = patch.get("node_id", "")
        target_position = patch.get("target_position")
        reason = patch.get("reason", "结构重组")

        if not node_id or target_position is None:
            return

        # 获取 root 节点
        root = session.get_node(session.root_id)
        if not root:
            return

        # 找到当前节点在 root.children_ids 中的位置
        try:
            current_idx = root.children_ids.index(node_id)
        except ValueError:
            if self.enable_logging:
                print_and_flush(f"    [Patch restructure] Node not found in root children: {node_id}")
            return

        # 执行移动
        root.children_ids.pop(current_idx)
        root.children_ids.insert(target_position, node_id)

        # 更新 prev_sibling_id
        prev_id = None
        for child_id in root.children_ids:
            child = session.get_node(child_id)
            if child:
                child.prev_sibling_id = prev_id
                prev_id = child_id

        if self.enable_logging:
            moved_node = session.get_node(node_id)
            print_and_flush(f"    [Patch restructure] Moved '{moved_node.title if moved_node else node_id}' from position {current_idx + 1} to {target_position + 1}")
            print_and_flush(f"    Reason: {reason}")

    def _patch_swap(self, session, chapter: OutlineNode, patch: Dict) -> None:
        """
        补丁：交换两个节点的位置

        Args:
            session: 会话实例
            chapter: 章节点
            patch: {"action": "swap", "node_a": "id1", "node_b": "id2"}
        """
        node_a_id = patch.get("node_a")
        node_b_id = patch.get("node_b")

        if not node_a_id or not node_b_id:
            if self.enable_logging:
                print_and_flush(f"    [Patch swap] Missing node_a or node_b in patch")
            return

        # 尝试在 chapter.children_ids 中查找两个节点
        try:
            idx_a = chapter.children_ids.index(node_a_id)
            idx_b = chapter.children_ids.index(node_b_id)
        except ValueError:
            # 如果按 node_id 找不到，尝试解析节点引用
            node_a = self._resolve_node_reference(session, chapter, node_a_id)
            node_b = self._resolve_node_reference(session, chapter, node_b_id)

            if node_a and node_b:
                node_a_id = node_a.node_id
                node_b_id = node_b.node_id
                try:
                    idx_a = chapter.children_ids.index(node_a_id)
                    idx_b = chapter.children_ids.index(node_b_id)
                except ValueError:
                    if self.enable_logging:
                        print_and_flush(f"    [Patch swap] Nodes not found in chapter children")
                    return
            else:
                if self.enable_logging:
                    print_and_flush(f"    [Patch swap] Could not resolve node references")
                return

        # 执行交换
        chapter.children_ids[idx_a], chapter.children_ids[idx_b] = chapter.children_ids[idx_b], chapter.children_ids[idx_a]

        # 更新 prev_sibling_id
        prev_id = None
        for child_id in chapter.children_ids:
            child = session.get_node(child_id)
            if child:
                child.prev_sibling_id = prev_id
                prev_id = child_id

        if self.enable_logging:
            node_a = session.get_node(node_a_id)
            node_b = session.get_node(node_b_id)
            title_a = node_a.title if node_a else node_a_id
            title_b = node_b.title if node_b else node_b_id
            print_and_flush(f"    [Patch swap] Swapped '{title_a}' (position {idx_a + 1}) with '{title_b}' (position {idx_b + 1})")

    def _default_checkpoint_result(self, chapter: OutlineNode) -> Dict[str, Any]:
        """返回默认的 checkpoint 结果"""
        return {
            "evaluation": {
                "relevance": 8, "accuracy": 8, "coherence": 8,
                "clarity": 8, "breadth_depth": 8, "reading_experience": 8
            },
            "issues": [],
            "suggestions": [],
            "needs_revision": False,
            "chapter_abstract": chapter.description or chapter.title,
            "transition_quality": "良好",
            "patches": []
        }

    def _calculate_avg_score(self, evaluation: Dict) -> float:
        """计算平均分"""
        if not evaluation:
            return 8.0
        scores = [
            evaluation.get("relevance", 8),
            evaluation.get("accuracy", 8),
            evaluation.get("coherence", 8),
            evaluation.get("clarity", 8),
            evaluation.get("breadth_depth", 8),
            evaluation.get("reading_experience", 8)
        ]
        return sum(scores) / len(scores)