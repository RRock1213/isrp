"""
Phase II — 全局验证与修复

在大纲树生成完毕后进行全文级别的系统性校验：
  - 规则检查：章节编号重复/混乱、编号体系不一致、单节点层级结构
  - 语义检查：标题风格一致性、结构完整性、时间矛盾
  - 自动修复：调用 LLM 生成修复计划，包括重新编号、标题修正、章节补充、结构调整
确保最终大纲符合既定的写作规范和格式要求。

"""
import sys
import re
from typing import List, Dict, Any, Optional

from ..schemas import OutlineNode, robust_json_parse
from ..utils.llm_client import LLMClient
from ..prompts import PromptTemplates
from ..engines import get_temperature

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout

class GlobalValidation:
    """Global Validation"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: PromptTemplates,
        enable_logging: bool = True
    ):
        self.llm = llm_client
        self.prompts = prompts
        self.enable_logging = enable_logging

    def execute(self, session) -> None:
        """
        执行大纲校验

        Args:
            session: 会话实例
        """
        if self.enable_logging:
            print_and_flush("\n[Global Validation] Outline Validation & Repair")
            print_and_flush("=" * 60)

        root = session.get_node(session.root_id)
        if not root:
            return

        # 收集所有章节及其子节点
        chapters = self._collect_chapters(session, root)

        if not chapters:
            if self.enable_logging:
                print_and_flush("  No chapters to validate")
            return

        # 检测问题
        issues = self._detect_issues(chapters)

        if not issues:
            if self.enable_logging:
                print_and_flush("  No issues detected, outline is clean")
            return

        if self.enable_logging:
            print_and_flush(f"  Detected {len(issues)} issues:")
            for issue in issues:
                print_and_flush(f"    - {issue}")

        # 调用 LLM 进行修复
        repairs = self._llm_validate_and_repair(session, chapters, issues)

        # 应用修复
        if repairs:
            self._apply_repairs(session, chapters, repairs)

        # 执行单子节点层级提升
        self._promote_single_child_nodes(session)

    def _collect_chapters(self, session, root: OutlineNode) -> List[Dict[str, Any]]:
        """收集所有章节及其子节点信息"""
        chapters = []

        for chapter_id in root.children_ids:
            chapter = session.get_node(chapter_id)
            if not chapter:
                continue

            chapter_info = {
                "node_id": chapter.node_id,
                "title": chapter.title,
                "description": chapter.description or "",
                "children": []
            }

            # 收集子节点
            for child_id in chapter.children_ids:
                child = session.get_node(child_id)
                if child:
                    chapter_info["children"].append({
                        "node_id": child.node_id,
                        "title": child.title,
                        "description": child.description or ""
                    })

            chapters.append(chapter_info)

        return chapters

    def _detect_issues(self, chapters: List[Dict[str, Any]]) -> List[str]:
        """检测大纲中的问题"""
        issues = []

        # 1. 检测编号重复/混乱
        chinese_nums = []
        arabic_nums = []

        for i, chapter in enumerate(chapters):
            title = chapter["title"]

            # 提取中文编号
            cn_match = re.search(r'^[一二两三四五六七八九十]+[章部节]', title)
            if cn_match:
                chinese_nums.append((i, cn_match.group()))

            # 提取阿拉伯数字编号
            ar_match = re.search(r'^第?(\d+)[章部节]?', title)
            if ar_match:
                arabic_nums.append((i, ar_match.group()))

            # 检测标题中是否包含多个编号
            if re.search(r'第[一二两三四五六七八九十\d]+章.*第[一二两三四五六七八九十\d]+章', title):
                issues.append(f"章节 '{title[:30]}...' 包含重复编号")

            # 检测编号与位置不符
            cn_num_match = re.search(r'^[一二两三四五六七八九十]+', title)
            if cn_num_match:
                cn_num = self._chinese_to_num(cn_num_match.group())
                if cn_num != i + 1:
                    issues.append(f"章节 '{title[:30]}...' 编号位置不匹配 (位置 {i+1}, 编号 {cn_num})")
        continuity_issues = self._detect_chapter_continuity_issues(chapters)
        issues.extend(continuity_issues)

        # 2. 检测编号系统混用
        if chinese_nums and arabic_nums:
            issues.append(f"混用中文({len(chinese_nums)})和阿拉伯数字({len(arabic_nums)})编号")

        # 3. 检测标题风格不一致
        titles = [c["title"] for c in chapters]
        styles = self._detect_title_styles(titles)
        if len(styles) > 1:
            issues.append(f"标题风格不一致: {', '.join(styles)}")

        # 4. 检测章节标题重复
        title_counts = {}
        for chapter in chapters:
            title = chapter["title"]
            # 清理编号后比较
            clean_title = self._clean_title_prefix(title)
            if clean_title in title_counts:
                title_counts[clean_title] += 1
            else:
                title_counts[clean_title] = 1

        for clean_title, count in title_counts.items():
            if count > 1:
                issues.append(f"存在 {count} 个同名章节: '{clean_title[:30]}...'")

        # 5. 检测论文结构完整性
        paper_issues = self._detect_paper_structure_issues(chapters)
        issues.extend(paper_issues)

        # 6. 检测时序矛盾（针对小说/故事类）
        timeline_issues = self._detect_timeline_issues(chapters)
        issues.extend(timeline_issues)

        # 7. 检测单子节点层级问题
        single_child_issues = self._detect_single_child_issues(chapters)
        issues.extend(single_child_issues)

        # 8. 检测跨章节数值一致性问题
        numerical_issues = self._detect_numerical_consistency_issues(chapters)
        issues.extend(numerical_issues)

        return issues

    def _detect_paper_structure_issues(self, chapters: List[Dict[str, Any]]) -> List[str]:
        """
        检测论文结构完整性

        检查是否缺少：摘要、引言、结论、参考文献

        改进：
        - 区分"学术论文"和"开题报告/研究报告"等类型
        - 开题报告、调研报告等不强制要求论文结构
        - 增强对文体类型的识别，避免误判
        """
        issues = []
        titles_text = " ".join([c["title"] for c in chapters])
        # 开题报告、调研报告、实习报告等不需要论文结构
        non_paper_types = [
            '开题报告', '调研报告', '调查报告', '实习报告', '实习总结',
            '工作报告', '述职报告', '党性分析', '心得体会', '总结',
            'proposal', 'internship', 'survey', 'report'
        ]
        for np_type in non_paper_types:
            if np_type.lower() in titles_text.lower():
                return issues  # 非学术论文，不检查论文结构

        # 检测是否是论文类型（包含"论文"关键词，或明确的研究论文特征）
        is_paper = any(kw in titles_text for kw in ['论文', 'thesis', 'paper', 'dissertation'])

        # 如果标题中没有明确的"论文"关键词，检查是否有论文特征
        if not is_paper:
            # 只有同时包含"研究"+"分析/方法/结果"等论文特征词才判断为论文
            has_research = any(kw in titles_text for kw in ['研究', 'research', 'study'])
            has_paper_features = any(kw in titles_text for kw in ['方法', '结果', '讨论', 'method', 'result', 'discussion'])
            is_paper = has_research and has_paper_features

        if not is_paper:
            return issues
        all_text_lower = titles_text.lower()
        for ch in chapters:
            all_text_lower += " " + (ch.get("description", "") or "").lower()
            for child in ch.get("children", []):
                all_text_lower += " " + (child.get("title", "") or "").lower()
                all_text_lower += " " + (child.get("description", "") or "").lower()

        # 检查论文必要组成部分
        # 优先检查标题，如果标题中没有则检查内容
        titles_lower = [c["title"].lower() for c in chapters]
        titles_lower_text = " ".join(titles_lower)

        has_abstract = any(kw in titles_lower_text for kw in ['摘要', 'abstract'])
        if not has_abstract:
            has_abstract = any(kw in all_text_lower for kw in ['摘要', 'abstract', '本文研究', '本研究旨在', '本文旨在'])

        has_intro = any(kw in titles_lower_text for kw in ['引言', '绪论', '前言', 'introduction'])
        has_conclusion = any(kw in titles_lower_text for kw in ['结论', '总结', '结语', 'conclusion'])
        has_refs = any(kw in titles_lower_text for kw in ['参考文献', 'reference', 'bibliography'])

        if not has_abstract:
            issues.append("[论文结构] 缺少'摘要'部分")
        if not has_intro:
            issues.append("[论文结构] 缺少'引言'部分")
        if not has_conclusion:
            issues.append("[论文结构] 缺少'结论'部分")
        if not has_refs:
            issues.append("[论文结构] 缺少'参考文献'部分")
        special_chapters = ['摘要', 'abstract', '参考文献', 'reference', '致谢', 'acknowledgment', '附录', 'appendix']
        for i, chapter in enumerate(chapters):
            title_lower = chapter["title"].lower()
            for special in special_chapters:
                if special in title_lower:
                    # 检查是否被编号为"第N章"
                    if re.match(r'^第[一二三四五六七八九十\d]+章', chapter["title"]):
                        issues.append(f"[论文格式] 特殊章节'{chapter['title'][:20]}'不应编号为'第N章'")
                    break

        return issues

    def _detect_timeline_issues(self, chapters: List[Dict[str, Any]]) -> List[str]:
        """
        检测时序矛盾

        检查小说/故事类文本中的时序问题：
        - 死亡事件：角色死后再次出现
        - 关键事件：事件顺序矛盾
        - 状态变化：已发生的状态被再次触发
        - 物品遗失/获得、地点到达/离开的时序检测
        """
        issues = []

        # 关键事件关键词
        death_keywords = ['死', '牺牲', '殉', '亡', '逝', '去世', '离世', '阵亡', '殒命', 'die', 'death', 'killed', 'deceased']
        revival_keywords = ['复活', '重生', '苏醒', '醒过来', 'revive', 'resurrect', 'awaken']

        # 状态变化关键词（一旦发生就不应再次触发）
        irreversible_keywords = ['结婚', '婚', '怀孕', '生子', '毕业', '退休', '辞职', '解散', '破产']
        item_loss_keywords = ['丢失', '遗失', '被夺', '被抢', '失去', '扔掉', '销毁', 'lost', 'destroyed']
        item_gain_keywords = ['获得', '得到', '找到', '买', '赠送', 'gain', 'acquire', 'found']

        # 收集关键事件
        events = []  # [(chapter_idx, event_type, character/event_name, keyword)]

        for i, chapter in enumerate(chapters):
            title = chapter["title"]
            description = chapter.get("description", "")

            # 检查标题和描述中的死亡事件
            for kw in death_keywords:
                if kw in title or kw in description:
                    # 尝试提取角色名（查找关键词前的词语）
                    char_match = self._extract_character_name(title + description, kw)
                    events.append((i, "death", char_match, kw))

            # 检查复活事件（通常不应该在死亡后出现，除非是特定的奇幻设定）
            for kw in revival_keywords:
                if kw in title or kw in description:
                    char_match = self._extract_character_name(title + description, kw)
                    events.append((i, "revival", char_match, kw))

            # 检查不可逆状态变化
            for kw in irreversible_keywords:
                if kw in title or kw in description:
                    char_match = self._extract_character_name(title + description, kw)
                    events.append((i, "irreversible", char_match, kw))
            for kw in item_loss_keywords:
                if kw in title or kw in description:
                    item_match = self._extract_character_name(title + description, kw)
                    if item_match:
                        events.append((i, "item_loss", item_match, kw))
            for kw in item_gain_keywords:
                if kw in title or kw in description:
                    item_match = self._extract_character_name(title + description, kw)
                    if item_match:
                        events.append((i, "item_gain", item_match, kw))

        # 分析事件序列，检测矛盾
        # 1. 检测死亡后再次出现（非复活设定）
        death_chars = {}  # {character: first_death_chapter_idx}
        revival_chars = {}  # {character: revival_chapter_idx}

        for i, event_type, char_name, kw in events:
            if event_type == "death" and char_name:
                if char_name not in death_chars:
                    death_chars[char_name] = i
            elif event_type == "revival" and char_name:
                revival_chars[char_name] = i

        # 检测同一角色多次死亡
        for char_name, first_death_idx in death_chars.items():
            # 查找是否在死亡后有其他涉及该角色的章节（非复活）
            for j in range(first_death_idx + 1, len(chapters)):
                chapter = chapters[j]
                title_desc = chapter["title"] + chapter.get("description", "")

                # 检查角色名是否再次出现
                if char_name and char_name in title_desc:
                    # 检查是否是复活事件
                    is_revival = any(kw in title_desc for kw in revival_keywords)
                    if not is_revival:
                        issues.append(
                            f"[时序矛盾] 角色'{char_name}'在第{first_death_idx + 1}章已死亡，"
                            f"但在第{j + 1}章'{chapter['title'][:20]}...'再次出现"
                        )

        # 2. 检测不可逆状态重复触发
        irreversible_states = {}  # {character+state: first_occurrence_idx}
        for i, event_type, char_name, kw in events:
            if event_type == "irreversible" and char_name:
                key = f"{char_name}_{kw}"
                if key in irreversible_states:
                    # 同一角色同一不可逆状态多次出现
                    issues.append(
                        f"[时序矛盾] '{char_name}'的'{kw}'事件在第{irreversible_states[key] + 1}章已发生，"
                        f"但在第{i + 1}章再次出现"
                    )
                else:
                    irreversible_states[key] = i
        item_status = {}  # {item_name: (status, first_chapter_idx)}，status: "lost" or "gained"
        for i, event_type, item_name, kw in events:
            if event_type == "item_loss" and item_name:
                if item_name in item_status and item_status[item_name][0] == "lost":
                    # 物品已经丢失，再次丢失是矛盾
                    issues.append(
                        f"[时序矛盾] 物品'{item_name}'在第{item_status[item_name][1] + 1}章已丢失，"
                        f"但在第{i + 1}章再次丢失"
                    )
                item_status[item_name] = ("lost", i)
            elif event_type == "item_gain" and item_name:
                if item_name in item_status and item_status[item_name][0] == "gained":
                    # 物品已经获得，再次获得可能是矛盾（除非是多个同类物品）
                    # 这里只做提示，不作为硬性矛盾
                    pass
                item_status[item_name] = ("gained", i)

        return issues

    def _extract_character_name(self, text: str, keyword: str) -> str:
        """
        从文本中提取关键词相关的人物名

        简化处理：查找关键词前后可能的名称
        """
        # 查找关键词位置
        idx = text.find(keyword)
        if idx == -1:
            return ""

        # 提取关键词前10个字符，寻找可能的名称
        before_text = text[:idx]

        # 常见的人物名模式（中文）
        # 1. 名字在动词前：张三死了 -> 张三
        # 2. 名字在句子开头：张三的牺牲 -> 张三

        # 简化处理：提取关键词前最后一个词
        # 使用简单的分词逻辑
        words = re.findall(r'[\u4e00-\u9fa5]{2,4}', before_text)
        if words:
            # 取最后一个词作为可能的人物名
            last_word = words[-1]
            # 过滤常见动词/形容词
            skip_words = ['因为', '为了', '但是', '然而', '虽然', '结果', '最后', '终于', '已经', '已经', '曾经', '曾经']
            if last_word not in skip_words:
                return last_word

        return ""

    def _chinese_to_num(self, cn: str) -> int:
        """中文数字转阿拉伯数字"""
        mapping = {
            '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
        }
        if cn in mapping:
            return mapping[cn]
        if cn.startswith('十'):
            if len(cn) == 1:
                return 10
            return 10 + mapping.get(cn[1], 0)
        return 0

    def _detect_title_styles(self, titles: List[str]) -> List[str]:
        """检测标题风格"""
        styles = set()

        for title in titles:
            if re.match(r'^第[一二三四五六七八九十\d]+章', title):
                styles.add("章节式")
            elif re.match(r'^[\d]+\.', title):
                styles.add("数字点式")
            elif re.match(r'^[一二三四五六七八九十]+[、.．]', title):
                styles.add("中文点式")
            elif not re.match(r'^[\d一二三四五六七八九十第]', title):
                styles.add("纯标题式")

        return list(styles)

    def _llm_validate_and_repair(
        self,
        session,
        chapters: List[Dict[str, Any]],
        issues: List[str]
    ) -> Optional[Dict[str, Any]]:
        """调用 LLM 进行校验和修复"""

        # 构建大纲摘要
        outline_summary = []
        for i, chapter in enumerate(chapters):
            summary = f"{i+1}. {chapter['title']}"
            if chapter['children']:
                child_titles = [f"   - {child['title']}" for child in chapter['children'][:3]]
                summary += "\n" + "\n".join(child_titles)
                if len(chapter['children']) > 3:
                    summary += f"\n   - ... 等 {len(chapter['children'])} 个子节点"
            outline_summary.append(summary)

        outline_text = "\n\n".join(outline_summary)

        # 构建 prompt
        lang = self.prompts._language
        prompt = self.prompts.get_template("PROMPT_VALIDATE_OUTLINE", lang).format(
            outline_text=outline_text,
            issues="\n".join(f"- {issue}" for issue in issues),
            chapter_count=len(chapters)
        )

        try:
            response = self.llm.call(
                prompt,
                temperature=get_temperature("word_allocation", "evaluate")
            )

            if not response or not response.strip():
                if self.enable_logging:
                    print_and_flush("  [Validate] Empty response from LLM")
                return None

            result = robust_json_parse(response, dict)

            if not result:
                if self.enable_logging:
                    print_and_flush("  [Validate] Failed to parse LLM response")
                return None

            return result

        except Exception as e:
            if self.enable_logging:
                print_and_flush(f"  [Validate] Error: {e}")
            return None

    def _apply_repairs(
        self,
        session,
        chapters: List[Dict[str, Any]],
        repairs: Dict[str, Any]
    ) -> None:
        """
        应用修复

        支持：
        - 删除多余章节（当 LLM 返回更少章节时）
        - 创建子节点结构（处理 LLM 返回的 children 字段）
        - 章节重排序（根据 LLM 建议的顺序重新排列）
        """

        # 检查是否需要修复
        if not repairs.get("needs_repair", False):
            if self.enable_logging:
                print_and_flush("  LLM determined no repair needed")
            return

        # 获取修复后的章节列表
        repaired_chapters = repairs.get("chapters", [])

        if not repaired_chapters:
            if self.enable_logging:
                print_and_flush("  No repair data provided")
            return

        if self.enable_logging:
            print_and_flush(f"  Applying repairs to {len(repaired_chapters)} chapters...")

        root = session.get_node(session.root_id)
        # 使用标题相似度匹配而非位置匹配
        title_to_chapter = {}
        for chapter in chapters:
            clean_title = self._clean_title_prefix(chapter["title"])
            title_to_chapter[clean_title] = chapter
            # 也保存原始标题作为备选
            title_to_chapter[chapter["title"]] = chapter
        new_children_order = []
        matched_chapters = set()

        for repaired in repaired_chapters:
            repaired_title = self._clean_title_prefix(repaired.get("title", ""))

            # 尝试精确匹配
            matched_chapter = None
            matched_node_id = None

            # 1. 尝试精确标题匹配
            for chapter in chapters:
                if chapter["node_id"] in matched_chapters:
                    continue
                chapter_title = self._clean_title_prefix(chapter["title"])
                if chapter_title == repaired_title:
                    matched_chapter = chapter
                    break

            # 2. 尝试模糊匹配（标题包含关系）
            if not matched_chapter:
                for chapter in chapters:
                    if chapter["node_id"] in matched_chapters:
                        continue
                    chapter_title = self._clean_title_prefix(chapter["title"])
                    if repaired_title in chapter_title or chapter_title in repaired_title:
                        matched_chapter = chapter
                        break

            # 3. 按位置匹配作为最后的备选
            if not matched_chapter:
                repaired_idx = repaired_chapters.index(repaired)
                if repaired_idx < len(chapters):
                    matched_chapter = chapters[repaired_idx]

            if matched_chapter:
                node_id = matched_chapter["node_id"]
                node = session.get_node(node_id)

                if node:
                    matched_chapters.add(node_id)
                    new_children_order.append(node_id)
                    # 只允许重排序和创建新章节，避免错误的重命名
                    # 原问题：当 LLM 返回的章节数量与原章节数量不一致时，
                    # 按位置匹配会导致所有章节被错误重命名
                    # if repaired_title and repaired_title != node.title:
                    #     old_title = node.title
                    #     node.title = repaired_title
                    #     if self.enable_logging:
                    #         print_and_flush(f"    Chapter: '{old_title[:30]}...' -> '{repaired_title[:30]}...'")

                    # 更新描述（仍然允许）
                    new_description = repaired.get("description", "")
                    if new_description:
                        node.description = new_description

                    # 处理子节点
                    repaired_children = repaired.get("children", [])
                    existing_children = matched_chapter.get("children", [])

                    if repaired_children and not existing_children:
                        if self.enable_logging:
                            print_and_flush(f"    Creating {len(repaired_children)} new child nodes")
                        self._create_child_nodes(session, node, repaired_children)

                    elif repaired_children and existing_children:
                        for j, repaired_child in enumerate(repaired_children):
                            if j >= len(existing_children):
                                if self.enable_logging:
                                    print_and_flush(f"    Creating extra child node {j+1}")
                                self._create_child_nodes(session, node, repaired_children[j:], start_idx=len(existing_children))
                                break

                            child_info = existing_children[j]
                            child_node = session.get_node(child_info["node_id"])

                            if child_node and repaired_child.get("title"):
                                clean_child_title = self._clean_title_prefix(repaired_child["title"])
                                if clean_child_title != child_node.title:
                                    child_node.title = clean_child_title
        if len(new_children_order) > 0 and root:
            # 保留未匹配的章节（追加到末尾）
            for chapter in chapters:
                if chapter["node_id"] not in matched_chapters:
                    new_children_order.append(chapter["node_id"])

            root.children_ids = new_children_order

            if self.enable_logging:
                print_and_flush(f"  Reordered {len(new_children_order)} chapters")
        if len(repaired_chapters) < len(chapters):
            if self.enable_logging:
                print_and_flush(f"  Note: LLM suggests {len(chapters) - len(repaired_chapters)} fewer chapters")

        if self.enable_logging:
            print_and_flush("  Repair completed")

    def _create_child_nodes(
        self,
        session,
        parent_node: OutlineNode,
        children_data: List[Dict[str, Any]],
        start_idx: int = 0
    ) -> None:
        """
        创建子节点

        根据 LLM 返回的 children 数据创建新节点

        Args:
            session: 会话实例
            parent_node: 父节点
            children_data: 子节点数据列表
            start_idx: 开始插入的位置
        """
        from ..schemas import generate_node_id, NodeStatus
        from ..engines import MIN_LEAF_WORDS, MAX_LEAF_WORDS

        # 计算父节点剩余可用字数（基于已分配字数）
        parent_allocated = parent_node.allocated_words or 0
        existing_children_count = len(parent_node.children_ids)
        new_children_count = len(children_data)
        total_children = existing_children_count + new_children_count

        # 为新子节点分配合理的字数
        if total_children > 0 and parent_allocated > 0:
            words_per_child = max(MIN_LEAF_WORDS, parent_allocated // (total_children + 1))
        else:
            words_per_child = MIN_LEAF_WORDS

        for j, child_data in enumerate(children_data):
            child_title = child_data.get("title", f"子节点 {start_idx + j + 1}")
            clean_title = self._clean_title_prefix(child_title)

            # 从 child_data 获取描述，或生成有意义的默认描述
            description = child_data.get("description", "")
            if not description or description.strip() == "":
                description = self._generate_default_description(clean_title, parent_node, child_data)

            new_child = OutlineNode(
                node_id=generate_node_id(),
                title=clean_title,
                description=description,
                level=parent_node.level + 1,
                status=NodeStatus.ACCEPTED,
                parent_id=parent_node.node_id,
                is_finalized=True,
                core_concepts=child_data.get("core_concepts", []),
                allocated_words=words_per_child  # 分配字数
            )
            session.register_node(new_child)
            parent_node.children_ids.append(new_child.node_id)

            if self.enable_logging:
                print_and_flush(f"    Created child: {clean_title} ({words_per_child} words)")

    def _generate_default_description(
        self,
        clean_title: str,
        parent_node: OutlineNode,
        child_data: Dict[str, Any]
    ) -> str:
        """
        生成有意义的默认描述
        当 LLM 未返回描述时，根据标题和父节点上下文生成有意义的描述，
        而不是使用空洞的"章节：xxx"格式。

        Args:
            clean_title: 清理后的节点标题
            parent_node: 父节点
            child_data: 子节点数据

        Returns:
            有意义的描述文本
        """
        # 1. 尝试从父节点描述中提取相关信息
        parent_desc = parent_node.description or ""
        parent_title = parent_node.title or ""

        # 2. 根据标题关键词生成描述
        title_lower = clean_title.lower()

        # 常见章节类型的描述模板
        description_templates = {
            # 背景类
            "背景": f"本部分阐述{clean_title}的背景信息，为后续分析提供上下文支撑。",
            "前言": f"作为开篇部分，本节介绍{clean_title}相关内容，引导读者理解全文框架。",
            "引言": f"本节作为引言部分，介绍{clean_title}的核心要点与研究意义。",
            "概述": f"本部分对{clean_title}进行全面概述，奠定整体认知基础。",

            # 分析类
            "分析": f"本部分深入分析{clean_title}，探讨其内在逻辑与关键因素。",
            "现状": f"本部分系统梳理{clean_title}的现状，分析当前发展态势与存在问题。",
            "问题": f"本部分聚焦{clean_title}，深入剖析问题的本质与成因。",
            "原因": f"本部分从多维度分析{clean_title}，揭示问题的深层根源。",

            # 方法类
            "方法": f"本部分详细说明{clean_title}的研究方法与实施步骤。",
            "过程": f"本部分记录{clean_title}的具体过程，展示关键环节与操作细节。",
            "方案": f"本部分提出针对{clean_title}的具体方案，明确实施路径与关键措施。",
            "路线": f"本部分规划{clean_title}的技术路线，确保研究有序推进。",

            # 结果类
            "结果": f"本部分呈现{clean_title}的研究结果，进行数据解读与分析。",
            "发现": f"本部分总结{clean_title}的关键发现，提炼核心结论。",
            "成果": f"本部分展示{clean_title}的主要成果，评估其价值与意义。",

            # 讨论类
            "讨论": f"本部分对{clean_title}进行深入讨论，阐释其内涵与启示。",
            "意义": f"本部分阐述{clean_title}的理论意义与实践价值。",
            "影响": f"本部分分析{clean_title}的多方面影响，评估其深远作用。",

            # 建议类
            "建议": f"本部分针对{clean_title}提出具体建议，为改进提供参考方向。",
            "对策": f"本部分提出针对{clean_title}的对策措施，明确行动路径。",
            "措施": f"本部分制定{clean_title}的具体措施，确保问题得到有效解决。",

            # 结论类
            "结论": f"本部分总结{clean_title}的核心观点，归纳全文主要结论。",
            "总结": f"本部分对{clean_title}进行全面总结，梳理关键要点。",
            "结语": f"本部分作为结语，对{clean_title}进行收尾与升华。",

            # 参考文献类
            "参考文献": "本部分列出研究过程中引用的所有文献资料，确保学术规范。",
            "文献": "本部分整理相关文献资料，为研究提供理论支撑。",

            # 其他常见类型
            "目标": f"本部分明确{clean_title}的具体目标，为后续工作指明方向。",
            "内容": f"本部分详细阐述{clean_title}的核心内容，展开深入论述。",
            "创新": f"本部分突出{clean_title}的创新点，展示研究的独特价值。",
            "可行性": f"本部分论证{clean_title}的可行性，分析实施条件与保障措施。",
            "基础": f"本部分奠定{clean_title}的基础，为后续研究提供支撑。",
            "规划": f"本部分制定{clean_title}的规划方案，明确阶段性目标与任务。",
        }

        # 3. 匹配描述模板
        for keyword, template in description_templates.items():
            if keyword in clean_title:
                return template

        # 4. 如果没有匹配的模板，生成具体描述
        if parent_title and len(parent_title) > 3:
            # 从父标题提取关键词
            parent_keyword = self._clean_title_prefix(parent_title)
            return f"本部分具体阐述{clean_title}的核心内容，承接{parent_keyword}的分析脉络，需结合实例进行深入论述。"

        # 5. 最简单的默认描述
        return f"本部分详细展开{clean_title}的具体内容，需提供明确的观点或数据支撑，确保论述的针对性。"

    def _clean_title_prefix(self, title: str) -> str:
        """清理标题前缀的编号"""
        # 移除各种编号格式
        patterns = [
            r'^第[一二三四五六七八九十\d]+章\s*',      # "第一章" "第1章"
            r'^第[一二三四五六七八九十\d]+部\s*',      # "第一部"
            r'^第[一二三四五六七八九十\d]+节\s*',      # "第一节"
            r'^[一二三四五六七八九十]+[、.．]\s*',     # "一、" "一."
            r'^\d+[\.\s]+\s*',                        # "1." "1 "
            r'^\d+章\s*',                             # "1章"
        ]

        cleaned = title
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned)

        return cleaned.strip()

    def _detect_single_child_issues(self, chapters: List[Dict[str, Any]]) -> List[str]:
        """
        检测单子节点层级问题
        当某章只有一个子节点，且该子节点有多个孙子节点时，
        应该将孙子节点提升为子节点

        场景：
        第N章
        └── N.1 (唯一子节点)
            ├── N.1.1 内容A
            └── N.1.2 内容B

        应该优化为：
        第N章
        ├── N.1 内容A (原 N.1.1 提升)
        └── N.2 内容B (原 N.1.2 提升)
        """
        issues = []

        for i, chapter in enumerate(chapters):
            children = chapter.get("children", [])

            # 只有一个子节点的情况
            if len(children) == 1:
                # 需要检查这个子节点是否有孙子节点
                # 由于 _collect_chapters 只收集两层，这里只能标记问题
                issues.append(f"[结构优化] 第{i+1}章只有1个子节点，建议检查是否需要层级提升")

        return issues

    def _detect_chapter_continuity_issues(self, chapters: List[Dict[str, Any]]) -> List[str]:
        """
        检测章节编号连续性问题

        检查：
        1. 章节编号是否从1开始
        2. 章节编号是否连续（无跳号）
        3. 是否存在章节编号跳跃（如从第3章直接到第5章）

        示例问题：
        - 章节从第2章开始（缺少第1章）
        - 第3章后是第5章（缺少第4章）
        - 章节编号顺序混乱（如第5章、第3章、第4章）
        """
        issues = []

        if not chapters:
            return issues

        # 收集所有章节编号
        chapter_nums = []  # [(index, num, title)]

        for i, chapter in enumerate(chapters):
            title = chapter["title"]

            # 提取中文编号
            cn_match = re.search(r'^第?([一二两三四五六七八九十]+)章', title)
            if cn_match:
                num = self._chinese_to_num(cn_match.group(1))
                chapter_nums.append((i, num, title))
                continue

            # 提取阿拉伯数字编号
            ar_match = re.search(r'^第?(\d+)章', title)
            if ar_match:
                num = int(ar_match.group(1))
                chapter_nums.append((i, num, title))
                continue

            # 尝试匹配纯数字开头
            pure_num_match = re.search(r'^(\d+)[\.\s]', title)
            if pure_num_match:
                num = int(pure_num_match.group(1))
                chapter_nums.append((i, num, title))

        if not chapter_nums:
            # 没有检测到章节编号，不报错
            return issues

        # 检查1：章节是否从1开始
        first_num = chapter_nums[0][1]
        if first_num > 1:
            issues.append(
                f"[章节编号] 章节从第{first_num}章开始，缺少第1章到第{first_num-1}章"
            )

        # 检查2：章节编号是否连续
        detected_nums = [num for _, num, _ in chapter_nums]

        for i in range(len(detected_nums) - 1):
            current_num = detected_nums[i]
            next_num = detected_nums[i + 1]

            # 检查是否跳号
            if next_num > current_num + 1:
                # 跳过了章节
                skipped = list(range(current_num + 1, next_num))
                if len(skipped) == 1:
                    issues.append(
                        f"[章节编号] 第{current_num}章后直接是第{next_num}章，缺少第{skipped[0]}章"
                    )
                else:
                    skipped_str = "、".join(f"第{n}章" for n in skipped)
                    issues.append(
                        f"[章节编号] 第{current_num}章后直接是第{next_num}章，缺少 {skipped_str}"
                    )
            elif next_num < current_num:
                # 编号顺序混乱
                issues.append(
                    f"[章节编号] 章节编号顺序混乱：第{current_num}章后出现第{next_num}章"
                )
            elif next_num == current_num:
                # 编号重复
                issues.append(
                    f"[章节编号] 存在重复的章节编号：第{current_num}章"
                )

        # 检查3：检测"第X章"编号与实际位置不符
        # 允许一定误差（如有序言、引言等非编号章节）
        sorted_nums = sorted(detected_nums)
        expected_sequence = list(range(1, len(detected_nums) + 1))

        # 如果编号数量与章节数量匹配但编号不从1开始
        if first_num > 1 and len(set(detected_nums)) == len(chapter_nums):
            # 检查是否只是偏移问题
            expected_start = 1
            if sorted_nums[0] != expected_start:
                issues.append(
                    f"[章节编号] 建议章节编号从第1章开始，当前从第{sorted_nums[0]}章开始"
                )

        return issues

    def _promote_single_child_nodes(self, session) -> None:
        """
        执行单子节点层级提升

        当某章只有一个子节点时：
        1. 如果子节点有孙节点（>=2个）：提升孙节点
        2. 如果子节点无孙节点：将子节点内容合并到章节
        """
        root = session.get_node(session.root_id)
        if not root:
            return

        merged_count = 0
        promoted_count = 0

        for chapter_id in root.children_ids[:]:  # 使用切片创建副本，因为会修改列表
            chapter = session.get_node(chapter_id)
            if not chapter:
                continue

            # 检查是否只有一个子节点
            if len(chapter.children_ids) != 1:
                continue

            single_child_id = chapter.children_ids[0]
            single_child = session.get_node(single_child_id)
            if not single_child:
                continue

            # 检查这个唯一子节点是否有子节点
            grand_children_ids = single_child.children_ids[:]

            if len(grand_children_ids) >= 2:
                # 情况1：提升孙节点（原有逻辑）
                promoted_count += self._promote_grand_children(
                    session, chapter, single_child, grand_children_ids
                )
            elif len(grand_children_ids) == 0:
                # 情况2：无孙节点，合并子节点到章节                if self._merge_single_child_to_chapter(session, chapter, single_child):
                    merged_count += 1
            # len == 1 的情况保持不变（孙子节点太少，结构暂时合理）

        if self.enable_logging and (merged_count > 0 or promoted_count > 0):
            print_and_flush(f"  [结构优化] 合并 {merged_count} 个单子节点，提升 {promoted_count} 个孙节点层级")

    def _promote_grand_children(self, session, chapter, single_child, grand_children_ids) -> int:
        """
        提升孙节点为章的直接子节点

        Returns:
            提升的孙节点数量
        """
        if self.enable_logging:
            print_and_flush(f"  [结构优化] 提升孙节点: {chapter.title[:20]}... ({len(grand_children_ids)} 个)")

        # 1. 清空唯一子节点的 children_ids
        single_child.children_ids = []

        # 2. 将孙子节点提升为章的直接子节点
        chapter.children_ids = []
        for j, grand_child_id in enumerate(grand_children_ids):
            grand_child = session.get_node(grand_child_id)
            if grand_child:
                # 更新层级和父节点
                grand_child.level = chapter.level + 1
                grand_child.parent_id = chapter.node_id
                # 添加到章的子节点列表
                chapter.children_ids.append(grand_child_id)

        # 3. 删除原来的唯一子节点
        single_child_id = single_child.node_id
        if single_child_id in session.node_registry:
            del session.node_registry[single_child_id]

        if self.enable_logging:
            print_and_flush(f"    提升完成: 删除 '{single_child.title[:20]}...', 新增 {len(chapter.children_ids)} 个子节点")

        return len(chapter.children_ids)

    def _merge_single_child_to_chapter(self, session, chapter, single_child) -> bool:
        """
        将唯一子节点合并到章节
        当章节只有一个子节点且该子节点无孙节点时，
        将子节点的描述和字数合并到章节，使章节变为叶子节点。

        Returns:
            是否成功合并
        """
        import re

        # 合并描述
        if single_child.description:
            # 移除编号前缀（如 "1.1 "、"2.1 " 等）
            clean_desc = re.sub(r'^\d+\.\d+\s*', '', single_child.description)

            if chapter.description:
                chapter.description = f"{chapter.description}\n{clean_desc}"
            else:
                chapter.description = clean_desc

        # 继承字数
        chapter.allocated_words = single_child.allocated_words

        # 清空子节点列表
        chapter.children_ids = []

        # 删除子节点
        single_child_id = single_child.node_id
        if single_child_id in session.node_registry:
            del session.node_registry[single_child_id]

        if self.enable_logging:
            print_and_flush(f"  [结构优化] 合并单子节点: '{single_child.title[:20]}...' -> '{chapter.title[:20]}...'")

        return True

    def _detect_numerical_consistency_issues(self, chapters: List[Dict[str, Any]]) -> List[str]:
        """
        检测跨章节数值一致性问题
        检查：
        1. 金额一致性（如投标保证金在不同章节中不一致）
        2. 百分比/权重一致性（如评分权重 60/40 vs 30/40/30）
        3. 数量一致性（如项目数量、人员数量等）

        示例问题：
        - 第一章写"投标保证金4万元"，第二章写"投标保证金2万元"
        - 评分标准一处写"技术权重60%"，另一处写"技术权重30%"
        """
        issues = []

        # 收集所有数值信息
        # 格式: {数值类型: {数值标识: [(章节索引, 具体值, 原文)]}}
        numerical_data = {
            "金额": {},      # 如 "投标保证金": [(0, 40000, "4万元"), (1, 20000, "2万元")]
            "百分比": {},    # 如 "技术权重": [(0, 60, "60%"), (1, 30, "30%")]
            "比例": {},      # 如 "分配比例": [(0, "60:40", "60/40")]
            "数量": {},      # 如 "项目数": [(0, 5, "5个项目")]
        }

        for i, chapter in enumerate(chapters):
            title = chapter["title"]
            description = chapter.get("description", "")
            combined_text = f"{title} {description}"

            # 检查子节点内容
            for child in chapter.get("children", []):
                combined_text += f" {child.get('title', '')} {child.get('description', '')}"

            # 1. 提取金额信息
            self._extract_amounts(combined_text, i, numerical_data["金额"])

            # 2. 提取百分比信息
            self._extract_percentages(combined_text, i, numerical_data["百分比"])

            # 3. 提取比例信息
            self._extract_ratios(combined_text, i, numerical_data["比例"])

            # 4. 提取数量信息
            self._extract_quantities(combined_text, i, numerical_data["数量"])

        # 分析数值一致性
        issues.extend(self._analyze_amount_consistency(numerical_data["金额"]))
        issues.extend(self._analyze_percentage_consistency(numerical_data["百分比"]))
        issues.extend(self._analyze_ratio_consistency(numerical_data["比例"]))
        issues.extend(self._analyze_quantity_consistency(numerical_data["数量"]))

        return issues

    def _extract_amounts(self, text: str, chapter_idx: int, amount_dict: Dict) -> None:
        """
        提取金额信息

        匹配模式：
        - X万元、X万、X元
        - ¥X、￥X
        - X百万、X千万、X亿
        """
        # 常见金额标识关键词
        amount_keywords = [
            '投标保证金', '保证金', '履约保证金', '履约担保',
            '预算', '金额', '费用', '成本', '总价', '合同金额',
            '投标报价', '报价', '招标金额', '中标金额',
            'budget', 'cost', 'price', 'amount'
        ]

        # 匹配金额数值
        # 1. 匹配"X万元"格式
        wan_pattern = r'(\d+(?:\.\d+)?)\s*万\s*元'
        for match in re.finditer(wan_pattern, text):
            value = float(match.group(1)) * 10000  # 转换为元
            self._add_numerical_entry(amount_dict, text, match.start(), value, match.group(0), chapter_idx, amount_keywords)

        # 2. 匹配"X元"格式（排除已匹配的万元）
        yuan_pattern = r'(\d+(?:\.\d+)?)\s*元'
        for match in re.finditer(yuan_pattern, text):
            if '万' not in match.group(0):  # 避免重复匹配
                value = float(match.group(1))
                self._add_numerical_entry(amount_dict, text, match.start(), value, match.group(0), chapter_idx, amount_keywords)

        # 3. 匹配 ¥/￥ 符号
        currency_pattern = r'[￥¥]\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(currency_pattern, text):
            value = float(match.group(1))
            self._add_numerical_entry(amount_dict, text, match.start(), value, match.group(0), chapter_idx, amount_keywords)

        # 4. 匹配"X亿"、"X千万"、"X百万"
        large_amount_pattern = r'(\d+(?:\.\d+)?)\s*(亿|千万|百万)'
        for match in re.finditer(large_amount_pattern, text):
            multiplier_map = {'亿': 100000000, '千万': 10000000, '百万': 1000000}
            value = float(match.group(1)) * multiplier_map.get(match.group(2), 1)
            self._add_numerical_entry(amount_dict, text, match.start(), value, match.group(0), chapter_idx, amount_keywords)

    def _extract_percentages(self, text: str, chapter_idx: int, percentage_dict: Dict) -> None:
        """
        提取百分比信息

        匹配模式：
        - X%、X百分比
        - 权重X%、比例X%
        """
        # 常见百分比标识关键词
        percentage_keywords = [
            '权重', '比例', '占比', '份额', '评分', '权重占比',
            'weight', 'ratio', 'percentage', 'score', 'allocation'
        ]

        # 匹配 X% 格式
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(percent_pattern, text):
            value = float(match.group(1))
            self._add_numerical_entry(percentage_dict, text, match.start(), value, match.group(0), chapter_idx, percentage_keywords)

    def _extract_ratios(self, text: str, chapter_idx: int, ratio_dict: Dict) -> None:
        """
        提取比例信息

        匹配模式：
        - X/Y、X:Y、X比Y
        - 如 60/40、3:2、60比40
        """
        # 常见比例标识关键词
        ratio_keywords = [
            '分配', '比例', '权重', '分割', '分成',
            'allocation', 'ratio', 'split', 'distribution'
        ]

        # 匹配 X/Y 或 X:Y 格式
        ratio_pattern = r'(\d+)\s*[/:\：]\s*(\d+)'
        for match in re.finditer(ratio_pattern, text):
            ratio_str = f"{match.group(1)}/{match.group(2)}"
            self._add_numerical_entry(ratio_dict, text, match.start(), ratio_str, match.group(0), chapter_idx, ratio_keywords)

        # 匹配 X比Y 格式
        ratio_word_pattern = r'(\d+)\s*比\s*(\d+)'
        for match in re.finditer(ratio_word_pattern, text):
            ratio_str = f"{match.group(1)}/{match.group(2)}"
            self._add_numerical_entry(ratio_dict, text, match.start(), ratio_str, match.group(0), chapter_idx, ratio_keywords)

    def _extract_quantities(self, text: str, chapter_idx: int, quantity_dict: Dict) -> None:
        """
        提取数量信息

        匹配模式：
        - X个、X项、X条、X种
        - X名、X人、X位
        - X件、X套、X台
        """
        # 常见数量标识关键词
        quantity_keywords = [
            '项目', '人员', '设备', '系统', '任务', '要求', '条款', '条件',
            'item', 'person', 'device', 'system', 'task', 'requirement'
        ]

        # 匹配数量格式
        quantity_pattern = r'(\d+)\s*(个|项|条|种|名|人|位|件|套|台|份|组)'
        for match in re.finditer(quantity_pattern, text):
            value = int(match.group(1))
            self._add_numerical_entry(quantity_dict, text, match.start(), value, match.group(0), chapter_idx, quantity_keywords)

    def _add_numerical_entry(
        self,
        data_dict: Dict,
        text: str,
        match_pos: int,
        value: Any,
        raw_match: str,
        chapter_idx: int,
        keywords: List[str]
    ) -> None:
        """
        将数值信息添加到字典中

        Args:
            data_dict: 数值字典
            text: 完整文本
            match_pos: 匹配位置
            value: 数值（可能是数字或字符串如"60/40")
            raw_match: 原始匹配文本
            chapter_idx: 章节索引
            keywords: 关键词列表，用于确定数值标识
        """
        # 在匹配位置前后查找关键词
        context_before = text[:match_pos]
        context_after = text[match_pos + len(raw_match):]

        # 查找最近的关键词作为标识
        best_keyword = None
        min_distance = float('inf')

        for keyword in keywords:
            # 查找关键词在匹配前后的位置
            before_pos = context_before.rfind(keyword)
            if before_pos != -1:
                distance = match_pos - (before_pos + len(keyword))
                if distance < min_distance and distance < 30:  # 关键词距离不超过30字符
                    min_distance = distance
                    best_keyword = keyword

            after_pos = context_after.find(keyword)
            if after_pos != -1:
                distance = after_pos
                if distance < min_distance and distance < 30:
                    min_distance = distance
                    best_keyword = keyword

        # 如果没有找到关键词，使用通用标识
        if best_keyword is None:
            best_keyword = "数值"

        # 添加到字典
        if best_keyword not in data_dict:
            data_dict[best_keyword] = []

        data_dict[best_keyword].append((chapter_idx, value, raw_match))

    def _analyze_amount_consistency(self, amount_dict: Dict) -> List[str]:
        """
        分析金额一致性

        检查同一标识的金额在不同章节是否一致
        """
        issues = []

        for keyword, entries in amount_dict.items():
            if len(entries) < 2:
                continue  # 只出现一次，无需检查

            # 比较数值
            values = [entry[1] for entry in entries]
            unique_values = set(values)

            if len(unique_values) > 1:
                # 存在不一致的金额
                first_entry = entries[0]
                for entry in entries[1:]:
                    if entry[1] != first_entry[1]:
                        diff_ratio = abs(entry[1] - first_entry[1]) / max(first_entry[1], entry[1], 1)
                        if diff_ratio > 0.1:  # 差异超过10%才报错
                            issues.append(
                                f"[金额矛盾] '{keyword}'在第{first_entry[0]+1}章为'{first_entry[2]}'，"
                                f"但在第{entry[0]+1}章为'{entry[2]}'"
                            )

        return issues

    def _analyze_percentage_consistency(self, percentage_dict: Dict) -> List[str]:
        """
        分析百分比一致性

        检查同一标识的百分比在不同章节是否一致
        """
        issues = []

        for keyword, entries in percentage_dict.items():
            if len(entries) < 2:
                continue

            values = [entry[1] for entry in entries]
            unique_values = set(values)

            if len(unique_values) > 1:
                first_entry = entries[0]
                for entry in entries[1:]:
                    if entry[1] != first_entry[1]:
                        # 百分比差异超过5%才报错
                        if abs(entry[1] - first_entry[1]) > 5:
                            issues.append(
                                f"[权重矛盾] '{keyword}'在第{first_entry[0]+1}章为'{first_entry[2]}'，"
                                f"但在第{entry[0]+1}章为'{entry[2]}'"
                            )

        # 额外检查：权重总和是否合理（应接近100%）
        weight_entries = percentage_dict.get('权重', [])
        if weight_entries:
            # 检查同一章节内的权重总和
            chapter_weights = {}  # {章节索引: [(关键词, 值)]}
            for keyword, entries in percentage_dict.items():
                for entry in entries:
                    if '权重' in keyword or keyword == '权重':
                        chapter_idx = entry[0]
                        if chapter_idx not in chapter_weights:
                            chapter_weights[chapter_idx] = []
                        chapter_weights[chapter_idx].append((keyword, entry[1]))

            for chapter_idx, weights in chapter_weights.items():
                total = sum(w[1] for w in weights)
                if len(weights) >= 2 and abs(total - 100) > 15:  # 权重总和偏离100超过15%
                    weight_str = "、".join([f"{w[0]}={w[1]}%" for w in weights])
                    issues.append(
                        f"[权重矛盾] 第{chapter_idx+1}章权重总和为{total}%（{weight_str}），偏离标准100%"
                    )

        return issues

    def _analyze_ratio_consistency(self, ratio_dict: Dict) -> List[str]:
        """
        分析比例一致性

        检查同一标识的比例在不同章节是否一致
        """
        issues = []

        for keyword, entries in ratio_dict.items():
            if len(entries) < 2:
                continue

            # 比例是字符串格式如"60/40"
            ratio_values = [entry[1] for entry in entries]
            unique_values = set(ratio_values)

            if len(unique_values) > 1:
                first_entry = entries[0]
                for entry in entries[1:]:
                    if entry[1] != first_entry[1]:
                        issues.append(
                            f"[比例矛盾] '{keyword}'在第{first_entry[0]+1}章为'{first_entry[2]}'，"
                            f"但在第{entry[0]+1}章为'{entry[2]}'"
                        )

        return issues

    def _analyze_quantity_consistency(self, quantity_dict: Dict) -> List[str]:
        """
        分析数量一致性

        检查同一标识的数量在不同章节是否一致
        """
        issues = []

        for keyword, entries in quantity_dict.items():
            if len(entries) < 2:
                continue

            values = [entry[1] for entry in entries]
            unique_values = set(values)

            if len(unique_values) > 1:
                first_entry = entries[0]
                for entry in entries[1:]:
                    if entry[1] != first_entry[1]:
                        # 数量差异超过1才报错
                        if abs(entry[1] - first_entry[1]) > 1:
                            issues.append(
                                f"[数量矛盾] '{keyword}'在第{first_entry[0]+1}章为'{first_entry[2]}'，"
                                f"但在第{entry[0]+1}章为'{entry[2]}'"
                            )

        return issues