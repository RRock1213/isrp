"""
ISRP 核心数据结构

定义大纲树的所有数据类型与任务会话管理器：
  - OutlineNode:              多叉树节点（支持扁平哈希索引）
  - NodeStatus:                节点生命周期状态（Draft -> Expanded -> Finalized）
  - PlanSession:               任务会话，贯穿全流程保存中间状态
  - KeyInfoTracker:            关键信息追踪器（防止信息遗忘）
  - ExplicitChapterFrame:      用户显式章节框架提取
  - StyleExtractionResult:     写作风格识别结果
  - PackageInfo / CompressedPrompt 等辅助结构
  - robust_json_parse:         LLM 输出的鲁棒 JSON 解析

设计原则:
  - 解耦内容规划与字数分配：DFS 建树时不锁定字数，树成型后 BFS 分配
  - 生成-评估-优化循环：每个节点经历 initial -> evaluate -> refine 迭代
  - 直接对接 Write 模块的段落格式

"""
import asyncio
import json
import re
import os
import time
import uuid
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from pathlib import Path


# ============================================================================
# LLM Temperature 枚举 - 不同场景使用不同温度
# ============================================================================

class LLMRole(Enum):
    """LLM 角色枚举，对应不同的温度值"""
    CREATOR = 0.7      # 用于生成大纲节点、发散思路（根节点建立、DFS 节点生成）
    SUMMARIZER = 0.5   # 用于写全局摘要
    REVISER = 0.3      # 用于反思、合并、拆分（根节点反思、字数分配重组）
    JUDGE = 0.2        # 用于 6 维打分、布尔值判定（DFS 评估、章节检查点）
    CALCULATOR = 0.1   # 用于评估权重、格式化输出转换（字数分配、格式化）


class InfoType(str, Enum):
    """关键信息类型枚举"""
    ENTITY = "entity"       # 实体（人名、地名、机构名）
    NUMBER = "number"       # 数字信息
    DATE = "date"           # 日期信息
    REQUIREMENT = "requirement"  # 格式要求


class FactType(str, Enum):
    """专业写作类型枚举"""
    TRUE_CRIME = "True Crime"     # 真实犯罪纪实
    NEWS = "News"                  # 新闻报道
    ACADEMIC = "Academic"          # 学术论文
    CASE_STUDY = "Case Study"      # 案例分析
    REPORT = "Report"              # 实习报告/工作报告
    APPLICATION = "Application"    # 申请书


# ============================================================================
# Pydantic Schemas - LLM 结构化输出定义
# ============================================================================

class NodeGenerationResult(BaseModel):
    """DFS 节点生成与深化结果

    移除未使用字段：
    - needs_horizontal_expansion: 横向扩展由 _horizontal_probe 方法独立实现
    - is_perfect: 由代码根据评分计算 (avg_score >= 9.0)
    """
    thinking_process: str = Field(description="思考过程：分析前置节点的连贯性以及当前节点应包含的内容")
    title: str = Field(description="当前节点的标题，例如：1.1 时代背景")
    description: str = Field(description="当前节点的核心内容阐述，越详细越好")
    core_concepts: List[str] = Field(description="提取3-5个专有名词或核心概念标签，用于防重", default_factory=list)
    needs_vertical_expansion: bool = Field(description="当前节点的内容是否过于宏大，需要进一步细分为下一级子节点？")


class RubricEvaluation(BaseModel):
    """6 维评分基准对齐（根节点建立、DFS 扩展、章节检查点的反思环节）"""
    analysis: str = Field(description="对六个维度的综合评估分析")
    relevance: int = Field(ge=1, le=10, description="相关性 (1-10): 需求相关性")
    accuracy: int = Field(ge=1, le=10, description="准确性 (1-10): 事实与逻辑准确性")
    coherence: int = Field(ge=1, le=10, description="连贯性 (1-10): 结构连贯性与上下文衔接")
    clarity: int = Field(ge=1, le=10, description="清晰度 (1-10): 表达清晰度与细节丰富度")
    breadth_depth: int = Field(ge=1, le=10, description="广度深度 (1-10): 内容广度与深度")
    reading_experience: int = Field(ge=1, le=10, description="阅读体验 (1-10): 阅读体验")

    @property
    def average_score(self) -> float:
        """计算平均分"""
        return (self.relevance + self.accuracy + self.coherence +
                self.clarity + self.breadth_depth + self.reading_experience) / 6.0

    @property
    def overall_passed(self) -> bool:
        """六维均分是否达到 8.0 以上，且无逻辑硬伤"""
        return self.average_score >= 8.0


class ChapterCheckpointResult(BaseModel):
    """章级 Checkpoint 与摘要生成

    Schema 与 PROMPT_4_CHECKPOINT 完全对齐
    """
    evaluation: RubricEvaluation = Field(description="针对本整章的6维评估")
    issues: List[str] = Field(default_factory=list, description="发现的问题列表")
    suggestions: List[str] = Field(default_factory=list, description="改进建议列表")
    needs_revision: bool = Field(description="是否需要退回修改结构？")
    chapter_abstract: str = Field(description="本章的全局摘要（100字左右），用于指导下一章的生成以保证衔接。")
    transition_quality: str = Field(default="", description="与前序章节的衔接评价")
    structure_issues: List[str] = Field(default_factory=list, description="结构问题列表（如结论位置异常等）")
    patches: List[Dict[str, Any]] = Field(default_factory=list, description="修复补丁列表")


class NodeWeight(BaseModel):
    """节点权重分配"""
    node_id: str = Field(description="保持与输入一致的节点 ID")
    reasoning: str = Field(description="为什么给这个权重？")
    relative_weight: int = Field(ge=1, le=10, description="相对权重 (1-10)")


class BFSAllocationResult(BaseModel):
    """BFS 全局结算与权重分配"""
    allocations: List[NodeWeight] = Field(description="各兄弟节点的权重分配列表")


class ReorganizedNode(BaseModel):
    """重组后的节点"""
    level: int = Field(ge=1, le=3, description="节点的层级 (1-3)")
    title: str = Field(description="合并或拆分后的新标题")
    description: str = Field(description="调整后的正文内容大纲")
    allocated_words: int = Field(ge=100, description="调整后该节点分配的字数（需在400-800之间）")


class TreeReorganizationResult(BaseModel):
    """全局打磨与甜点重组"""
    analysis: str = Field(description="合并与拆分操作的思考链")
    flattened_nodes: List[ReorganizedNode] = Field(description="修改后的全树平铺列表（先序遍历顺序）")


def normalize_structure_text(value: Any) -> str:
    """Normalize LLM structure outputs into text for root establishment schemas."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, (list, dict)):
                item = normalize_structure_text(item)
            parts.append(f"{key}: {item}")
        return "\n".join(parts)
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, dict):
                title = (
                    item.get("section")
                    or item.get("title")
                    or item.get("name")
                    or f"Part {index}"
                )
                details = []
                for key, detail in item.items():
                    if key in {"section", "title", "name"}:
                        continue
                    if isinstance(detail, (list, dict)):
                        detail = normalize_structure_text(detail)
                    details.append(f"{key}: {detail}")
                suffix = f" - {'; '.join(details)}" if details else ""
                lines.append(f"{index}. {title}{suffix}")
            else:
                lines.append(f"{index}. {normalize_structure_text(item)}")
        return "\n".join(lines)
    return str(value)


class RootDivergenceResult(BaseModel):
    """根节点建立 — 意图发散结果"""
    angle_name: str = Field(description="切入角度名称")
    rationale: str = Field(description="为什么选择这个角度，它的优势是什么")
    high_level_structure: str = Field(description="宏观的架构简述")

    @field_validator("high_level_structure", mode="before")
    @classmethod
    def normalize_high_level_structure(cls, value: Any) -> str:
        return normalize_structure_text(value)


# ============================================================================

# ============================================================================

class ChapterFrameItem(BaseModel):
    """章节框架项"""
    title: str = Field(description="章节标题，如：摘要、引言、方法")
    needs_expansion: bool = Field(
        default=True,
        description="是否需要展开为子节点。固定格式章节(如参考文献)设为 false"
    )
    estimated_weight: int = Field(
        default=5,
        ge=1, le=10,
        description="预估权重(1-10)，用于字数分配参考"
    )
    is_fixed_format: bool = Field(
        default=False,
        description="是否为固定格式章节(如标题页、参考文献)"
    )


class ExplicitChapterFrame(BaseModel):
    """明确的章节框架（从用户 Prompt 中提取）"""
    has_explicit_frame: bool = Field(description="是否提供了明确框架")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")
    chapters: List[ChapterFrameItem] = Field(default_factory=list, description="章节列表")
    frame_source: str = Field(default="", description="框架来源原文")
    word_allocation_hint: str = Field(default="", description="字数分配提示")


class FrameValidationResult(BaseModel):
    """章节框架验证结果"""
    is_valid: bool = Field(description="框架是否合理")
    missing_chapters: List[str] = Field(default_factory=list, description="缺少的章节")
    order_issues: List[str] = Field(default_factory=list, description="顺序问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")
    adjusted_chapters: List[str] = Field(default_factory=list, description="调整后的章节列表")


# ============================================================================
# 分包信息 Schema - 解决分包语义误解问题
# ============================================================================

class PackageType(str, Enum):
    """分包类型枚举"""
    CONSTRUCTION = "construction"    # 施工包/建设包
    SUPERVISION = "supervision"      # 监理包/第三方监督
    ACCEPTANCE = "acceptance"        # 验收包/测试评审
    DESIGN = "design"                # 设计包
    EQUIPMENT = "equipment"          # 设备采购包
    SERVICE = "service"              # 服务包
    OTHER = "other"                  # 其他类型


class PackageItem(BaseModel):
    """单个分包信息"""
    package_id: str = Field(description="分包标识（如 Package A, 第一包, 分包1）")
    package_name: str = Field(default="", description="分包名称/内容描述")
    package_type: PackageType = Field(default=PackageType.OTHER, description="分包类型")
    budget: str = Field(default="", description="预算金额（原始文本格式）")
    budget_value: Optional[float] = Field(default=None, description="预算金额数值")
    description: str = Field(default="", description="分包详细描述/工作内容")


class PackageInfo(BaseModel):
    """分包信息汇总（新增）

    用于解决分包语义误解问题：
    - 正确识别分包类型（监理、验收、施工等）
    - 记录各分包的预算金额
    - 为大纲生成注入正确的分包上下文

    示例：
    招标公告原文：
    - Package A: 平台建设（施工包）- ¥1,506,230
    - Package B: 监理（第三方监督）- ¥20,260
    - Package C: 验收（测试评审）- ¥80,000

    检测结果：
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
        ],
        package_summary="Package A 为施工包，Package B 为监理包，Package C 为验收包"
    )
    """
    has_packages: bool = Field(default=False, description="是否包含分包信息")
    package_count: int = Field(default=0, description="分包数量")
    packages: List[PackageItem] = Field(default_factory=list, description="分包列表")
    package_summary: str = Field(default="", description="分包关系理解摘要")


class StyleExtractionResult(BaseModel):
    """查询风格识别结果"""
    writing_type: str = Field(description="写作类型，如：自媒体文章、科普文章、学术论文、作文、报刊文章等")
    style_keywords: List[str] = Field(description="风格关键词列表，如：['接地气', '口语化', '易读']")
    tone_guidance: str = Field(description="语气语调指导，如：轻松活泼、严肃正式、亲切自然")
    structure_preference: str = Field(description="结构偏好，如：引人入胜式、总分总、起承转合")
    language_style: str = Field(description="语言风格，如：口语化、书面语、学术化")
    target_audience: str = Field(default="", description="目标读者群体（可选）")
    
    requires_real_facts: bool = Field(default=False, description="是否需要基于真实事实的写作类型")
    fact_type: Optional[FactType] = Field(default=None, description="事实类型（当requires_real_facts为True时）")
    
    key_entities: List[str] = Field(
        default_factory=list,
        description="关键实体列表（人名、地名、机构名等），必须确保这些实体在生成内容中出现"
    )
    key_requirements: List[str] = Field(
        default_factory=list,
        description="必须满足的格式要求列表（如：吸引人的标题、特定格式等）"
    )
    
    explicit_chapter_frame: Optional[ExplicitChapterFrame] = Field(
        default=None,
        description="从用户 Prompt 中提取的明确章节框架（如有）"
    )
    
    core_subject: str = Field(
        default="",
        description="核心主题（如'节能建筑'、'单细胞数据分析'、'党性分析'）"
    )
    theme_keywords: List[str] = Field(
        default_factory=list,
        description="主题关键词列表，用于注入到章节描述中"
    )
    subject_domain: str = Field(
        default="",
        description="学科领域（如'建筑学'、'生物医学'、'政治学'）"
    )
    
    user_provided_facts: List[str] = Field(
        default_factory=list,
        description="用户提供的具体事实信息（时间、地点、人物、数据、事件等），必须在生成内容中使用"
    )
    # 分包信息提取 - 解决分包语义误解问题
    package_info: Optional[PackageInfo] = Field(
        default=None,
        description="分包信息（用于招标/投标类文档，确保正确理解分包类型）"
    )


class RootRefinementResult(BaseModel):
    """根节点建立 — 串行反思与深化结果
    
    注意：LLM 返回的是扁平结构，evaluation 字段内嵌在顶层
    """
    analysis: str = Field(default="", description="对六个维度的综合评估分析")
    relevance: int = Field(default=7, ge=1, le=10, description="相关性 (1-10)")
    accuracy: int = Field(default=7, ge=1, le=10, description="准确性 (1-10)")
    coherence: int = Field(default=7, ge=1, le=10, description="连贯性 (1-10)")
    clarity: int = Field(default=7, ge=1, le=10, description="清晰度 (1-10)")
    breadth_depth: int = Field(default=7, ge=1, le=10, description="广度深度 (1-10)")
    reading_experience: int = Field(default=7, ge=1, le=10, description="阅读体验 (1-10)")
    is_perfect: bool = Field(default=False, description="当前方案是否已完善，无需进一步修改")
    refined_angle_name: str = Field(default="", description="深化后的角度名称（如需修改）")
    refined_rationale: str = Field(default="", description="深化后的理由（如需修改）")
    refined_structure: str = Field(default="", description="深化后的结构（如需修改）")

    @field_validator("refined_structure", mode="before")
    @classmethod
    def normalize_refined_structure(cls, value: Any) -> str:
        return normalize_structure_text(value)

    @property
    def average_score(self) -> float:
        """计算平均分"""
        return (self.relevance + self.accuracy + self.coherence +
                self.clarity + self.breadth_depth + self.reading_experience) / 6.0

    @property
    def evaluation(self) -> RubricEvaluation:
        """兼容旧代码，返回 RubricEvaluation 对象"""
        return RubricEvaluation(
            analysis=self.analysis,
            relevance=self.relevance,
            accuracy=self.accuracy,
            coherence=self.coherence,
            clarity=self.clarity,
            breadth_depth=self.breadth_depth,
            reading_experience=self.reading_experience
        )


class GlobalAbstractResult(BaseModel):
    """全局摘要结果"""
    global_abstract: str = Field(description="文章的核心主旨、起承转合基调、以及主要任务")
    total_word_budget: int = Field(description="确认的总字数预算")


class CompressedPrompt(BaseModel):
    """Prompt 智能压缩结果

    当用户原始 prompt 超过阈值时，使用 LLM 进行信息提纯和压缩。
    """
    original_prompt: str = Field(description="原始用户 prompt")
    compressed_prompt: str = Field(description="压缩后的 prompt")
    core_topic: str = Field(default="", description="核心主题（一句话概括）")
    key_requirements: List[str] = Field(default_factory=list, description="关键需求列表")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="约束条件（字数、风格、格式等）")
    must_include: List[str] = Field(default_factory=list, description="必须包含的信息")

    @property
    def compression_ratio(self) -> float:
        """压缩比例"""
        if len(self.original_prompt) == 0:
            return 0.0
        return 1.0 - (len(self.compressed_prompt) / len(self.original_prompt))


# ============================================================================
# 核心数据结构 - OutlineNode
# ============================================================================

class NodeStatus(str, Enum):
    """节点状态枚举"""
    PENDING = "Pending"
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"
    FINALIZED = "Finalized"  # 新增：已完善状态


@dataclass
class OutlineNode:
    """
    大纲节点 数据结构

    特性：
    - 支持多叉树结构
    - 维护全局扁平哈希表指针
    - 支持延迟字数分配
    - 支持核心概念标签（用于全局去重）
    - 支持 prev_sibling_id 实现极速横向上下文获取
    """
    # 基础属性
    node_id: str
    title: str
    description: str
    level: int  # 0: Root, 1: 章, 2: 节, 3: 叶子节点(关键点)

    # 树结构指针
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    prev_sibling_id: Optional[str] = None  # 新增：前置兄弟节点ID

    # 字数相关
    word_budget: int = 0
    allocated_words: int = 0  # 新增：最终分配字数
    original_budget: int = 0

    # 评估与状态
    score: float = 0.0
    status: NodeStatus = NodeStatus.PENDING
    is_finalized: bool = False  # 新增：是否已完善

    
    core_concepts: List[str] = field(default_factory=list)
    relative_weight: int = 5  # 默认权重为5 (1-10)
    chapter_abstract: str = ""  # 仅 Level 1 使用
    mandatory_points: str = ""  # 基于6维细则的写作要点

    # DFS 动态伸缩标识
    needs_vertical_expansion: bool = True  # 默认值改为 True，确保节点默认需要扩展
    # needs_horizontal_expansion: 横向扩展由 _horizontal_probe 方法独立实现

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "prev_sibling_id": self.prev_sibling_id,
            "level": self.level,
            "title": self.title,
            "description": self.description,
            "word_budget": self.word_budget,
            "allocated_words": self.allocated_words,
            "original_budget": self.original_budget,
            "score": self.score,
            "status": self.status.value,
            "is_finalized": self.is_finalized,
            "children_count": len(self.children_ids),
            "core_concepts": self.core_concepts,
            "relative_weight": self.relative_weight,
            "chapter_abstract": self.chapter_abstract,
            "mandatory_points": self.mandatory_points,
            "metadata": self.metadata,
        }

    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children_ids) == 0

    # 特殊章节关键词（不编号为"第N章"）
    SPECIAL_CHAPTER_KEYWORDS = [
        '摘要', 'abstract', '参考文献', 'reference', 'bibliography',
        '致谢', 'acknowledgement', '附录', 'appendix', '标题页', 'title page'
    ]

    def _is_special_chapter(self, title: str) -> bool:
        """
        判断是否为特殊章节（摘要、参考文献等）

        特殊章节不编号为"第N章"，而是直接使用标题
        """
        title_lower = title.lower()
        for keyword in self.SPECIAL_CHAPTER_KEYWORDS:
            if keyword.lower() in title_lower:
                return True
        return False

    def get_path_context(self, node_registry: Dict[str, 'OutlineNode']) -> str:
        """
        获取从根节点到当前节点的完整路径上下文（带章节编号）

        Args:
            node_registry: 全局节点注册表

        Returns:
            路径字符串（如：【第一章：引言】-【1.1 时代背景】-【1.1.1 具体内容】）

        - 递归获取父节点的完整编号，正确处理三级及以上节点
        - 检测并去除标题中已有的编号前缀，避免重复
        - 特殊章节（摘要、参考文献等）不编号为"第N章"
        """
        path_parts = []
        current = self

        while current is not None:
            parent_id = current.parent_id
            if parent_id is None:
                if current.level == 0:
                    path_parts.append(f"【{current.title}】")
                break

            parent = node_registry.get(parent_id)
            if parent is None:
                break

            sibling_index = 0
            for i, sibling_id in enumerate(parent.children_ids, 1):
                if sibling_id == current.node_id:
                    sibling_index = i
                    break

            # 获取原始标题
            raw_title = current.title

            # 去除标题中已有的编号前缀（如 "1.1 ", "2.3.1 " 等）
            title_without_index = re.sub(r'^[\d.]+\s*', '', raw_title).strip()

            if parent.level == 0:
                # 父节点是根节点
                # 检查是否为特殊章节
                if self._is_special_chapter(title_without_index):
                    # 特殊章节不编号，直接使用标题
                    path_parts.append(f"【{title_without_index}】")
                else:
                    # 普通章节生成章节格式
                    chapter_num = sibling_index
                    prefix = f"第{chapter_num}章"
                    path_parts.append(f"【{prefix}：{title_without_index}】")
            else:
                # 递归获取父节点的完整编号
                parent_num = self._get_node_number(parent, node_registry)
                if parent_num:
                    section_num = f"{parent_num}.{sibling_index}"
                else:
                    section_num = f"{sibling_index}"

                path_parts.append(f"【{section_num} {title_without_index}】")

            current = parent

        return " - ".join(reversed(path_parts))

    def _get_node_number(self, node: 'OutlineNode', node_registry: Dict[str, 'OutlineNode']) -> str:
        """
        递归获取节点的完整编号

        Args:
            node: 目标节点
            node_registry: 节点注册表

        Returns:
            节点编号字符串（如 "1.3" 或 "2.1"）
        """
        if node.parent_id is None:
            return ""

        parent = node_registry.get(node.parent_id)
        if parent is None:
            return ""

        # 找到当前节点在父节点子列表中的索引
        sibling_index = 0
        for i, sibling_id in enumerate(parent.children_ids, 1):
            if sibling_id == node.node_id:
                sibling_index = i
                break

        if parent.level == 0:
            # 父节点是根节点，返回章节号
            return f"{sibling_index}"
        else:
            # 递归获取父节点的编号
            parent_num = self._get_node_number(parent, node_registry)
            if parent_num:
                return f"{parent_num}.{sibling_index}"
            else:
                return f"{sibling_index}"


# ============================================================================
# 任务会话管理器 - PlanSession
# ============================================================================

class ConceptRecord(BaseModel):
    """概念记录：追踪概念的来源和层级"""
    concept: str = Field(description="概念名称")
    source_node_id: str = Field(description="来源节点ID")
    source_node_title: str = Field(description="来源节点标题")
    level: int = Field(description="节点层级")
    created_at: float = Field(default_factory=time.time, description="创建时间")


class KeyInfoRecord(BaseModel):
    """关键信息记录：追踪用户提供的特定信息"""
    info_type: InfoType = Field(description="信息类型：entity/number/date/requirement")
    content: str = Field(description="具体内容")
    source: str = Field(description="来源（user_prompt/context）")
    used: bool = Field(default=False, description="是否已在生成中使用")
    created_at: float = Field(default_factory=time.time, description="创建时间")


class KeyInfoTracker:
    """
    关键信息追踪器

    功能：
    1. 追踪用户指令中的关键实体（人名、地名、机构名等）
    2. 追踪格式要求（如"吸引人的标题"）
    3. 检查关键信息在生成内容中的覆盖率
    """

    def __init__(self):
        self.records: List[KeyInfoRecord] = []
        self._entity_set: Set[str] = set()
        self._requirement_set: Set[str] = set()

    def add_entity(self, entity: str, source: str = "user_prompt") -> None:
        """添加实体信息"""
        if entity and entity not in self._entity_set:
            self.records.append(KeyInfoRecord(
                info_type=InfoType.ENTITY,
                content=entity,
                source=source
            ))
            self._entity_set.add(entity)

    def add_requirement(self, requirement: str, source: str = "user_prompt") -> None:
        """添加格式要求"""
        if requirement and requirement not in self._requirement_set:
            self.records.append(KeyInfoRecord(
                info_type=InfoType.REQUIREMENT,
                content=requirement,
                source=source
            ))
            self._requirement_set.add(requirement)

    def add_number(self, number: str, source: str = "user_prompt") -> None:
        """添加数字信息"""
        if number:
            self.records.append(KeyInfoRecord(
                info_type=InfoType.NUMBER,
                content=number,
                source=source
            ))

    def add_date(self, date: str, source: str = "user_prompt") -> None:
        """添加日期信息"""
        if date:
            self.records.append(KeyInfoRecord(
                info_type=InfoType.DATE,
                content=date,
                source=source
            ))

    def get_unused_requirements(self) -> List[str]:
        """获取未使用的格式要求"""
        return [r.content for r in self.records
                if r.info_type == InfoType.REQUIREMENT and not r.used]

    def get_all_requirements(self) -> List[str]:
        """获取所有格式要求"""
        return [r.content for r in self.records
                if r.info_type == InfoType.REQUIREMENT]

    def get_unused_entities(self) -> List[str]:
        """获取未使用的实体"""
        return [r.content for r in self.records
                if r.info_type == InfoType.ENTITY and not r.used]

    def get_all_entities(self) -> List[str]:
        """获取所有实体"""
        return [r.content for r in self.records
                if r.info_type == InfoType.ENTITY]

    def mark_used(self, content: str) -> None:
        """标记信息已使用"""
        for r in self.records:
            if r.content == content:
                r.used = True

    def mark_entity_used(self, entity: str) -> None:
        """标记实体已使用"""
        for r in self.records:
            if r.info_type == InfoType.ENTITY and r.content == entity:
                r.used = True

    def check_coverage(self) -> float:
        """检查信息覆盖率"""
        if not self.records:
            return 1.0
        used = sum(1 for r in self.records if r.used)
        return used / len(self.records)

    def get_coverage_report(self) -> Dict[str, Any]:
        """获取覆盖率报告"""
        total = len(self.records)
        if total == 0:
            return {"total": 0, "used": 0, "coverage": 1.0, "unused": []}

        used = sum(1 for r in self.records if r.used)
        unused = [r.content for r in self.records if not r.used]

        return {
            "total": total,
            "used": used,
            "coverage": used / total,
            "unused": unused
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "records": [r.model_dump() for r in self.records],
            "entities": list(self._entity_set),
            "requirements": list(self._requirement_set),
            "coverage": self.check_coverage()
        }


class PlanSession:
    """
    任务会话管理器

    功能：
    1. 管理全局扁平节点注册表 (node_registry)
    2. 管理全局已覆盖概念库 (covered_concepts_registry) - 用于避免内容重复
    3. 隔离不同 User Prompt 的内存
    4. 提供 O(1) 复杂度的节点访问
    5. 新增：关键信息追踪 (key_info_tracker)

    【概念库设计原则】
    - 目的：避免不同节点覆盖相同的内容，确保信息增量
    - 来源：节点的 core_concepts + 标题关键词
    - 追踪：记录概念来源节点，便于调试和分析
    """

    def __init__(self):
        self.node_registry: Dict[str, OutlineNode] = {}  # 全局节点注册表
        self.covered_concepts_registry: Dict[str, ConceptRecord] = {}  # 已覆盖概念库（带来源追踪）
        self._concept_set: Set[str] = set()  # 快速查找用
        self.root_id: Optional[str] = None  # 根节点ID
        self.global_target_words: int = 0  # 全局目标字数
        self.global_abstract: str = ""  # 全局摘要
        # 风格相关字段
        self.style: Optional['StyleExtractionResult'] = None  # 风格信息
        self.style_summary: str = ""  # 风格摘要字符串（用于注入 Prompt）
        self.original_prompt: str = ""  # 用户原始指令
        
        self.compressed_prompt: Optional['CompressedPrompt'] = None  # 压缩结果
        self.effective_prompt: str = ""  # 实际使用的 prompt（全局摘要及后续阶段使用）
        
        self.key_info_tracker: Optional[KeyInfoTracker] = None  # 关键信息追踪器

    def get_effective_prompt(self) -> str:
        """获取有效 prompt（优先使用压缩后的）"""
        return self.effective_prompt or self.original_prompt

    def get_key_requirements(self) -> List[str]:
        """获取关键需求（如果有压缩结果）"""
        if self.compressed_prompt:
            return self.compressed_prompt.key_requirements
        return []

    def register_node(self, node: OutlineNode) -> None:
        """注册节点到全局注册表"""
        self.node_registry[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[OutlineNode]:
        """O(1) 获取节点"""
        return self.node_registry.get(node_id)

    def get_children(self, node_id: str) -> List[OutlineNode]:
        """获取指定节点的所有子节点"""
        node = self.get_node(node_id)
        if node is None:
            return []
        return [self.get_node(cid) for cid in node.children_ids if self.get_node(cid) is not None]

    def get_prev_sibling(self, node_id: str) -> Optional[OutlineNode]:
        """获取前置兄弟节点"""
        node = self.get_node(node_id)
        if node is None or node.prev_sibling_id is None:
            return None
        return self.get_node(node.prev_sibling_id)

    def get_parent(self, node_id: str) -> Optional[OutlineNode]:
        """获取父节点"""
        node = self.get_node(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.get_node(node.parent_id)

    def add_to_concept_registry(
        self, 
        concepts: List[str], 
        source_node: Optional[OutlineNode] = None
    ) -> None:
        """
        将核心概念添加到已覆盖概念库
        
        Args:
            concepts: 概念列表
            source_node: 来源节点（可选，用于追踪）
        """
        for concept in concepts:
            normalized = concept.lower().strip()
            if normalized and normalized not in self._concept_set:
                record = ConceptRecord(
                    concept=normalized,
                    source_node_id=source_node.node_id if source_node else "unknown",
                    source_node_title=source_node.title if source_node else "unknown",
                    level=source_node.level if source_node else 0
                )
                self.covered_concepts_registry[normalized] = record
                self._concept_set.add(normalized)

    def is_concept_covered(self, concept: str) -> bool:
        """检查概念是否已被覆盖"""
        return concept.lower().strip() in self._concept_set
    
    def get_covered_concepts_by_level(self, level: int) -> List[str]:
        """获取特定层级的已覆盖概念"""
        return [
            record.concept for record in self.covered_concepts_registry.values()
            if record.level == level
        ]
    
    def get_covered_concepts_set(self) -> Set[str]:
        """获取已覆盖概念的 Set（用于快速查找）"""
        return self._concept_set.copy()
    
    def get_concept_source(self, concept: str) -> Optional[ConceptRecord]:
        """获取概念的来源信息"""
        return self.covered_concepts_registry.get(concept.lower().strip())

    # 兼容旧接口
    @property
    def global_concept_registry(self) -> Set[str]:
        """兼容旧接口：返回概念集合"""
        return self._concept_set

    def get_all_leaf_nodes(self) -> List[OutlineNode]:
        """获取所有叶子节点（按 DFS 顺序）"""
        if self.root_id is None:
            return []
        return self._dfs_collect_leaves(self.root_id)

    def _dfs_collect_leaves(self, node_id: str) -> List[OutlineNode]:
        """DFS 收集叶子节点（保持树的结构顺序）"""
        node = self.get_node(node_id)
        if node is None:
            return []

        # 如果是叶子节点，直接返回
        if node.is_leaf():
            return [node]

        # 否则递归收集子节点中的叶子
        leaves = []
        for child_id in node.children_ids:
            leaves.extend(self._dfs_collect_leaves(child_id))
        return leaves

    def get_all_nodes(self) -> List[OutlineNode]:
        """获取所有节点（先序遍历）"""
        if self.root_id is None:
            return []
        return self._dfs_collect(self.root_id)

    def _dfs_collect(self, node_id: str) -> List[OutlineNode]:
        """DFS 收集所有节点"""
        node = self.get_node(node_id)
        if node is None:
            return []
        result = [node]
        for child_id in node.children_ids:
            result.extend(self._dfs_collect(child_id))
        return result

    def set_style(self, style: 'StyleExtractionResult') -> None:
        """
        设置风格信息

        Args:
            style: 风格提取结果
        """
        self.style = style
        # 生成风格摘要字符串
        parts = [
            f"写作类型: {style.writing_type}",
            f"风格关键词: {', '.join(style.style_keywords)}",
            f"语气语调: {style.tone_guidance}",
            f"结构偏好: {style.structure_preference}",
            f"语言风格: {style.language_style}"
        ]
        if style.target_audience:
            parts.append(f"目标读者: {style.target_audience}")
        
        if style.requires_real_facts:
            parts.append(f"需要真实事实: 是")
            if style.fact_type:
                parts.append(f"事实类型: {style.fact_type}")
        self.style_summary = "\n".join(parts)

        
        self.key_info_tracker = KeyInfoTracker()
        # 添加关键实体
        for entity in style.key_entities:
            self.key_info_tracker.add_entity(entity)
        # 添加关键要求
        for requirement in style.key_requirements:
            self.key_info_tracker.add_requirement(requirement)

    def get_style_context(self) -> str:
        """
        获取风格上下文字符串

        Returns:
            格式化的风格上下文，用于注入 Prompt
        """
        if not self.style_summary:
            return "【写作风格】通用文章风格"
        return f"【写作风格】\n{self.style_summary}"

    def get_fact_type_guidance(self) -> str:
        """
        获取专业写作类型的指导信息

        Returns:
            专业写作类型的具体指导，用于注入根节点建立 Prompt
        """
        if not self.style or not self.style.requires_real_facts:
            return "本写作任务允许虚构和想象，无需严格基于真实事实。"

        fact_type = self.style.fact_type

        # 使用 FactType 枚举值作为键
        guidance_map = {
            FactType.TRUE_CRIME: """【True Crime 真实犯罪纪实故事】
- 必须基于真实案件，禁止虚构关键人物和情节
- 采用叙事性写作风格，而非学术分析报告
- 语言通俗易懂，具有故事性和可读性，避免过度学术化术语
- 结构建议：引人入胜的开头 → 案件背景 → 案发经过（细节描写）→ 调查过程 → 案件结果 → 引人深思的结尾
- 保持悬念和张力，但必须基于真实事实
- 可以适当加入人物心理描写和环境渲染，增强故事性""",

            FactType.NEWS: """【新闻报道】
- 必须基于真实事件，禁止虚构
- 遵循新闻写作的客观性原则
- 结构：导语（5W1H）→ 主体 → 结尾
- 语言简洁、准确、客观""",

            FactType.ACADEMIC: """【学术论文】
- 数据和引用必须真实准确
- 遵循学术写作规范
- 结构：摘要 → 引言 → 文献综述 → 方法 → 结果 → 讨论 → 结论
- 语言严谨、专业""",

            FactType.CASE_STUDY: """【案例分析】
- 必须基于真实案例
- 结构：案例背景 → 问题分析 → 解决方案 → 启示总结
- 分析要客观、深入""",

            FactType.REPORT: """【实习报告/工作报告】
- 基于真实的实习或工作经历
- 语言简洁明了，避免过度学术化
- 使用第一人称"我"，增强真实感
- 结构：前言 → 正文（工作内容）→ 收获与反思 → 结语
- 避免使用晦涩的术语，保持可读性""",

            FactType.APPLICATION: """【申请书】
- 必须真实反映个人情况和意愿
- 采用标准书信格式：标题 + 称呼 + 正文 + 结尾 + 署名 + 日期
- 语言真诚朴实，避免套话空话
- 结构：开头（表明意图）→ 正文（陈述理由）→ 结尾（表态）"""
        }

        # fact_type 是 FactType 枚举，直接使用作为键
        return guidance_map.get(fact_type, f"本写作任务类型为 {fact_type.value if fact_type else '未知'}，需要基于真实事实。")

    def get_key_entities_context(self) -> str:
        """
        获取关键实体上下文字符串

        Returns:
            格式化的关键实体上下文，用于注入 Prompt
        """
        if not self.key_info_tracker:
            return "【关键实体】无特定要求"

        entities = self.key_info_tracker.get_all_entities()
        requirements = self.key_info_tracker.get_all_requirements()

        if not entities and not requirements:
            return "【关键实体】无特定要求"

        parts = []
        if entities:
            parts.append(f"关键实体: {', '.join(entities)}")
            parts.append("注意：以上实体必须在生成内容中正确出现，不得遗漏或改名")
        if requirements:
            parts.append(f"格式要求: {', '.join(requirements)}")
            parts.append("注意：以上格式要求必须得到满足")

        return f"【关键信息追踪】\n" + "\n".join(parts)

    def check_key_info_coverage(self) -> Dict[str, Any]:
        """
        检查关键信息覆盖率

        Returns:
            覆盖率报告
        """
        if not self.key_info_tracker:
            return {"total": 0, "used": 0, "coverage": 1.0, "unused": []}
        return self.key_info_tracker.get_coverage_report()

    def clear(self) -> None:
        """清空会话（内存回收）"""
        self.node_registry.clear()
        self.covered_concepts_registry.clear()
        self._concept_set.clear()
        self.root_id = None
        self.global_target_words = 0
        self.global_abstract = ""
        self.style = None
        self.style_summary = ""
        self.original_prompt = ""
        self.key_info_tracker = None


# ============================================================================
# JSON 解析器 - robust_json_parse
# ============================================================================

def robust_json_parse(llm_output: str, target_model: type) -> BaseModel:
    """
    通用工业级 JSON 解析器

    功能：
    1. 剥离 Markdown 代码块护栏
    2. 定位第一个 { 或 [
    3. Pydantic 校验与反序列化
    4. 【增强】尝试修复截断的 JSON

    Args:
        llm_output: LLM 原始输出
        target_model: 目标 Pydantic 模型类

    Returns:
        解析后的 Pydantic 模型实例

    Raises:
        Exception: JSON 解析或类型校验失败
    """
    # 1. 剥离 Markdown 护栏
    cleaned = re.sub(r"^```(?:json)?", "", llm_output.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

    # 2. 尝试定位第一个 { 或 [
    match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)

    # 3. 清理 JSON 字符串
    cleaned = _clean_json_string(cleaned)

    # 4. 解析与 Pydantic 校验
    try:
        data_dict = json.loads(cleaned)
        return target_model(**data_dict)
    except json.JSONDecodeError as e:
        # 【增强】尝试修复截断的 JSON
        try:
            repaired = _repair_truncated_json(cleaned)
            data_dict = json.loads(repaired)
            return target_model(**data_dict)
        except Exception as repair_error:
            raise Exception(f"JSON 解析失败: {e}\n修复尝试也失败: {repair_error}\nRaw Output: {llm_output[:500]}")
    except Exception as e:
        raise Exception(f"Pydantic 校验失败: {e}\nRaw Output: {llm_output[:500]}")


def _clean_json_string(json_str: str) -> str:
    """清理 JSON 字符串

    处理的常见问题：
    1. 移除注释
    2. 移除尾随逗号
    3. 清理无效控制字符
    4. 修复缺少引号的字符串值
    """
    cleaned = json_str.strip()

    # 移除注释
    cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

    # 移除尾随逗号
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)

    # 清理无效控制字符（ASCII 0-31，除了已允许的 \t, \n, \r）
    # 这些控制字符在 JSON 字符串中会导致解析失败
    def clean_control_chars(match):
        char = match.group(0)
        # 保留制表符、换行符、回车符（它们在 JSON 中是有效的转义序列）
        if char in '\t\n\r':
            return char
        # 其他控制字符替换为空格
        return ' '

    # 匹配 ASCII 控制字符（0x00-0x1F），但不在转义序列中的
    cleaned = re.sub(r'[\x00-\x1f]', clean_control_chars, cleaned)

    # 修复缺少引号的字符串值
    # 问题示例：
    #   "title": 雷雨夜的拓印...  （值没有引号）
    #   "description": 赵希孟引导...  （值没有引号）
    cleaned = _fix_unquoted_string_values(cleaned)

    return cleaned.strip()


def _fix_unquoted_string_values(json_str: str) -> str:
    """
    修复 JSON 中缺少引号的字符串值

    问题场景：
    1. "key": value_without_quotes
    2. "key": value_with_trailing_quote"  （有结尾引号但无开头引号）
    3. "key": value_with_trailing_comma,

    策略：
    - 逐行处理，检测并修复缺少引号的字符串值
    - 排除对象 {、数组 [、数字开头的值
    """
    lines = json_str.split('\n')
    result_lines = []

    for line in lines:
        # 检测模式: "key": 后面跟的值
        match = re.search(r'^(\s*"([^"]+)"\s*:\s*)(.*)$', line)
        if match:
            prefix = match.group(1)  # "key": 部分（包含空白）
            key = match.group(2)
            rest = match.group(3)    # 值部分

            rest_stripped = rest.strip()

            # 排除不需要修复的情况
            # 1. 空值
            if not rest_stripped:
                result_lines.append(line)
                continue

            # 2. 对象 { 或数组 [ 开头
            if rest_stripped.startswith('{') or rest_stripped.startswith('['):
                result_lines.append(line)
                continue

            # 3. 数字开头
            if rest_stripped[0].isdigit() or rest_stripped[0] == '-':
                result_lines.append(line)
                continue

            # 4. 布尔值或 null
            if rest_stripped.startswith(('true', 'false', 'null')):
                result_lines.append(line)
                continue

            # 5. 值已经有正确的开头引号
            if rest_stripped.startswith('"'):
                result_lines.append(line)
                continue

            # 需要修复：值缺少开头引号
            # 情况: 值中间有一个孤立的双引号（原来的错误结尾引号）
            # 例如: Female Friendship and Social Networks as Sites of Resistance",
            # 或者: 维度裂隙的物理显化"  （值末尾有引号但缺少开头引号）
            orphan_quote_match = re.search(r'"[\s]*([,\}]|\Z)', rest)
            if orphan_quote_match:
                # 有孤立引号，值从开头到这个引号前
                value = rest[:orphan_quote_match.start()].strip()
                # 获取引号后的分隔符（逗号或右括号）
                separator = orphan_quote_match.group(1) if orphan_quote_match.group(1) else ''
                fixed_line = f'{prefix}"{value}"{separator}'
            else:
                # 没有孤立引号，找到逗号或右括号
                end_match = re.search(r'[,\}]', rest)
                if end_match:
                    value = rest[:end_match.start()].strip()
                    after = rest[end_match.start():]
                    fixed_line = f'{prefix}"{value}"{after}'
                else:
                    # 没有找到结束符，可能 JSON 截断
                    value = rest.strip()
                    # 移除末尾可能存在的孤立引号
                    if value.endswith('"'):
                        value = value[:-1].strip()
                    fixed_line = f'{prefix}"{value}"'

            result_lines.append(fixed_line)
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def robust_json_parse_list(llm_output: str, target_model: type) -> List[BaseModel]:
    """
    解析 JSON 数组为 Pydantic 模型列表

    Args:
        llm_output: LLM 原始输出
        target_model: 目标 Pydantic 模型类

    Returns:
        解析后的 Pydantic 模型实例列表
    """
    # 1. 剥离 Markdown 护栏
    cleaned = re.sub(r"^```(?:json)?", "", llm_output.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

    # 2. 尝试定位 [...]
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # 3. 清理
    cleaned = _clean_json_string(cleaned)

    # 4. 解析
    try:
        data_list = json.loads(cleaned)
        if not isinstance(data_list, list):
            raise ValueError("Expected a JSON array")
        return [target_model(**item) for item in data_list]
    except json.JSONDecodeError as e:
        raise Exception(f"JSON 数组解析失败: {e}\nRaw Output: {llm_output[:500]}")


def robust_json_parse_dict(llm_output: str, default: dict = None) -> dict:
    """
    鲁棒的 JSON 解析器（返回 dict）

    专门用于处理可能被截断的 LLM JSON 输出。

    特性：
    1. 尝试标准 JSON 解析
    2. 如果失败，尝试修复截断的 JSON（补充缺失的引号和括号）
    3. 返回解析后的 dict 或默认值

    Args:
        llm_output: LLM 原始输出
        default: 解析失败时的默认返回值，默认为空 dict

    Returns:
        解析后的字典
    """
    if default is None:
        default = {}

    if not llm_output or not llm_output.strip():
        return default

    # 1. 剥离 Markdown 护栏
    cleaned = re.sub(r"^```(?:json)?", "", llm_output.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

    # 2. 尝试定位第一个 JSON 对象
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # 3. 清理 JSON 字符串
    cleaned = _clean_json_string(cleaned)

    # 4. 尝试标准解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # 5. 尝试修复截断的 JSON
        try:
            repaired = _repair_truncated_json(cleaned)
            return json.loads(repaired)
        except:
            # 修复失败，返回默认值
            return default


def _repair_truncated_json(json_str: str) -> str:
    """
    尝试修复被截断的 JSON 字符串

    常见的截断情况：
    1. 字符串值被截断：{"key": "truncated val...
    2. 对象被截断：{"key": {"nested": "val...
    3. 数组被截断：{"arr": [1, 2, 3...

    修复策略：
    1. 计算未闭合的括号/引号数量
    2. 补充缺失的闭合符号
    """
    result = json_str.rstrip()

    # 如果已经以 } 结尾，可能已经完整
    if result.endswith('}') or result.endswith(']'):
        return result

    # 计算括号和引号状态
    brace_stack = []  # 追踪 { 和 [
    in_string = False
    escape_next = False

    i = 0
    while i < len(result):
        char = result[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            escape_next = True
            i += 1
            continue

        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '{':
                brace_stack.append('}')
            elif char == '[':
                brace_stack.append(']')
            elif char == '}':
                if brace_stack and brace_stack[-1] == '}':
                    brace_stack.pop()
            elif char == ']':
                if brace_stack and brace_stack[-1] == ']':
                    brace_stack.pop()
        i += 1

    # 如果字符串未闭合，添加闭合引号
    if in_string:
        result += '"'

    # 移除末尾可能的不完整键值对
    # 模式: "key": 或 "key": value_no_closing
    result = re.sub(r',?\s*"[^"]*"\s*:\s*$', '', result)
    result = re.sub(r',\s*$', '', result)

    # 补充缺失的闭合括号
    result += ''.join(reversed(brace_stack))

    return result


# ============================================================================
# 工具函数
# ============================================================================

def generate_node_id() -> str:
    """生成唯一节点 ID"""
    return f"node_{uuid.uuid4().hex[:8]}"


def detect_language(text: str) -> str:
    """
    检测文本语言（中文/英文）

    Args:
        text: 输入文本

    Returns:
        "zh" 或 "en"
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.sub(r'\s', '', text))

    if total_chars == 0:
        return "zh"

    chinese_ratio = chinese_chars / total_chars

    if chinese_ratio > 0.3:
        return "zh"
    else:
        return "en"


def validate_paragraph_format(text: str) -> bool:
    """
    校验最终输出是否符合 [Paragraph X - Main Point : ... - Word Count : ...] 格式

    只支持基本格式：
    [Paragraph X - Main Point : ... - Word Count : N]

    Args:
        text: 待校验文本

    Returns:
        是否符合格式
    """
    pattern = r"^\[Paragraph\s+\d+\s*-\s*Main Point\s*:\s*.+?\s*-\s*Word Count\s*:\s*\d+\]$"

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not re.match(pattern, line):
            return False
    return True


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ISRP 基础设施测试")
    print("=" * 70)

    # 测试 PlanSession
    print("\n[测试 1] PlanSession 数据结构")
    session = PlanSession()

    # 创建根节点
    root = OutlineNode(
        node_id="root",
        title="Root Node",
        description="This is the root",
        level=0,
        word_budget=10000
    )
    session.register_node(root)
    session.root_id = "root"

    # 创建子节点
    child1 = OutlineNode(
        node_id="child1",
        title="Child 1",
        description="First child",
        level=1,
        word_budget=5000,
        parent_id="root"
    )
    child2 = OutlineNode(
        node_id="child2",
        title="Child 2",
        description="Second child",
        level=1,
        word_budget=5000,
        parent_id="root",
        prev_sibling_id="child1"
    )
    session.register_node(child1)
    session.register_node(child2)

    # 更新父节点的 children_ids
    root.children_ids = ["child1", "child2"]

    # 测试概念注册
    session.add_to_concept_registry(["人工智能", "机器学习", "深度学习"])
    print(f"  概念黑名单: {session.global_concept_registry}")
    print(f"  '人工智能' 已使用: {session.is_concept_covered('人工智能')}")
    print(f"  '区块链' 已使用: {session.is_concept_covered('区块链')}")

    # 测试路径上下文
    path = child2.get_path_context(session.node_registry)
    print(f"  child2 路径上下文: {path}")

    # 测试 Pydantic Schema
    print("\n[测试 2] Pydantic Schemas")
    eval_result = RubricEvaluation(
        analysis="测试评估",
        relevance=5,
        accuracy=4,
        coherence=4,
        clarity=5,
        breadth_depth=4,
        reading_experience=4
    )
    print(f"  RubricEvaluation 平均分: {eval_result.average_score}")
    print(f"  整体通过: {eval_result.overall_passed}")

    # 测试 robust_json_parse
    print("\n[测试 3] robust_json_parse")
    dirty_json = """
    ```json
    {
        "thinking_process": "这是思考过程",
        "title": "测试标题",
        "description": "测试描述",
        "core_concepts": ["概念1", "概念2"],
        "needs_vertical_expansion": true
    }
    ```
    """
    try:
        result = robust_json_parse(dirty_json, NodeGenerationResult)
        print(f"  解析成功: {result.title}")
        print(f"  需要纵向扩展: {result.needs_vertical_expansion}")
    except Exception as e:
        print(f"  解析失败: {e}")

    # 测试格式校验
    print("\n[测试 4] 格式校验")
    valid_text = "[Paragraph 1 - Main Point : 测试内容 - Word Count : 500]"
    invalid_text = "Paragraph 1 - Main Point: 测试内容 - Word Count: 500"
    print(f"  有效格式: {validate_paragraph_format(valid_text)}")
    print(f"  无效格式: {validate_paragraph_format(invalid_text)}")

    # 清理会话
    session.clear()
    print("\n[测试 5] 会话清理完成")

    print("\n" + "=" * 70)
    print("ISRP 基础设施测试完成！")
    print("=" * 70)
