"""
ISRP — Iterative Self-Refining Planner for Long-Text Generation

三阶段树结构大纲规划与优化框架：
  Phase I   Preprocessing              查询压缩 + 风格识别
  Phase II  Outline Tree Generation     DFS 节点扩展 + 两级验证（章节检查点 + 全局验证）
  Phase III Outline Tree Optimization   抗遗忘注入 + BFS 字数分配

子模块:
  - schemas      核心数据结构 (OutlineNode, PlanSession)
  - engines      上下文引擎与字数归一化引擎
  - prompts      Prompt 模板统一管理
  - planner      核心规划编排器
  - steps        管线步骤处理器
  - utils        工具函数
  - evaluators   动态评估器 (ISRP 6 维 / WritingBench checklist)

用法:
    from isrp import SyncISRPPlanner
"""

from .schemas import (
    OutlineNode,
    PlanSession,
    NodeStatus,
    NodeGenerationResult,
    RubricEvaluation,
    ChapterCheckpointResult,
    BFSAllocationResult,
    NodeWeight,
    RootDivergenceResult,
    RootRefinementResult,
    GlobalAbstractResult,
    StyleExtractionResult,
    CompressedPrompt,
    PackageInfo,        # 分包信息
    PackageItem,        # 分包项
    PackageType,        # 分包类型枚举
    robust_json_parse,
    robust_json_parse_list,
    robust_json_parse_dict,
    validate_paragraph_format,
    generate_node_id,
    detect_language,
    LLMRole,
)

from .engines import (
    ContextEngine,
    WordAllocationEngine,
    MIN_LEAF_WORDS,
    MAX_LEAF_WORDS,
    MAX_DEPTH,
    MAX_WIDTH,
    PROMPT_COMPRESS_THRESHOLD,
    PROMPT_COMPRESS_TARGET,
    build_dynamic_expansion_prompt,
    get_temperature,
    extract_word_count,
    detect_package_info,         # 分包检测函数
    get_package_context,         # 分包上下文获取
)

# 新模块结构
from .prompts import PromptTemplates, MAX_ITERATIONS
from .planner import ISRPPlanner, SyncISRPPlanner

# 工具模块
from .utils import LLMClient, NodeOperations, TreeOperations, PromptCompressor

# 管线步骤处理器
from .steps import (
    QueryStyleRecognition,
    RootNodeEstablishment,
    GlobalAbstract,
    DFSExpansionEngine,
    ChapterLevelValidation,
    DescriptionOptimize,
    WordCountAllocation,
    GlobalValidation,
    Format,
)

# 评估模块
from .evaluators import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorFactory,
    ISRPEvaluator,
    WritingBenchEvaluator,
)

# 向后兼容：提供旧的类名别名
ISRPPlannerComplete = ISRPPlanner


__all__ = [
    # 核心数据结构
    "OutlineNode",
    "PlanSession",
    "NodeStatus",
    "NodeGenerationResult",
    "RubricEvaluation",
    "ChapterCheckpointResult",
    "BFSAllocationResult",
    "NodeWeight",
    "RootDivergenceResult",
    "RootRefinementResult",
    "GlobalAbstractResult",
    "StyleExtractionResult",
    "CompressedPrompt",
    "PackageInfo",       # 分包信息
    "PackageItem",       # 分包项
    "PackageType",       # 分包类型枚举
    # 引擎
    "ContextEngine",
    "WordAllocationEngine",
    # 规划器
    "ISRPPlanner",
    "SyncISRPPlanner",
    "ISRPPlannerComplete",  # 向后兼容
    # Prompt 模板
    "PromptTemplates",
    # 工具模块
    "LLMClient",
    "NodeOperations",
    "TreeOperations",
    "PromptCompressor",
    # 管线步骤处理器
    "QueryStyleRecognition",
    "RootNodeEstablishment",
    "GlobalAbstract",
    "DFSExpansionEngine",
    "ChapterLevelValidation",
    "DescriptionOptimize",
    "WordCountAllocation",
    "GlobalValidation",
    "Format",
    # 辅助函数
    "robust_json_parse",
    "robust_json_parse_list",
    "robust_json_parse_dict",
    "validate_paragraph_format",
    "generate_node_id",
    "detect_language",
    "build_dynamic_expansion_prompt",
    "get_temperature",
    "extract_word_count",
    "detect_package_info",     # 分包检测
    "get_package_context",     # 分包上下文
    # 常量
    "MIN_LEAF_WORDS",
    "MAX_LEAF_WORDS",
    "MAX_DEPTH",
    "MAX_WIDTH",
    "PROMPT_COMPRESS_THRESHOLD",
    "PROMPT_COMPRESS_TARGET",
    "MAX_ITERATIONS",
    "LLMRole",
    # 评估模块
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluatorFactory",
    "ISRPEvaluator",
    "WritingBenchEvaluator",
]