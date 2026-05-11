"""
ISRP Ablation — 消融实验版本

核心模块：
  - ablation_config: 消融实验开关配置
  - schemas: 核心数据结构
  - engines: 上下文引擎与字数归一化引擎
  - prompts: 提示词模板（从 isrp.prompts 导入）
  - planner: 核心规划器（消融版本）
  - steps: 步骤处理器
  - utils: 工具函数
  - evaluators: 动态评估模块

使用方法：
    from isrp_ablation import SyncISRPPlanner, set_ablation_config

    set_ablation_config('iteration')
    planner = SyncISRPPlanner(client)
    outline = planner.generate_plan("写一篇关于AI的文章", 5000)
"""

from .ablation_config import (
    set_ablation_config,
    is_ablation_iteration_enabled,
    is_ablation_anti_forget_enabled,
    is_ablation_delayed_constraint_enabled,
)

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
    PackageInfo,
    PackageItem,
    PackageType,
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
    PROMPT_COMPRESS_THRESHOLD,
    PROMPT_COMPRESS_TARGET,
    build_dynamic_expansion_prompt,
    get_temperature,
    extract_word_count,
    detect_package_info,
    get_package_context,
)

from .prompts import PromptTemplates, MAX_ITERATIONS
from .planner import ISRPPlanner, SyncISRPPlanner

# 工具模块
from .utils import LLMClient, NodeOperations, TreeOperations, PromptCompressor

# 步骤处理器
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

# 向后兼容别名
ISRPPlannerComplete = ISRPPlanner

__all__ = [
    # 消融配置
    "set_ablation_config",
    "is_ablation_iteration_enabled",
    "is_ablation_anti_forget_enabled",
    "is_ablation_delayed_constraint_enabled",
    # 核心数据结构
    "OutlineNode", "PlanSession", "NodeStatus",
    "NodeGenerationResult", "RubricEvaluation",
    "ChapterCheckpointResult", "BFSAllocationResult", "NodeWeight",
    "RootDivergenceResult", "RootRefinementResult", "GlobalAbstractResult",
    "StyleExtractionResult", "CompressedPrompt",
    "PackageInfo", "PackageItem", "PackageType",
    # 引擎
    "ContextEngine", "WordAllocationEngine",
    # 规划器
    "ISRPPlanner", "SyncISRPPlanner", "ISRPPlannerComplete",
    # 提示模板
    "PromptTemplates",
    # 工具
    "LLMClient", "NodeOperations", "TreeOperations", "PromptCompressor",
    # 步骤处理器
    "QueryStyleRecognition", "RootNodeEstablishment", "GlobalAbstract",
    "DFSExpansionEngine", "ChapterLevelValidation",
    "DescriptionOptimize", "WordCountAllocation",
    "GlobalValidation", "Format",
    # 辅助函数
    "robust_json_parse", "robust_json_parse_list", "robust_json_parse_dict",
    "validate_paragraph_format", "generate_node_id", "detect_language",
    "build_dynamic_expansion_prompt", "get_temperature",
    "extract_word_count", "detect_package_info", "get_package_context",
    # 常量
    "MIN_LEAF_WORDS", "MAX_LEAF_WORDS", "MAX_DEPTH",
    "PROMPT_COMPRESS_THRESHOLD", "PROMPT_COMPRESS_TARGET",
    "MAX_ITERATIONS", "LLMRole",
    # 评估模块
    "BaseEvaluator", "EvaluationResult", "EvaluatorFactory",
    "ISRPEvaluator", "WritingBenchEvaluator",
]
