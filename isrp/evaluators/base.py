"""
ISRP 评估器基类

定义评估器的统一接口（BaseEvaluator）和评估结果数据结构（EvaluationResult）。
支持不同评估体系的动态切换，新增评估器只需继承基类并实现 get_refine_prompt / get_checkpoint_prompt / get_dfs_eval_prompt / get_final_eval_prompt / parse_result 五个核心方法。

"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class EvaluationResult(BaseModel):
    """评估结果基类"""
    analysis: str = Field(default="", description="综合评估分析")
    average_score: float = Field(default=0.0, description="平均分")
    passed: bool = Field(default=False, description="是否通过阈值")
    raw_scores: Dict[str, int] = Field(default_factory=dict, description="各维度原始分数")
    extra: Dict[str, Any] = Field(default_factory=dict, description="扩展字段")


class EvalType(str, Enum):
    """评估类型枚举"""
    REFINE = "refine"          # 根节点建立 — 反思评估
    CHECKPOINT = "checkpoint"  # 章节检查点评估
    DFS_EVAL = "dfs_eval"      # DFS 节点评估


class BaseEvaluator(ABC):
    """
    评估器基类

    所有评估器必须实现此接口，以支持 planner_complete.py 的动态注入。
    """

    def __init__(self, threshold: float = 8.0):
        """
        初始化评估器

        Args:
            threshold: 通过阈值（10分制）
        """
        self.threshold = threshold

    @abstractmethod
    def get_refine_prompt(
        self,
        user_prompt: str,
        style_context: str,
        angle_name: str,
        rationale: str,
        high_level_structure: str,
        language: str = "zh"
    ) -> str:
        """
        生成根节点建立反思评估提示词

        Args:
            user_prompt: 用户原始指令
            style_context: 风格上下文
            angle_name: 切入角度名称
            rationale: 角度理由
            high_level_structure: 高层结构
            language: 语言代码 ("zh" 或 "en")

        Returns:
            完整的评估提示词
        """
        pass

    @abstractmethod
    def get_checkpoint_prompt(
        self,
        user_prompt: str,
        style_context: str,
        global_abstract: str,
        prev_chapter_abstract: str,
        chapter_content: str,
        language: str = "zh"
    ) -> str:
        """
        生成章节检查点评估提示词

        Args:
            user_prompt: 用户原始指令
            style_context: 风格上下文
            global_abstract: 全局摘要
            prev_chapter_abstract: 前序章节摘要
            chapter_content: 当前章节内容
            language: 语言代码 ("zh" 或 "en")

        Returns:
            完整的检查点评估提示词
        """
        pass

    @abstractmethod
    def get_dfs_eval_prompt(
        self,
        node_title: str,
        node_description: str,
        context: str,
        language: str = "zh"
    ) -> str:
        """
        生成 DFS 节点评估提示词

        Args:
            node_title: 节点标题
            node_description: 节点描述
            context: 上下文信息
            language: 语言代码 ("zh" 或 "en")

        Returns:
            完整的 DFS 节点评估提示词
        """
        pass

    @abstractmethod
    def get_final_eval_prompt(
        self,
        user_prompt: str,
        generated_text: str,
        language: str = "zh"
    ) -> str:
        """
        生成最终输出评估提示词

        Args:
            user_prompt: 用户原始指令
            generated_text: 生成的文本
            language: 语言代码 ("zh" 或 "en")

        Returns:
            完整的评估提示词
        """
        pass

    @abstractmethod
    def parse_result(self, llm_output: str, eval_type: EvalType) -> EvaluationResult:
        """
        解析 LLM 输出为评估结果

        Args:
            llm_output: LLM 原始输出
            eval_type: 评估类型

        Returns:
            解析后的评估结果
        """
        pass

    def get_threshold(self) -> float:
        """获取通过阈值"""
        return self.threshold

    def set_threshold(self, threshold: float) -> None:
        """设置通过阈值"""
        self.threshold = threshold

    @abstractmethod
    def get_dimensions(self) -> List[str]:
        """
        获取评分维度列表

        Returns:
            维度名称列表
        """
        pass

    def get_dimension_names(self, language: str = "zh") -> Dict[str, str]:
        """
        获取评分维度的本地化名称

        Args:
            language: 语言代码

        Returns:
            维度英文名到本地化名称的映射
        """
        return {}