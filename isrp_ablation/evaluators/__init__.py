"""
ISRP 评估器模块

支持两套评估体系：
  - ISRPEvaluator:         内部 6 维评分（relevance, accuracy, coherence, clarity, depth, readability）
                           用于迭代优化过程中的即时反馈
  - WritingBenchEvaluator:  外部 checklist 评估（与基准评测对齐）
                           每个样本 5 条实例化评估标准，LLM-as-a-Judge 范式

通过 EvaluatorFactory 根据配置动态创建。

"""
from .base import BaseEvaluator, EvaluationResult
from .factory import EvaluatorFactory
from .isrp_evaluator import ISRPEvaluator
from .writingbench_evaluator import WritingBenchEvaluator

__all__ = [
    'BaseEvaluator',
    'EvaluationResult',
    'EvaluatorFactory',
    'ISRPEvaluator',
    'WritingBenchEvaluator',
]